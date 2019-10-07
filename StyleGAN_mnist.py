# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torch import optim
from torchvision.utils import save_image
import utils
import torch.cuda.comm
import torch.cuda.nccl
import dlutils.pytorch.count_parameters as count_param_override
import lod_driver
from dataloader import *
from model import Model
from net import *
from tracker import LossTracker
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from tqdm import tqdm
from launcher import run
from defaults import get_cfg_defaults
import dlutils.download
import dlutils.batch_provider
import dlutils.reader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def save_sample(lod2batch, tracker, sample, x, logger, model, cfg, discriminator_optimizer, generator_optimizer):
    os.makedirs('results', exist_ok=True)

    logger.info('\n[%d/%d] - ptime: %.2f, %s, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        discriminator_optimizer.param_groups[0]['lr'],
        generator_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(lod2batch.lod, lod2batch.get_blend_factor(), z=sample)

        @utils.async_func
        def save_pic(x, x_rec):
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            #x_rec = F.interpolate(x_rec, 128)
            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            save_image(result_sample, os.path.join(cfg.OUTPUT_DIR,
                                                   'sample_%d_%d.jpg' % (
                                                       lod2batch.current_epoch + 1,
                                                       lod2batch.iteration // 1000)
                                                   ), nrow=16)

        save_pic(x, x_rec)


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=1)
    model.cuda(local_rank)
    model.train()

    if local_rank == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=1)
        del model_s.discriminator
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        generator = model.module.generator
        discriminator = model.module.discriminator
        mapping = model.module.mapping
        dlatent_avg = model.module.dlatent_avg
    else:
        generator = model.generator
        discriminator = model.discriminator
        mapping = model.mapping
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(generator)

    logger.info("Trainable parameters discriminator:")
    count_parameters(discriminator)

    arguments = dict()
    arguments["iteration"] = 0

    generator_optimizer = LREQAdam([
        {'params': generator.parameters()},
        {'params': mapping.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    discriminator_optimizer = LREQAdam([
        {'params': discriminator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'generator': generator_optimizer,
                                    'discriminator': discriminator_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32)

    model_dict = {
        'discriminator': discriminator,
        'generator': generator,
        'mapping': mapping,
        'dlatent_avg': dlatent_avg
    }

    if local_rank == 0:
        model_dict['generator_s'] = model_s.generator
        model_dict['mapping_s'] = model_s.mapping

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'generator_optimizer': generator_optimizer,
                                    'discriminator_optimizer': discriminator_optimizer,
                                    'tracker': tracker
                                },
                                scheduler=scheduler,
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = generator.layer_to_resolution

    dlutils.download.mnist()
    mnist = dlutils.reader.Mnist('mnist').items
    mnist = np.asarray([x[1] for x in mnist], np.float32)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    sample = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(mnist))

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [generator_optimizer, discriminator_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, dataset size: %d" % (lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                lod2batch.lod,
                                                                2 ** lod2batch.get_lod_power2(),
                                                                2 ** lod2batch.get_lod_power2(),
                                                                len(mnist)))

        dlutils.shuffle.shuffle_ndarray(mnist)

        r = 2 ** lod2batch.get_lod_power2()
        mnist_ = F.interpolate(torch.tensor(mnist).view(mnist.shape[0], 1, 28, 28), r).detach().cpu().numpy()

        scheduler.set_batch_size(32)

        model.train()

        need_permute = False

        class BatchCollator(object):
            def __init__(self, device=torch.device("cpu")):
                self.device = device

            def __call__(self, batch):
                with torch.no_grad():
                    x = batch
                    x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                    return x

        batches = dlutils.batch_provider(mnist_, lod2batch.get_per_GPU_batch_size(), BatchCollator(local_rank), report_progress=False)

        with torch.autograd.profiler.profile(use_cuda=True, enabled=False) as prof:
            for x_orig in tqdm(batches):
                if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x_orig = x_orig.permute(0, 3, 1, 2)
                x_orig = (x_orig / 127.5 - 1.)

                blend_factor = lod2batch.get_blend_factor()

                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig

                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, needed_resolution)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

                discriminator_optimizer.zero_grad()
                loss_d = model(x, lod2batch.lod, blend_factor, d_train=True)
                tracker.update(dict(loss_d=loss_d))
                loss_d.backward()
                discriminator_optimizer.step()

                if local_rank == 0:
                    betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                    model_s.lerp(model, betta)

                generator_optimizer.zero_grad()
                loss_g = model(x, lod2batch.lod, blend_factor, d_train=False)
                tracker.update(dict(loss_g=loss_g))
                loss_g.backward()
                generator_optimizer.step()

                lod2batch.step()
                if local_rank == 0 and lod2batch.is_time_to_report():
                    save_sample(lod2batch, tracker, sample, x, logger, model_s, cfg, discriminator_optimizer, generator_optimizer)

        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        if local_rank == 0:
            save_sample(lod2batch, tracker, sample, x, logger, model_s, cfg, discriminator_optimizer, generator_optimizer)
            if epoch > 2:
                checkpointer.save("model_tmp")

        scheduler.step()

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_mnist.yaml',
        world_size=gpu_count)
