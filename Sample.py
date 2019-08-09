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

from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from dataloader import PickleDataset, BatchCollator
from model import Model
from net import *
from tracker import LossTracker
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging

im_size = 128


def process_batch(batch):
    data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size : (y + 1) * im_size, x * im_size : (x + 1) * im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(8, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT // 8,
        layer_count= cfg.MODEL.LAYER_COUNT + 3,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=0.7, #cfg.MODEL.TRUNCATIOM_PSI,
        channels=3)
    del model.discriminator
    model.eval()

    torch.cuda.manual_seed_all(110)

    logger.info("Trainable parameters generator:")
    count_parameters(model.generator)

    if False:
        model_dict = {
            'generator': model.generator,
            'mapping': model.mapping,
            'dlatent_avg': model.dlatent_avg,
        }
    else:
        model_dict = {
            'generator_s': model.generator,
            'mapping_s': model.mapping,
            #'dlatent_avg': model.dlatent_avg,
        }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=True)

    file_name = 'model_final_small'

    # checkpointer.load(file_name=file_name + '.pth')
    # checkpointer.save(file_name + '_stripped')

    sample_b = torch.randn(1, cfg.MODEL.LATENT_SPACE_SIZE).view(-1, cfg.MODEL.LATENT_SPACE_SIZE)

    for i in range(500):
        if i % 20 == 0:
            sample_a = sample_b
            sample_b = torch.randn(1, cfg.MODEL.LATENT_SPACE_SIZE).view(-1, cfg.MODEL.LATENT_SPACE_SIZE)
        x = (i % 20) / 20.0
        sample = sample_a * (1.0 - x) + sample_b * x
        save_sample(model, sample, i)

    exit()

    im_count = 16
    canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])
    cut_layer_b = 0
    cut_layer_e = 2

    styles = model.mapping(sample)
    styles = list(styles.split(1, 1))

    for i in range(im_count):
        torch.cuda.manual_seed_all(110)
        style = [x[i] for x in styles]
        style = torch.cat(style, dim=0)[None, ...]
        rec = model.generator.decode(style, cfg.MODEL.LAYER_COUNT - 1, 0.7)
        place(canvas, rec[0], 1, 2 + i)

        place(canvas, rec[0], 2 + i, 1)

    for i in range(im_count):
        for j in range(im_count):
            style_a = [x[i] for x in styles[:cut_layer_b]]
            style_b = [x[j] for x in styles[cut_layer_b:cut_layer_e]]
            style_c = [x[i] for x in styles[cut_layer_e:]]
            style = style_a + style_b + style_c
            torch.cuda.manual_seed_all(110)
            style = torch.cat(style, dim=0)[None, ...]
            rec = model.generator.decode(style, cfg.MODEL.LAYER_COUNT - 1, 0.7)
            place(canvas, rec[0], 2 + i, 2 + j)

    save_image(torch.Tensor(canvas), 'reconstruction.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
    parser.add_argument(
        "--config-file",
        default="configs/experiment_small.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    sample(cfg, logger)
