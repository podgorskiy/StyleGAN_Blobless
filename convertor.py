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

import os
import sys
import argparse
import logging
import torch
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from defaults import get_cfg_defaults
from StyleGAN import train
from torch import distributed
import dnnlib
import dnnlib.tflib
import dnnlib.tflib as tflib
import pickle
from model import Model
#import stylegan
#import stylegan.training.networks_stylegan
import numpy as np
from torchvision.utils import save_image
import PIL
from dnnlib import EasyDict
from checkpointer import Checkpointer


def save_sample(model, sample):
    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample.png', nrow=16)

        save_pic(x_rec)


def load_from(name, cfg):
    dnnlib.tflib.init_tf()
    with open(name, 'rb') as f:
        m = pickle.load(f)

    Gs = m[2]

    #Gs_ = tflib.Network('G', func_name='stylegan.training.networks_stylegan.G_style', num_channels=3, resolution=1024)

    #Gs_.copy_vars_from(Gs)

    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count= cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        truncation_psi=0.7, #cfg.MODEL.TRUNCATIOM_PSI,
        channels=3)

    for i in range(cfg.MODEL.MAPPING_LAYERS):
        weight = Gs.trainables['G_mapping/Dense%d/weight' % i]
        bias = Gs.trainables['G_mapping/Dense%d/bias' % i]

        block = getattr(model.mapping, "block_%d" % (i + 1))

        block.fc.weight[:] = torch.tensor(np.transpose(weight.eval(), (1, 0)))
        block.fc.bias[:] = torch.tensor(bias.eval())

    model.dlatent_avg.buff[:] = torch.tensor(Gs.vars['dlatent_avg'].eval())

    model.generator.const[:] = torch.tensor(Gs.trainables['G_synthesis/4x4/Const/const'].eval())

    model.generator.decode_block[0].noise_weight_1[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/4x4/Const/Noise/weight'].eval())
    model.generator.decode_block[0].noise_weight_2[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/4x4/Conv/Noise/weight'].eval())
    model.generator.decode_block[0].conv_2.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/4x4/Conv/weight'].eval(), (3, 2, 0, 1)))
    model.generator.decode_block[0].bias_1[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/4x4/Const/bias'].eval())
    model.generator.decode_block[0].bias_2[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/4x4/Conv/bias'].eval())
    model.generator.decode_block[0].style_1.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/4x4/Const/StyleMod/weight'].eval(), (1, 0)))
    model.generator.decode_block[0].style_1.bias[:] = torch.tensor(Gs.trainables['G_synthesis/4x4/Const/StyleMod/bias'].eval())
    model.generator.decode_block[0].style_2.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/4x4/Conv/StyleMod/weight'].eval(), (1, 0)))
    model.generator.decode_block[0].style_2.bias[:] = torch.tensor(Gs.trainables['G_synthesis/4x4/Conv/StyleMod/bias'].eval())
    model.generator.to_rgb[0].to_rgb.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/ToRGB_lod8/weight'].eval(), (3, 2, 0, 1)))
    model.generator.to_rgb[0].to_rgb.bias[:] = torch.tensor(Gs.trainables['G_synthesis/ToRGB_lod8/bias'].eval())

    for i in range(1, model.generator.layer_count):
        name = '%dx%d' % (2**(2+i), 2**(2+i))
        model.generator.decode_block[i].noise_weight_1[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv0_up/Noise/weight' % name].eval())
        model.generator.decode_block[i].noise_weight_2[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv1/Noise/weight' % name].eval())

        if model.generator.decode_block[i].fused_scale:
            w = Gs.trainables['G_synthesis/%s/Conv0_up/weight' % name].eval()
            w = np.transpose(w, (0, 1, 3, 2))
            w = np.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='constant')
            w = w[1:, 1:] + w[:-1, 1:] + w[1:, :-1] + w[:-1, :-1]

            model.generator.decode_block[i].conv_1.weight[:] = torch.tensor(np.transpose(w, (3, 2, 0, 1)))
        else:
            model.generator.decode_block[i].conv_1.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/%s/Conv0_up/weight' % name].eval(), (3, 2, 0, 1)))

        model.generator.decode_block[i].conv_2.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/%s/Conv1/weight' % name].eval(), (3, 2, 0, 1)))
        model.generator.decode_block[i].bias_1[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv0_up/bias' % name].eval())
        model.generator.decode_block[i].bias_2[0, :, 0, 0] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv1/bias' % name].eval())
        model.generator.decode_block[i].style_1.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/%s/Conv0_up/StyleMod/weight' % name].eval(), (1, 0)))
        model.generator.decode_block[i].style_1.bias[:] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv0_up/StyleMod/bias' % name].eval())
        model.generator.decode_block[i].style_2.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/%s/Conv1/StyleMod/weight' % name].eval(), (1, 0)))
        model.generator.decode_block[i].style_2.bias[:] = torch.tensor(Gs.trainables['G_synthesis/%s/Conv1/StyleMod/bias' % name].eval())

        model.generator.to_rgb[i].to_rgb.weight[:] = torch.tensor(np.transpose(Gs.trainables['G_synthesis/ToRGB_lod%d/weight' % (8-i)].eval(), (3, 2, 0, 1)))
        model.generator.to_rgb[i].to_rgb.bias[:] = torch.tensor(Gs.trainables['G_synthesis/ToRGB_lod%d/bias' % (8-i)].eval())
    return model #, Gs_


def train_net(args):
    torch.cuda.set_device(0)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sample = torch.randn(1, cfg.MODEL.LATENT_SPACE_SIZE).view(-1, cfg.MODEL.LATENT_SPACE_SIZE)

    model = load_from('karras2019stylegan-ffhq-1024x1024.pkl', cfg)
    #model, Gs = load_from('karras2019stylegan-ffhq-1024x1024.pkl', cfg)

    # Generate image.
    #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    #images = Gs.run(sample.cpu().detach().numpy(), None, truncation_psi=0.7, randomize_noise=True, output_transform=None)

    save_sample(model, sample, )

    #png_filename = os.path.join('example.png')
    #PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


    model_dict = {
        'generator_s': model.generator,
        'mapping_s': model.mapping,
        'dlatent_avg': model.dlatent_avg,
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=True)

    checkpointer.save('karras2019stylegan-ffhq')


def run():
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
    parser.add_argument(
        "--config-file",
        default="configs/experiment_stylegan.yaml",
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

    train_net(args)


if __name__ == '__main__':
    run()
