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
from model import Model
from net import *
from checkpointer import Checkpointer

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging
import sys
import bimpy
import lreq


lreq.use_implicit_lreq.set(True)

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
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count= cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=3)
    model.eval()

    logger.info("Trainable parameters generator:")
    count_parameters(model.generator)

    model_dict = {
        'generator_s': model.generator,
        'mapping_fl_s': model.mapping,
        'dlatent_avg': model.dlatent_avg,
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=True)

    checkpointer.load()

    ctx = bimpy.Context()
    remove = bimpy.Bool(False)
    layers = bimpy.Int(7)

    ctx.init(1800, 1600, "Styles")

    rnd = np.random.RandomState(5)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
    sample = torch.tensor(latents).float().cuda()

    def update_image(sample):
        with torch.no_grad():
            torch.manual_seed(0)
            model.eval()
            x_rec = model.generate(layers.value, remove.value, z=sample)
            #model.generator.set(l.value, c.value)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]

            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    with torch.no_grad():
        save_image(model.generate(7, True, z=sample) * 0.5 + 0.5, 'sample_removed.png')
        save_image(model.generate(7, False, z=sample) * 0.5 + 0.5, 'sample_not_removed.png')

    im = bimpy.Image(update_image(sample))
    while(not ctx.should_close()):
        with ctx:

            bimpy.set_window_font_scale(2.0)

            if bimpy.checkbox('REMOVE BLOB', remove):
                im = bimpy.Image(update_image(sample))
            if bimpy.button('NEXT'):
                latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
                sample = torch.tensor(latents).float().cuda()
                im = bimpy.Image(update_image(sample))
            if bimpy.slider_int("Layers", layers, 0, 8):
                im = bimpy.Image(update_image(sample))
            bimpy.image(im, bimpy.Vec2(1024, 1024))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN blobless")
    parser.add_argument(
        "--config-file",
        default="configs/experiment_ffhq.yaml",
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
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    sample(cfg, logger)
