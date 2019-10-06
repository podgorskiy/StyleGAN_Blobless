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
from torchvision.utils import save_image
from model import Model
from net import *
from checkpointer import Checkpointer

from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging
import sys
import lreq

im_size = 128

lreq.use_implicit_lreq.set(False)


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
    del model.discriminator
    model.eval()

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
            'dlatent_avg': model.dlatent_avg,
        }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=True)

    file_name = 'karras2019stylegan-ffhq'

    checkpointer.load(file_name=file_name + '.pth')

    rnd = np.random.RandomState(5)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
    sample = torch.tensor(latents).float().cuda()

    with torch.no_grad():
        model.eval()
        images = []
        for i in range(100):
            image = model.generate(model.generator.layer_count - 1, 1, z=sample)

            resultsample = (image * 0.5 + 0.5)
            images.append(resultsample)

    resultsample = torch.stack(images).mean(0)

    save_image(images[0], 'test_individual.png')
    save_image(resultsample, 'test_average.png')

    # with torch.no_grad():
    #     model.eval()
    #     image = model.generate(model.generator.layer_count - 1, 1, z=sample) * 0.5 + 0.5
    #     save_image(image, 'test_expected.png')



if __name__ == '__main__':
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
