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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    distributed.destroy_process_group()


def train_net(rank, world_size, args):
    if world_size > 1:
        setup(rank, world_size)
    torch.cuda.set_device(rank)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(args)

    logger.info("Using {} GPUs".format(world_size))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args.distributed = world_size > 1
    train(cfg, rank, world_size, args.distributed, logger)


def run(fn, world_size):
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
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

    try:
        if world_size > 1:
            mp.spawn(fn,
                     args=(world_size, args),
                     nprocs=world_size,
                     join=True)
        else:
            train_net(0, world_size, args)
    finally:
        if world_size > 1:
            cleanup()


if __name__ == '__main__':
    run(train_net, torch.cuda.device_count())
