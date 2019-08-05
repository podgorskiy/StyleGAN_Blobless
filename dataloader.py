import pickle

import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data

cpu = torch.device('cpu')


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, logger, rank=0, data_train=None):
        self.cfg = cfg
        self.logger = logger
        self.last_data = ""
        if not data_train:
            self.data_train = []
            self.switch_fold(rank, 0)
        else:
            self.data_train = data_train

    def switch_fold(self, rank, lod):
        data_to_load = self.cfg.DATASET.PATH % (rank % self.cfg.DATASET.FOLD_COUNT, lod)

        if data_to_load != self.last_data:
            self.last_data = data_to_load
            self.logger.info("Switching data!")

            with open(data_to_load, 'rb') as pkl:
                self.data_train = pickle.load(pkl)

        self.logger.info("Train set size: %d" % len(self.data_train))
        self.data_train = self.data_train[:4 * (len(self.data_train) // 4)]
        self.data_train = np.asarray(self.data_train, dtype=np.uint8)

    def __getitem__(self, index):
        return self.data_train[index]

    def __len__(self):
        return len(self.data_train)


class BatchCollator(object):
    def __call__(self, batch):
        with torch.no_grad():
            x = np.asarray(batch, dtype=np.float32)
            x = torch.tensor(x, requires_grad=True, device=torch.device("cpu"))
            #x = x.pin_memory()
            assert not x.is_cuda
            #assert x.is_pinned()
            return x

