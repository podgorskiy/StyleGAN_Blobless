import unittest
import tempfile
import checkpointer
import torch
from torch import nn
import logging
import os
import copy
from dataloader import TFRecordsDataset, make_dataloader
import pickle


class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)


class Cfg:
    def __init__(self):
        self.OUTPUT_DIR = tempfile.mkdtemp()
        self.DATASET  = lambda: None
        self.DATASET.SIZE = 70000
        self.DATASET.PART_COUNT = 16
        self.DATASET.FFHQ_SOURCE = '/data/datasets/ffhq-dataset/tfrecords/ffhq/ffhq-r%02d.tfrecords'
        self.DATASET.PATH = '/data/datasets/ffhq-dataset/tfrecords/ffhq/splitted/ffhq-r%02d.tfrecords.%03d'
        self.DATASET.MAX_RESOLUTION_LEVEL = 10


class CheckpointerTest(unittest.TestCase):
    def test_initialization_loading_saving(self):
        cfg = Cfg()
        model = DummyNetwork()
        logger = logging.getLogger("logger")
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)

        ch.load()

        ch.save('model').wait()

        model2 = DummyNetwork()
        ch2 = checkpointer.Checkpointer(cfg, {'dummynet': model2}, logger=logger)
        ch2.load()
        self.assertTrue(torch.all(list(model.parameters())[0] == list(model2.parameters())[0]))

    def test_initialization_from_scratch(self):
        cfg = Cfg()
        logger = logging.getLogger("logger")
        model = DummyNetwork()
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)
        ch.save("model").wait()
        os.remove(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'))

        model = DummyNetwork()
        tmp = copy.deepcopy(model)
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)
        ch.load()
        self.assertTrue(torch.all(list(tmp.parameters())[0] == list(model.parameters())[0]))

    def test_initialization_from_file(self):
        cfg = Cfg()
        logger = logging.getLogger("logger")
        model = DummyNetwork()
        tmp = copy.deepcopy(model)
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)
        ch.save("model").wait()
        os.remove(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'))

        model = DummyNetwork()
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)
        ch.load()
        self.assertFalse(torch.all(list(tmp.parameters())[0] == list(model.parameters())[0]))

        model = DummyNetwork()
        ch = checkpointer.Checkpointer(cfg, {'dummynet': model}, logger=logger)
        ch.load(file_name=os.path.join(cfg.OUTPUT_DIR, "model.pth"))
        self.assertTrue(torch.all(list(tmp.parameters())[0] == list(model.parameters())[0]))

    def test_tf_record_reading(self):
        cfg = Cfg()
        logger = logging.getLogger("logger")
        dataset1 = TFRecordsDataset(cfg, logger, rank=0, world_size=4, buffer_size_mb=1)
        dataset2 = TFRecordsDataset(cfg, logger, rank=1, world_size=4, buffer_size_mb=1)
        dataset3 = TFRecordsDataset(cfg, logger, rank=2, world_size=4, buffer_size_mb=1)
        dataset4 = TFRecordsDataset(cfg, logger, rank=3, world_size=4, buffer_size_mb=1)

        for r, b in [(2, 32), (3, 64)]:
            dataset1.reset(2, 32)
            dataset2.reset(2, 32)
            dataset3.reset(2, 32)
            dataset4.reset(2, 32)
            dataset = set()

            for i in list(dataset1):
                for img in i[0]:
                    assert img.tobytes() not in dataset
                    dataset.add(img.tobytes())

            for i in list(dataset2):
                for img in i[0]:
                    assert img.tobytes() not in dataset
                    dataset.add(img.tobytes())

            for i in list(dataset3):
                for img in i[0]:
                    assert img.tobytes() not in dataset
                    dataset.add(img.tobytes())

            for i in list(dataset4):
                for img in i[0]:
                    assert img.tobytes() not in dataset
                    dataset.add(img.tobytes())

            self.assertTrue(len(dataset) == 70000)

    def test_tf_record_reading2(self):
        cfg = Cfg()
        logger = logging.getLogger("logger")
        dataset1 = TFRecordsDataset(cfg, logger, rank=0, world_size=2, buffer_size_mb=1024)
        dataset2 = TFRecordsDataset(cfg, logger, rank=1, world_size=2, buffer_size_mb=1024)

        torch.cuda.init()

        for r, b in [(2, 32), (3, 64)]:
            dataset1.reset(2, 32)
            dataset2.reset(2, 32)

            l1 = make_dataloader(cfg, logger, dataset1, 32, 0)
            l2 = make_dataloader(cfg, logger, dataset2, 32, 1)

            dataset = set()

            for i in list(l1):
                for img in i:
                    b = img.cpu().detach().numpy().tobytes()
                    assert b not in dataset
                    dataset.add(b)

            for i in list(l2):
                for img in i:
                    b = img.cpu().detach().numpy().tobytes()
                    assert b not in dataset
                    dataset.add(b)

            self.assertTrue(len(dataset) == 70000)

            ordering = dict()
            dataset1.reset(2, 32)
            l1 = make_dataloader(cfg, logger, dataset1, 32, 0)

            k = 0
            for i in list(l1):
                for img in i:
                    b = img.cpu().detach().numpy().tobytes()
                    ordering[b] = k
                    k += 1

            dataset1.reset(2, 32)
            l1 = make_dataloader(cfg, logger, dataset1, 32, 0)

            k = 0
            for i in list(l1):
                for img in i:
                    b = img.cpu().detach().numpy().tobytes()
                    print(ordering[b])
                    k += 1
                    if k > 100:
                        break


if __name__ == '__main__':
    unittest.main()
