import unittest
import tempfile
import checkpointer
import torch
from torch import nn
import logging
import os
import copy


class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)


class Cfg:
    def __init__(self):
        self.OUTPUT_DIR = tempfile.mkdtemp()


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


if __name__ == '__main__':
    unittest.main()
