import os 
import random 
import torch 
import numpy as np 
import logging


class NestedTensor:

    def __init__(self, tensors, mask):
        self.tensors = tensors 
        self.mask = mask 

    def to(self, device):
        cast_tensors = self.tensors.to(device)
        if self.mask is not None:
            cast_mask = self.mask.to(device)
        else:
            cast_mask = None 

        return NestedTensor(cast_tensors, cast_mask)

    def decompose(self):
        return self.tensors, self.mask 

    def __repr__(self):
        return str(self.tensors)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def create_logger(log_path):
    # logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(level=logging.INFO)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
