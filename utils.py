
import math
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms


def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)
