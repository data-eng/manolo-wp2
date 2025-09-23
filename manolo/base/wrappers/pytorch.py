from manolo.base.wrappers import version_test
import torch; version_test(torch)

from torch import Tensor

import torch.nn as torch_nn
import torch.nn.utils.prune as prune
import torch.nn.functional as nn_functional

import torch.optim as torch_optim

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
