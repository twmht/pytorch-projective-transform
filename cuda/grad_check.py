from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

from stn import STNFunction

device = torch.device("cuda")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

X = torch.randn(10, 3, 128, 128, **kwargs)
W = torch.randn(10, 8, **kwargs)

print (X.dtype)
print (W.dtype)


variables = [X, W]


if gradcheck(STNFunction.apply, variables):
    print('Ok')
