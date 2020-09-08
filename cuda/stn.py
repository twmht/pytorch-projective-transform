import math
from torch import nn
from torch.autograd import Function
import torch

import stn_cuda

torch.manual_seed(42)


class STNFunction(Function):
    @staticmethod
    def forward(ctx, input, weights):
        print (input.dtype)
        print (weights.dtype)
        outputs = stn_cuda.forward(input, weights)
        new_h, source = outputs[:2]
        ctx.save_for_backward(input, source)

        return new_h

    @staticmethod
    def backward(ctx, grad_input):
        old_input, source = ctx.saved_tensors
        outputs = stn_cuda.backward(old_input.contiguous(), source.contiguous(), grad_input.contiguous())
        d_old_input, d_old_weight = outputs
        return d_old_input, d_old_weight


class STN(nn.Module):
    def __init__(self, output_h, output_w):
        super(STN, self).__init__()
        self.output_h = output_h
        self.output_w = output_w

    def forward(self, input, weight):
        return STNFunction.apply(input, weight)

#  if __name__ == '__main__':
    #  device = torch.device("cuda")
    #  m = STN(112, 112)

    #  kwargs = {'dtype': torch.float32,
              #  'device': device,
              #  'requires_grad': True}

    #  X = torch.randn(10, 3, 128, 128, **kwargs)
    #  W = torch.randn(10, 8, **kwargs)
    #  new_h = m(X,W)
    #  #  print (new_h.shape)
    #  loss = new_h.sum()
    #  loss.backward()
    #  print (loss)
