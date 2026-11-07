"""Upstream causal convolution with FP32 trainable weights for the Llama student."""
import torch
from torch import nn
from torch.func import functional_call

from .logkv.attention import CausalConvBlock


class _DispatchConv(CausalConvBlock):
    def forward(self, hidden, cache=None, *, streaming=False):
        if streaming:
            return self.step(hidden, cache)
        return super().forward(hidden)


class LlamaCausalConv(nn.Module):
    """Keep upstream initialization/operations and immutable k-1 token history.

    As with LoRA, master parameters stay FP32. Differentiable casts also allow
    inference without an autocast context and FP64 reference comparisons.
    """

    def __init__(self, dim, kernel_size, *, device):
        super().__init__()
        self.block = _DispatchConv(dim, kernel_size).to(device=device, dtype=torch.float32)

    def forward(self, hidden, cache=None, *, streaming=False):
        weights = {name: p.to(hidden.dtype) for name, p in self.block.named_parameters()}
        return functional_call(self.block, weights, (hidden, cache),
                               {'streaming': streaming}, strict=True)
