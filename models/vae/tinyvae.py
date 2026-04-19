from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.activations import get_activation

import torch
import torch.nn as nn

@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution (空间卷积)
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 关键参数，使每个输入通道单独卷积
            bias=False
        )
        
        # Pointwise convolution (1x1卷积，通道变换)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AutoencoderTinyBlock(nn.Module):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.Tensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        act_fn = get_activation(act_fn)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))

class EncoderTiny(nn.Module):
    r"""
    The `EncoderTiny` layer is a simpler version of the `Encoder` layer.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderTinyBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        act_fn: str,
    ):
        super().__init__()

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]

            if i == 0:
                layers.append(DepthwiseSeparableConv(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(
                    DepthwiseSeparableConv(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

        layers.append(DepthwiseSeparableConv(block_out_channels[-1], out_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `EncoderTiny` class."""
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            # scale image from [-1, 1] to [0, 1] to match TAESD convention
            x = self.layers(x.add(1).div(2))

        return x


class DecoderTiny(nn.Module):
    r"""
    The `DecoderTiny` layer is a simpler version of the `Decoder` layer.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderTinyBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        upsampling_scaling_factor (`int`):
            The scaling factor to use for upsampling.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
    ):
        super().__init__()

        layers = [
            DepthwiseSeparableConv(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor, mode=upsample_fn))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                DepthwiseSeparableConv(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `DecoderTiny` class."""
        # Clamp.
        x = torch.tanh(x / 3) * 3

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            x = self.layers(x)

        # scale image from [0, 1] to [-1, 1] to match diffusers convention
        return x.mul(2).sub(1)
    
