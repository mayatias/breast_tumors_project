import torch
import torch.nn as nn
from Blocks import CoBlock, IdentityBlock, MultiResBlock, Bottleneck
from config import e_blocks_output_channels, num_filters_e_blocks

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    # E1: CoBlock (num of filter = 16)
    self.e1 = CoBlock(in_channels=1, num_filters=num_filters_e_blocks["e1"])
    self.pool1 = nn.MaxPool2d(2)
    output_channels_e1 = e_blocks_output_channels["e1"]
    # E2: CoBlock (num of filter = 32)
    self.e2 = CoBlock(in_channels=output_channels_e1, num_filters=num_filters_e_blocks["e2"])
    self.pool2 = nn.MaxPool2d(2)
    output_channels_e2 = e_blocks_output_channels["e2"]
    # E3: IdentityBlock (num of filter = 64)
    self.e3 = IdentityBlock(in_channels=output_channels_e2, num_filters=num_filters_e_blocks["e3"])
    self.pool3 = nn.MaxPool2d(2)
    output_channels_e3 = e_blocks_output_channels["e3"]
    # E4: CoBlock (num of filter = 128)
    self.e4 = CoBlock(in_channels=output_channels_e3, num_filters=num_filters_e_blocks["e4"])
    self.pool4 = nn.MaxPool2d(2)
    output_channels_e4 = e_blocks_output_channels["e4"]
    # E5: MultiResBlock (num of filter = 256)
    self.e5 = MultiResBlock(in_channels=output_channels_e4, num_filters=num_filters_e_blocks["e5"])
    self.pool5 = nn.MaxPool2d(2)
    output_channels_e5 = e_blocks_output_channels["e5"]
    # E6: BottleNeck (num of filter = 512)
    self.e6 = Bottleneck(in_channels=output_channels_e5, num_filters=num_filters_e_blocks["e6"], dropout_p=0.3)

  def forward(self, x):
    O_e_skips = []
    O_me_skips = []
    # E1
    O_e1 = self.e1(x)
    O_e_skips.append(O_e1)
    O_me1 = self.pool1(O_e1)
    O_me_skips.append(O_me1)
    # E2
    O_e2 = self.e2(O_me1)
    O_e_skips.append(O_e2)
    O_me2 = self.pool2(O_e2)
    O_me_skips.append(O_me2)
    # E3
    O_e3 = self.e3(O_me2)
    O_e_skips.append(O_e3)
    O_me3 = self.pool3(O_e3)
    O_me_skips.append(O_me3)
    # E4
    O_e4 = self.e4(O_me3)
    O_e_skips.append(O_e4)
    O_me4 = self.pool4(O_e4)
    O_me_skips.append(O_me4)
    # E5
    O_e5 = self.e5(O_me4)
    O_e_skips.append(O_e5)
    O_me5 = self.pool5(O_e5)
    O_me_skips.append(O_me5)
    # E6 (bottleneck)
    O_e6 = self.e6(O_me5)
    return O_e6, O_e_skips, O_me_skips
