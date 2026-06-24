import torch
import torch.nn as nn
from Blocks import CoBlock, IdentityBlock, MultiResBlock
from config import e_blocks_output_channels, num_filters_e_blocks

class DecoderBlock(nn.Module):
  def __init__(self, block_type, in_channels, skip_channels, out_channels):
    super().__init__()
    # upsampling
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
    # block type (matching to the encoder)
    if block_type == "CoBlock":
      self.block = CoBlock(in_channels + skip_channels, out_channels)
    elif block_type == "IdentityBlock":
      self.block = IdentityBlock(in_channels + skip_channels, out_channels)
    elif block_type == "MultiResBlock":
      self.block = MultiResBlock(in_channels + skip_channels, out_channels)

  def forward(self, o_d_i_plus_1, concat_skip=None, residual_skip=None, o_me_i_minus_1=None):
    # upsampling and conv layer
    o_d_i_plus_1 = self.upsample(o_d_i_plus_1)
    o_d_i_plus_1 = self.conv(o_d_i_plus_1)
    # build fb
    if concat_skip is not None and o_me_i_minus_1 is not None:
      fb = torch.cat([concat_skip, o_me_i_minus_1, o_d_i_plus_1], dim=1)
    elif concat_skip is not None:
      fb = torch.cat([concat_skip, o_d_i_plus_1], dim=1)
    else:
      fb = o_d_i_plus_1
    # process block
    fb = self.block(fb)
    if residual_skip is not None:
      out = fb + residual_skip
    else:
      out = fb
    return out

class Decoder(nn.Module):
  def __init__(self, output_channels=None):
    super().__init__()
    if output_channels is None:
      output_channels = e_blocks_output_channels
    skip_channels5 = output_channels["e5"] # D5 has 2 inputs: O_e(5) & O_d(6)
    skip_channels4 = output_channels["e4"] + output_channels["e3"] # D4 has 3 inputs: O_e(4) & O_me(3) & O_d(5)
    skip_channels3 = 0 # D3 has one input: O_d(4)
    skip_channels2 = output_channels["e2"] + output_channels["e1"] # D4 has 3 inputs: O_e(2) & O_me(1) & O_d(3)
    skip_channels1 = output_channels["e1"] # D1 has 2 inputs: O_e(1) & O_d(2)
    # D5 (E5: MultiRes)
    self.d5 = DecoderBlock(block_type="MultiResBlock", in_channels=num_filters_e_blocks["e6"], skip_channels=skip_channels5, out_channels=num_filters_e_blocks["e5"])
    # D4 (E4: CoBlock)
    self.d4 = DecoderBlock(block_type="CoBlock", in_channels=output_channels["e5"], skip_channels=skip_channels4, out_channels=num_filters_e_blocks["e4"])
    # D3 (E3: IdentityBlock)
    self.d3 = DecoderBlock(block_type="IdentityBlock", in_channels=output_channels["e4"], skip_channels=skip_channels3, out_channels=num_filters_e_blocks["e3"])
    # D2 (E2: CoBlock)
    self.d2 = DecoderBlock(block_type="CoBlock", in_channels=output_channels["e3"], skip_channels=skip_channels2, out_channels=num_filters_e_blocks["e2"])
    # D1 (E1: CoBlock)
    self.d1 = DecoderBlock(block_type="CoBlock", in_channels=output_channels["e2"], skip_channels=skip_channels1, out_channels=num_filters_e_blocks["e1"])

  def forward(self, o_d6, O_e_skips, O_me_skips):
    # unpack encoder skips
    o_e1, o_e2, o_e3, o_e4, o_e5 = O_e_skips
    o_me1, o_me2, o_me3, o_me4, o_me5 = O_me_skips
    o_d5 = self.d5(o_d_i_plus_1=o_d6, concat_skip=o_e5, residual_skip=o_e5) # D5 has 2 inputs: O_e(5) & O_d(6)
    o_d4 = self.d4(o_d_i_plus_1=o_d5, concat_skip=o_e4, residual_skip=o_e4, o_me_i_minus_1=o_me3) # D4 has 3 inputs: O_e(4) & O_me(3) & O_d(5)
    o_d3 = self.d3(o_d_i_plus_1=o_d4, residual_skip=o_e3) # D3 has one input: O_d(4)
    o_d2 = self.d2(o_d_i_plus_1=o_d3, concat_skip=o_e2, residual_skip=o_e2, o_me_i_minus_1=o_me1) # D4 has 3 inputs: O_e(2) & O_me(1) & O_d(3)
    o_d1 = self.d1(o_d_i_plus_1=o_d2, concat_skip=o_e1, residual_skip=o_e1) # D1 has 2 inputs: O_e(1) & O_d(2)
    return o_d1