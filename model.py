import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from config import e_blocks_output_channels

class CResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_conv = nn.Conv2d(in_channels=e_blocks_output_channels["e1"], out_channels=1, kernel_size=1)
    def forward(self, x):
        o_e6, O_e_skips, O_me_skips = self.encoder(x)
        o_d1 = self.decoder(o_d6=o_e6, O_e_skips=O_e_skips, O_me_skips=O_me_skips)
        out = self.final_conv(o_d1)
        return out