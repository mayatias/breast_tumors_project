import torch
import torch.nn as nn

class CoBlock(nn.Module):
  def __init__(self, in_channels, num_filters):
      super().__init__()
      conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1)
      batch_norm1 = nn.BatchNorm2d(num_filters)
      activation_layer1 = nn.ReLU(inplace=True)
      self.conv1 = nn.Sequential(conv1, batch_norm1, activation_layer1)
      # input to conv2 is x1, so channels = F
      conv2 = nn.Conv2d(in_channels=num_filters, out_channels=2 * num_filters, kernel_size=3, padding=1)
      batch_norm2 = nn.BatchNorm2d(2 * num_filters)
      activation_layer2 = nn.ReLU(inplace=True)
      self.conv2 = nn.Sequential(conv2, batch_norm2, activation_layer2)
      # input to conv3 is c1 = concat(x1, x2), so channels = F + 2F = 3F
      conv3 = nn.Conv2d(in_channels=3 * num_filters, out_channels=4 * num_filters, kernel_size=3, padding=1)
      batch_norm3 = nn.BatchNorm2d(4 * num_filters)
      activation_layer3 = nn.ReLU(inplace=True)
      self.conv3 = nn.Sequential(conv3, batch_norm3, activation_layer3)
  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    c1 = torch.cat([x1, x2], dim=1)
    x3 = self.conv3(c1)
    c_output = torch.cat([c1, x3], dim=1)
    return c_output


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=1)
        batch_norm1 = nn.BatchNorm2d(num_filters)
        activation_layer1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(conv1, batch_norm1, activation_layer1)
        conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        batch_norm2 = nn.BatchNorm2d(num_filters)
        activation_layer2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(conv2, batch_norm2, activation_layer2)
        conv3 = nn.Conv2d(in_channels=num_filters, out_channels=4 * num_filters, kernel_size=1)
        batch_norm3 = nn.BatchNorm2d(4 * num_filters)
        self.conv3 = nn.Sequential(conv3, batch_norm3)
        self.relu = nn.ReLU(inplace=True)
        # adjust the number of the input channels to match the residual branch dimensions before performing the element-wise addition
        conv = nn.Conv2d(in_channels=in_channels, out_channels=4 * num_filters, kernel_size=1)
        batch_norm = nn.BatchNorm2d(4 * num_filters)
        self.conv = nn.Sequential(conv, batch_norm)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv(x)
        out = x3 + x  # residual connection
        out = self.relu(out)
        return out

class MultiResBlock(nn.Module):
  def __init__(self, in_channels, num_filters):
    super(MultiResBlock, self).__init__()
    #main branch
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=3, padding=1)
    out_channels = num_filters + 2*num_filters + 4*num_filters
    self.bn_concat = nn.BatchNorm2d(out_channels)
    #shortcut branch, the original information bypasses the deep conv
    self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    # after concatenation
    self.relu = nn.ReLU()
    self.bn_out = nn.BatchNorm2d(out_channels)
  def forward(self, x):
    # path 1
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    concat = torch.cat([x1, x2, x3], dim=1)
    concat = self.bn_concat(concat)
    shortcut = self.conv_short(x) # path 2
    out = concat + shortcut # merge paths
    # activation & normalization
    out = self.relu(out)
    out = self.bn_out(out)
    return out

class Bottleneck(nn.Module):
  def __init__(self, in_channels, num_filters, dropout_p=0.3):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
    self.dropout = nn.Dropout2d(p=dropout_p)
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.dropout(x)
    return x