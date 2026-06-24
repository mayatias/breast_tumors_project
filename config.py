output_channels_e1 = 16 + 32 + 64
output_channels_e2 = 32 + 64 + 128
output_channels_e3 = 4*64 # maybe will be change- x3 & x don't have the same number of channels
output_channels_e4 = 128 + 256 + 512
output_channels_e5 = 256 + 512 + 1024 # maybe will be change- c_tilde & x_tilde don't have the same number of channels
e_blocks_output_channels = {"e1": output_channels_e1, "e2": output_channels_e2,
                            "e3": output_channels_e3, "e4": output_channels_e4,
                            "e5": output_channels_e5}
num_filters_e_blocks = {"e1": 16, "e2": 32, "e3": 64, "e4": 128, "e5": 256, "e6": 512}
