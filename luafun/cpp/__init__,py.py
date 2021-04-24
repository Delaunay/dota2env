from torch.utils.cpp_extension import load

mmcust = load(name="mmcust", sources=["mmcust.cpp"], verbose=True)


import torch.nn as nn

nn.Linear