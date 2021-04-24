from torch.utils.cpp_extension import load

mmcust = load(name="mmcust", sources=["mmcust.cpp"], verbose=True)
