#!/usr/bin/env python3
import torch

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda version:', torch.version.cuda)
    print('device:', torch.cuda.get_device_name(0))
    try:
        print('bf16 supported:', torch.cuda.is_bf16_supported())
    except Exception as e:
        print('bf16 supported: error', e)
