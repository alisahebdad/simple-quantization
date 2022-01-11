import torch
import numpy as np


##########################################################################################
####  Uniform quantization
##########################################################################################

# References: 
#   https://arxiv.org/abs/1806.08342
#   https://arxiv.org/abs/1712.05877 
def uniform_symmetric_quantizer(x, bits=8.0, minv=None, maxv=None, signed=True, 
                                scale_bits=0.0, num_levels=None, scale=None, simulated=True):
    if minv is None:
        maxv = torch.max(torch.abs(x))
        minv = - maxv if signed else 0

    if signed:
        maxv = np.max([-float(minv), float(maxv)])
        minv = - maxv 
    else:
        minv = 0
    
    if num_levels is None:
        num_levels = 2 ** bits

    if scale is None:
        scale = (maxv - minv) / (num_levels - 1)

    if scale_bits > 0:
        scale_levels = 2 ** scale_bits
        scale = torch.round(torch.mul(scale, scale_levels)) / scale_levels
            
    ## clamp
    x = torch.clamp(x, min=float(minv), max=float(maxv))
        
    x_int = torch.round(x / scale)
    
    if signed:
        x_quant = torch.clamp(x_int, min=-num_levels/2, max=num_levels/2 - 1)
        assert(minv == - maxv)
    else:
        x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
        assert(minv == 0 and maxv > 0)
        
    x_dequant = x_quant * scale
    
    return x_dequant if simulated else x_quant
