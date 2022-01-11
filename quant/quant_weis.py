import torch
import numpy as np

from .uniform import *


##########################################################################################
####  Quantization of Weights 
##########################################################################################

def quant_checkpoint(checkpoint, weight_layers, bits,logleve=0):
    '''
    Quantize the weights per-channel per-layer for a pre-trained checkpoint
    '''

    print('quantizing weights into %s bits, %s layers' % (bits, len(weight_layers)))

    if bits == 0:
        return checkpoint, 0

    all_quant_error, all_quant_num = 0, 0
    all_tail_num = 0
    for each_layer in sorted(weight_layers):
        
        each_layer_weights = checkpoint[each_layer].clone()

        if logleve > 2 : print('quantize for: %s, size: %s' % (each_layer, each_layer_weights.size()))
        if logleve > 2 : print('weights range: (%.4f, %.4f)' % 
                            (torch.min(each_layer_weights), torch.max(each_layer_weights)))

        quant_error, quant_num, layer_tail_num = 0, 0, 0
        output_channel_num = each_layer_weights.size()[0]
        # channel-wise quant for each output channel
        for c in range(output_channel_num):  
            w = each_layer_weights[c, :].clone()
            qw, err  = quant_weights(w, bits)

            each_layer_weights[c, :] = qw
            quant_error += err
            quant_num += len(qw.reshape(-1, 1))

        all_quant_num += quant_num
        all_quant_error += quant_error

        checkpoint[each_layer] = each_layer_weights
        if logleve > 2 : print('layer quant RMSE: %.4e' % np.sqrt(quant_error / quant_num))
        if logleve > 2 : print('layer tail region percentage: %.2f' % (layer_tail_num / quant_num * 100))
        
    rmse = np.sqrt(all_quant_error / all_quant_num)
    print('\ntotal quant RMSE: %.4e' % rmse)
    print('toatl tail region percentage: %.2f' % (all_tail_num / all_quant_num * 100))

    return checkpoint, rmse


def quant_weights(w,bits):
    '''
    Quantize a tensor of weights 
    '''
    bkp_ratio = 0.0

    qw = uniform_symmetric_quantizer(w, bits=bits)

    # # bias correction
    # if args.bias_corr:
    #     mean_w, std_w = torch.mean(w), torch.std(w)
    #     mean_diff = mean_w - torch.mean(qw)
    #     std_ratio = torch.div(std_w, torch.std(qw) + 1e-12)
    #     if args.scale_bits > 0:
    #         scale_levels = 2 ** args.scale_bits
    #         mean_diff = torch.round(torch.mul(mean_diff, scale_levels)) / scale_levels
    #         std_ratio = torch.round(torch.mul(std_ratio, scale_levels)) / scale_levels

    #     qw = torch.mul(qw + mean_diff, std_ratio)

    err = float(torch.sum(torch.mul(qw - w, qw - w)))

    # abs_max = torch.max(torch.abs(w))
    # break_point = abs_max * bkp_ratio
    # tail_num = np.sum(torch.abs(w).detach().numpy() > float(break_point))

    return qw, err