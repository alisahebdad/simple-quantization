
import copy
from typing import Sequence
import torch
from torch.functional import Tensor
import torch.nn as nn
import os

from zmq import device 
#from .pwlq import *
from .uniform import *

##########################################################################################
####  Quantization of Activations 
##########################################################################################

class QuantAct(nn.Module):
    '''
    Quantize actications including:
    (1) the input of conv layer
    (2) the input of linear fc layer
    (3) the input of pooling layer
    '''
    def __init__(self, act_bits, get_stats, minv=None, maxv=None, 
        cali_sample_size=512, cali_batch_size=4, topk=10):
        '''
        cali_sample_size: calibration sample size, typically from random training data
        cali_batch_size: calibration sampling batch size
        topk: calibrate topk lower and upper bounds
        '''
        super(QuantAct, self).__init__()
        self.act_bits = act_bits
        self.get_stats = get_stats
        self.index = 0
        self.topk = topk
        self.sample_batches = cali_sample_size // cali_batch_size
        stats_size = (self.sample_batches, self.topk) if self.get_stats else 1
        self.register_buffer('minv', torch.zeros(stats_size))
        self.register_buffer('maxv', torch.zeros(stats_size))

    def forward(self, x):
        if self.get_stats:
            y = x.clone()
            y = torch.reshape(y, (-1,))
            y, indices = torch.sort(y)
            topk_mins = y[:self.topk]
            topk_maxs = y[-self.topk:]
            if self.index < self.sample_batches:
                self.minv[self.index, :] = topk_mins
                self.maxv[self.index, :] = topk_maxs
                self.index += 1

        if self.act_bits > 0:
            ## uniform quantization
            print (self.minv,'===')
            if self.minv is not None:
                if self.minv >= 0.0: # activation after relu
                    self.minv *= 0.0
                    self.signed = False
                else: 
                    self.maxv = max(-self.minv, self.maxv) 
                    self.minv = - self.maxv
                    self.signed = True
            x = uniform_symmetric_quantizer(x, bits=self.act_bits, 
                    minv=self.minv, maxv=self.maxv, signed=self.signed)
        return x


def quant_model_acts(model, act_bits, get_stats, cali_batch_size=4):
    """
    Add quantization of activations for a pretrained model recursively
    """
    if type(model) in [nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d]:
        quant_act = QuantAct(act_bits, get_stats, cali_batch_size=cali_batch_size)
        return nn.Sequential(quant_act, model)
    elif type(model) == nn.Sequential:
        modules = []
        for name, module in model.named_children():
            modules.append(quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return nn.Sequential(*modules)
    else:
        quantized_model = copy.deepcopy(model)
        for attribute in dir(model):
            module = getattr(model, attribute)
            if isinstance(module, nn.Module):
                setattr(quantized_model, attribute, 
                    quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return quantized_model


def simple_quant_mode_acts(model,bit=8,name='',mode='monitor'):

    if type(model) in [nn.Conv2d]:
        if mode == 'monitor':
            return nn.Sequential(model,SimpleLimit(name=name,mode='monitor'))
        if mode == 'qunatized_act':
            return nn.Sequential(SimpleQunAct(name=name),model,SimpleLimit(name=name))
    if type(model) == nn.Sequential:
        output = []
        num = 0
        for name_,modules in model.named_children():
            output.append(simple_quant_mode_acts(modules,name=(name+str(num)) ))
            num+=1
        return nn.Sequential(*output)
    

    quantized_model = copy.deepcopy(model)
    for attribute in dir(model):
        module = getattr(model, attribute)
        if isinstance(module, nn.Module):
            setattr(quantized_model, attribute,simple_quant_mode_acts(module,name=(name+attribute)))
    return quantized_model





class SimpleQunAct(nn.Module):
    allLayer = []
    def __init__(self,bit=8,mode='forward',function='None',name='') -> None:
        super(SimpleQunAct, self).__init__()
        self.bit = bit
        self.mode = mode
        self.func = function
        self.name = name
        self.maxQ = None
        self.minQ = None
        SimpleQunAct.allLayer.append(self)

    def __repr__(self):
        return super().__repr__()[:-1]+self.name+ '('+self.mode+'_'+self.func +'))'

    def load_min_max(self,addr):
        pass
        self.maxQ = torch.load(os.path.join(addr,self.name+'_max.pt'))
        self.minQ = torch.load(os.path.join(addr,self.name+'_min.pt'))



    def forward(self,x):
        if self.mode == 'forward':
            #print ('-'*100)
            #print (self.maxQ.shape)
            #print (x.shape)
            
            
            return x
        if self.mode == 'monitor':
            if self.func == 'Border':
                if self.maxQ is not None:
                    temp = torch.amax(x,dim=(0,2,3))
                    self.maxQ = torch.max(temp,self.maxQ)
                else:
                    self.maxQ = torch.amax(x,dim=(0,2,3))

                if self.minQ is not None:
                    temp = torch.amin(x,dim=(0,2,3))
                    self.minQ = torch.min(temp,self.minQ)
                else:
                    self.minQ = torch.amin(x,dim=(0,2,3))



                return x 
        if self.mode == 'quan':


            return x


class SimpleLimit(nn.Module):
    allLayer = []
    def __init__(self,mode='forward',name='') -> None:
        super(SimpleLimit, self).__init__()
        self.mode = mode
        self.name = name
        self.maxQ = None
        self.minQ = None
        SimpleLimit.allLayer.append((name,self))


    def save_min_max(self,path):
        min_addr = os.path.join(path,self.name+'_min.pt')
        max_addr = os.path.join(path,self.name+'_max.pt')
        torch.save(self.minQ,min_addr)
        torch.save(self.maxQ,max_addr)

    def load_min_max(self,path):
        min_addr = os.path.join(path,self.name+'_min.pt')
        max_addr = os.path.join(path,self.name+'_max.pt')
        self.minQ = torch.load(min_addr)
        self.maxQ = torch.load(max_addr)


    def setMode(self,newMode):
        self.mode = newMode

    def __repr__(self):
        return super().__repr__()[:-1]+self.name+'('+self.mode+'))'

    def forward(self,x):
        if self.mode == 'forward':
            return x
        if self.mode == 'monitor':
            if self.maxQ is not None:
                temp = torch.amax(x,dim=(0,2,3))
                self.maxQ = torch.max(temp,self.maxQ)
            else:
                self.maxQ = torch.amax(x,dim=(0,2,3))
            
            if self.minQ is not None:
                temp = torch.amin(x,dim=(0,2,3))
                self.minQ = torch.min(temp,self.minQ)
            else:
                self.minQ = torch.amin(x,dim=(0,2,3))
            return x 
        if self.mode == 'org_bound':
            for i in range(x.shape[1]):
                x[:,i,:,:] = torch.clamp(x[:,i,:,:],min=self.minQ[i],max=self.maxQ[i])

            return x

        if self.mode == 'limit_bound':
            for i in range(x.shape[1]):
                x[:,i,:,:] = torch.clamp(x[:,i,:,:],min=self.minQ[i]*self.limit,max=self.maxQ[i]*self.limit)

            return x


        if self.mode == 'quan':
            return x
