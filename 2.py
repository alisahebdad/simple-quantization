import torchvision.models as models
import torch
from fold_batch_norm import *
from quant import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import math

def accuracy(output,target):
    print ('accuracy')
    with torch.no_grad():
        _,pred = output.topk(1,1,True,True)
        #print (pred.view(-1),target)
        #print (pred.view(-1).eq(target))
        #print (torch.sum(pred.view(-1).eq(target)).item())
        return torch.sum(pred.view(-1).eq(target)).item()


def unifrom_quantizer(x,bit=8,):
    #print (x.shape)
    #print (x)
    min_v = torch.min(x).item()
    max_v = torch.max(x).item()
    bound = max(abs(min_v),abs(max_v))
    #print (bound)
    #print (min_v,max_v)
    levels = 2 ** (bit-1)
    scale = bound/levels
    #x = torch.clamp(x,min=-bound,max=bound)
    x_int = torch.round(x/scale)*scale
    #print (x_int)
    #print (x)
    
    return x_int


def uniform_quantizer_conv2d(layer,bit=8):
    weight = layer.weight.data
    #print (weight.shape)
    for c in range(weight.shape[0]):
        #print (weight[c])
        weight[c] = unifrom_quantizer(weight[c],bit=bit)
        #print (weight[c])
        #exit(0)


def quntize(model,kind='uniform',bit=8):
    for layer in model.modules():
        #print (type (layer))
        if isinstance(layer,nn.Conv2d):
            print ('Quantize conv2d',type(layer))
            uniform_quantizer_conv2d(layer,bit=bit)






def validate(model,val_loader,criterion):
    pass
    
    model.eval()
    all_item = 0
    correct_item = 0

    with torch.no_grad():
        for i , (images,target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(images)
            #print ('image',images.shape)
            #print ('output',output.shape)
            #print (target,output)
            #print ('target',target.shape)
            loss = criterion(output,target)
            #print ('[remove] :',loss)            
            correct_item += accuracy(output,target)
            all_item += images.shape[0]
            print (correct_item,all_item,(correct_item/all_item)*100)
            if all_item>10000:
                break

model_arch = 'vgg11_bn'
batch_size = 8
shuffle_option = False




def main():

    model = models.__dict__[model_arch](pretrained=True)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()



    # load data 
    crop_size = 224
    scale = 0.875
    large_crop_size = int(round(crop_size / scale))
    valdir = '/home/alimohammad/Dataset/ILSVRC2012/val'
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(large_crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)


    #print (model)
    #quntize(model,bit=6)
    print (model)
    model = simple_quant_mode_acts(model,8)
#    model = simple_quant_mode_acts(model,8)
    #print (simple_quant_mode_acts(model,8))
    print ('-'*100)
    print (model)
    validate(model,val_loader,criterion=criterion)
    #print (model)
    return model



if __name__=='__main__':
    model = main()





