from pydoc import describe
from statistics import mode
import torchvision.models as models
import torch
from zmq import device
from fold_batch_norm import *
from quant import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import time
import math
import os 
import random
import argparse
import models as model_zoo
import copy
from Utility import *
from tqdm import tqdm


batch_size = 32
shuffle_option = False
save_max_min_act = False
load_max_min_act = False

save_max_min_act_addr = './stat'
def temp():
    return int(os.popen("nvidia-smi -q -d temperature | grep GPU | grep Cu | grep -Eo '[0-9]{1,4}'").read())


def safe_temp():
    if temp() > 79:
        print ('stop for temp')
        while temp() > 73:
            time.sleep(1)
        print ('start again')


def accuracy(output,target):
    #print ('accuracy')
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
    
    return x_int,min_v,max_v


def uniform_quantizer_conv2d(layer,bit=8):
    weight = layer.weight.data
    #layer.weight.bound = []
    setattr(layer.weight,'bound',[])
    #print (weight.shape)
    for c in range(weight.shape[0]):
        #print (weight[c])
        weight[c],min_v,max_v = unifrom_quantizer(weight[c],bit=bit)
        layer.weight.bound.append((min_v,max_v))
        #print (weight[c])
        #exit(0)


def quntize(model,kind='uniform',bit=8):
    i = 1
    for layer in model.modules():
        #print (type (layer))
        if isinstance(layer,nn.Conv2d):
            print ('Quantize conv2d',type(layer),i)
            i+=1
            uniform_quantizer_conv2d(layer,bit=bit)
            #print (layer.weight.bound,'+++')


def tensor_fault(tensor,rate,bit):
    #print (tensor.shape)
    safe_temp()
    q = list(tensor.shape)
    q.insert(0,bit)
     
    u = torch.rand(q)
    w = u<rate
    #print (tensor)
    #print ('-'*100)
    #print (w.nonzero())
    zeross = w.nonzero()
    l = (len(zeross)//100)+1
    p_bar = tqdm(range(len(zeross)))
    xx = 0
    for i in zeross:
        #print ('-'*100)
        #print (i[1:],i[0].item(),tensor[list(i[1:])])
        org = int(tensor[list(i[1:])].item())
        #print ('org:',org)
        
        if i[0] != (bit-1):

            if org>=0:
                if (org//(2**(i[0].item())))%2 == 1:
                    org-= 2**i[0].item()
                else:
                    org+=2**i[0].item()
            else:
                if (org//(2**(i[0].item())))%2 == 1:
                    org+= 2**i[0].item()
                else:
                    org-=2**i[0].item()

        else:
            org *= -1
        
        if xx%l == 0:
            p_bar.update(l)
            p_bar.refresh()
        tensor[list(i[1:])] = org
        xx+=1
        #print ('new data :',org)
    #print (tensor)


    #exit(0)
    return



    if len(list(tensor.shape)) == 1:
        #print ('____________',tensor)
        #exit(0)
        for i in range(len(tensor)):
            sign = -1 if tensor[i]<0 else 1
            b_ = list(bin(int(abs(tensor[i].item())))[2:])
            #print (tensor[i].item())            
            while len(b_) < (bit-1):
                b_.insert(0,'0')
            #print (b_)
            for w in range(bit-1):
                if random.random() < rate:
                    b_[w] = '0' if b_[w] == '1' else '1'
            if random.random() < rate:
                sign *= -1
            tensor[i] = int(''.join(b_),2)*sign

    else:
        for i in tensor:
            tensor_fault(i,rate,bit)
        



def fault_inject(weight,rate,bit,bound):
    #print (x.shape)
    #print (x)
    min_v = bound[0]#torch.min(x).item()
    max_v = bound[1]#torch.max(x).item()
    bound = max(abs(min_v),abs(max_v))
    #print (bound)
    #print (min_v,max_v)
    levels = 2 ** (bit-1)
    scale = bound/levels
    #x = torch.clamp(x,min=-bound,max=bound)
    #x_int = torch.round(x/scale)*scale
    level = torch.div(weight,scale)
    tensor_fault(level,rate,bit)

    #print (x_int)
    #print (x)
    x_fault = level*scale    

    return x_fault

def fault_inject2(weight,rate,bit,bound):
   
    levels = torch.rand_like(weight,device=torch.device('cuda:0'))

    for c in range(weight.shape[0]):
        
        
        
        min_v = bound[c][0]#torch.min(x).item()
        max_v = bound[c][1]#torch.max(x).item()
        bound_ = max(abs(min_v),abs(max_v))
        #print (bound)
        #print (min_v,max_v)
        levels_ = 2 ** (bit-1)
        scale = bound_/levels_
        #x = torch.clamp(x,min=-bound,max=bound)
        #x_int = torch.round(x/scale)*scale
        levels[c] = torch.div(weight[c],scale)

    tensor_fault(levels,rate,bit)

    for c in range(weight.shape[0]):
        min_v = bound[c][0]#torch.min(x).item()
        max_v = bound[c][1]#torch.max(x).item()
        bound_ = max(abs(min_v),abs(max_v))
        #print (bound)
        #print (min_v,max_v)
        levels_ = 2 ** (bit-1)
        scale = bound_/levels_
        levels[c] = torch.mul(levels[c],scale)        

    return levels





def fault_conv2d(layer,bit,rate):
    weight = layer.weight.data
    #print ('fault_conv2d',layer.weight.bound)
    #for c in range(weight.shape[0]):
    #    weight[c] = fault_inject(weight[c],rate,bit,layer.weight.bound[c])
    weight = fault_inject2(weight,rate,bit,layer.weight.bound)
    layer.weight.data = weight


def fault(model,rate,bit=8,args=None):
    i = 0
    # for layer in model.modules():
    #     if isinstance(layer,nn.Conv2d):
    #         print (layer.weight.shape)


    for layer in model.modules():
        #print (type (layer),'fault injecting ')
        if isinstance(layer,nn.Conv2d):
            if args is not None :
                if not args.colab:
                    safe_temp()
            else:
                safe_temp()
            
            print ('Quantize conv2d',type(layer),i)
            i+=1
            fault_conv2d(layer,bit,rate)


def validate(model,val_loader,criterion,args,datakeeper=None,method=''):
    pass
    
    model.eval()
    all_item = 0
    correct_item = 0
    wait_sleep = 0
    p_bar = tqdm(range(len(val_loader)))
    with torch.no_grad():
        for i , (images,target) in enumerate(val_loader):
            wait_sleep += 1
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
            #print (correct_item,all_item,all_item/len(val_loader),(correct_item/all_item)*100)
            if not args.colab:
                safe_temp()
            p_bar.update(1)
            p_bar.refresh()

        result_ = {
                'accuracy':str((correct_item/all_item)*100),
                'band':str(args.quantize_weight),
                'dataset':args.dataset,
                'method':method,
                'fault':str(args.fault),
                'arch':str(args.arch)
            }
        del p_bar

        if datakeeper is not None:
            datakeeper.addRow(row=result_)
        else:
            print (result_)
    
    
    
    


    if save_max_min_act:
        for i in model.modules():
            if type(i) in [SimpleQunAct]:
                if i.minQ is not None:
                    torch.save(i.minQ,os.path.join(save_max_min_act_addr,i.name+'_min.pt'))
                if i.maxQ is not None:
                    torch.save(i.maxQ,os.path.join(save_max_min_act_addr,i.name+'_max.pt'))
        exit(0)

def changeMode(model,new_mode,bit=8):
    if new_mode == 'Border':
        for i in model.modules():
            if type(i) in [SimpleQunAct]:
                #print ('*****',i)
                i.mode = 'monitor'
                i.func = 'Border'
    if new_mode == 'Forward':
        for i in model.modules():
            if type(i) in [SimpleQunAct]:
                #print ('*****',i)
                i.mode = 'forward'
                i.bit = bit


def load_max_min_act(model,addr):
    for i in model.modules():
        if type(i) in [SimpleQunAct]:
            i.load_min_max(addr)



def main():

    parser = argparse.ArgumentParser(description='Simple Quantize')
    parser.add_argument('--data', default='/home/alimohammad/Dataset/ILSVRC2012/val')
    parser.add_argument('-d','--dataset',default='cifar10',help='which dataset imagenet or cifar',choices=['cifar10','imagenet'])
    parser.add_argument('-a', '--arch', default='resnet50',help='network architecture',choices=['resnet50','vgg19','vgg11_bn'])
    parser.add_argument('-m','--model',default='/home/alimohammad/Models',help='model address')
    parser.add_argument('--quantize-weight',default=0,help='Weight Quantization with bit width param',type=int)
    parser.add_argument('--fault',default=0,help='fault rate ')
    parser.add_argument('--repeat',default=1,help='if repeate != 0 faulting and validating repeat ',type=int)
    parser.add_argument('--result',default='./result.csv',help='location of result find',type=str)
    parser.add_argument('--batch-size',default=32,help='batch size for validation ',type=int)
    parser.add_argument('--colab',action='store_true')
    parser.add_argument('--save-bound',help='save max and min',action='store_true') 
    parser.add_argument('--protector',help='protect against fault',choices=['org','limit']) 
    parser.add_argument('--limit',help='protect against fault with tight the  activations ',type=float) 



    args = parser.parse_args()
    print(str(args))

    myDatakeeper = DataKeeper(args.result,['accuracy','band','dataset','method','fault','arch'])
    batch_size = args.batch_size


    model = None
    criterion = None
    
    val_loader = None
    
    for i in range(args.repeat):
        print('-'*100,i)
        method = ''
        
        if args.dataset == 'cifar10':
            model_name = '{}_{}.pth'.format(args.arch,args.dataset)
            model_addr = os.path.join(args.model,model_name)
            model =  model_zoo.vgg.VGG('VGG19')#torch.load(model_addr)
            
            __ = torch.load(model_addr)['net']
            ___ = {}
            for i in __:
                ___[i[7:]] = __[i]



            model.load_state_dict(___)
            model.eval()
            criterion = nn.CrossEntropyLoss()

            print ('model load ...')

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR10(
                root=args.data, train=True, download=True, transform=transform_train)
            val_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(
                root=args.data, train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')





        if args.dataset == 'imagenet':
            model = models.__dict__[args.arch](pretrained=True)
            criterion = nn.CrossEntropyLoss()
            print ('model load')
            crop_size = 224
            scale = 0.875
            large_crop_size = int(round(crop_size / scale))
            valdir = args.data
            val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(large_crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True)
            testloader = val_loader
        if torch.cuda.is_available():
            model.cuda()

        #original_model = copy.deepcopy(model)
    
        
        #model = simple_quant_mode_acts(model,8)
        #model = simple_quant_mode_acts(model,8)
        #print (simple_quant_mode_acts(model,8))
        print ('-'*100)
        #changeMode(model,'Forward')
        #print (model)
        #load_max_min_act(model,'./stat')
        #model_org = copy.deepcopy(model)
            

        #model = copy.deepcopy(original_model)
        if args.quantize_weight != 0:

            quntize(model,bit=args.quantize_weight)
            print ('Quantized')
        if args.save_bound:
            pass
            # add simpleQuanAct at the end of each conv2d
            #print (model)
            model = simple_quant_mode_acts(model,args.quantize_weight,mode='monitor')
            #print (model)
            #print (SimpleLimit.allLayer)
            validate(model,testloader,criterion=criterion,args=args)

            for i in SimpleLimit.allLayer:
                print (i[0])
                i[1].save_min_max('./stat')
            exit(0)

        if args.protector:
            if args.protector == 'org':
                method += 'org'
                model = simple_quant_mode_acts(model,args.quantize_weight,mode='monitor')
                for i in SimpleLimit.allLayer:
                    #print (i[0])
                    i[1].load_min_max('./stat')
                    i[1].setMode('org_bound')
            if args.protector == 'limit':
                model = simple_quant_mode_acts(model,args.quantize_weight,mode='monitor')
                if not args.limit:
                    raise Exception('limit bound without limit rate ')
                method += 'limit' + str(args.limit)
                
                for i in SimpleLimit.allLayer:
                    #print (i[0])
                    setattr(i[1],'limit',args.limit)
                    i[1].load_min_max('./stat')
                    i[1].setMode('limit_bound')


        if args.fault !=0:
            fault(model,float(args.fault),args.quantize_weight,args=args)

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        validate(model,testloader,criterion=criterion,datakeeper=myDatakeeper,args=args,method=method)
        
    
    
    #print (model)
    return model



if __name__=='__main__': 
    model = main()





