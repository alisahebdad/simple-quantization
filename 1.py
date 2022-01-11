import torchvision.models as models
import torch
from fold_batch_norm import *
from quant import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time 
#resnet50



def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    global stop_after

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available() is not None:
                images = images.cuda( non_blocking=True)
            target = target.cuda( non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


model_arch = 'resnet50'
batch_size = 16
shuffle_option = False



model = models.__dict__['resnet50'](pretrained=True)

checkpoint = model.state_dict()
print('----- pretrained model loaded -----')

#print (checkpoint.keys())

#print (model)

checkpoint, weight_layers = fold_batch_norm(checkpoint, arch=model_arch)
#print (checkpoint['layer4.2.conv3.weight'][0:20,0])
#print (weight_layers)
checkpoint, rmse = quant_checkpoint(checkpoint, weight_layers,bits=8)

#print (checkpoint['layer4.2.conv3.weight'][0:20,0])
#print (weight_layers)
model.load_state_dict(checkpoint)
del checkpoint


model = quant_model_acts(model,8.0, True, 16)
model = model.cuda()

#print (model)

criterion = nn.CrossEntropyLoss().cuda()

# load data
crop_size = 224
scale = 0.875
large_crop_size = int(round(crop_size / scale))

traindir = '/home/alimohammad/Dataset/ILSVRC2012/train'
valdir = '/home/alimohammad/Dataset/ILSVRC2012/val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])



val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(large_crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=shuffle_option,
    num_workers=8, pin_memory=True)


validate(val_loader, model, criterion )






