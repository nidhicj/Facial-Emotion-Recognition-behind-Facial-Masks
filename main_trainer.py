import argparse 
import os,sys,shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from prettytable import PrettyTable
# from raf_db_dataset import ImageList

import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch AffectNet Training using novel attention+region branches')

parser.add_argument('--root_path', type=str, default='../data/AffectNetdataset/Manually_Annotated_Images_aligned/',
                    help='path to root path of images')
parser.add_argument('--DB', type=str, default='ddd',
                    help='Which Database for train. (Flatcam, FERPLUS)')

parser.add_argument('--main_folder', type=str, default='/projects/joshi/projects/', help = 'where to save the docs')
parser.add_argument('--title', type=str, default='Nothing', help = 'title for the graph')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=121, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='checkpoints_affectnet7', type=str)

parser.add_argument('--train_list', type=str, default = '../data/Affectnetmetadata/training.csv',
                    help='path to training list')
parser.add_argument('--valid_list', type=str, default =  '../data/Affectnetmetadata/validation.csv',
                    help='path to validation list')
parser.add_argument('--test_list', type=str, default =  '../data/Affectnetmetadata/validation.csv',
                    help='path to test list')

parser.add_argument('--train_landmarksfile', type=str, default = '../data/Affectnetmetadata/training_affectnet_landmarks_scores.pkl',
                    help='path to landmarksdictionary')
parser.add_argument('--test_landmarksfile', type=str, default = '../data/Affectnetmetadata/validation_affectnet_landmarks_scores.pkl',
                    help='path to landmarksdictionary')


parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--end2end', default=True,help='if true, using end2end with dream block, else, using naive architecture')

parser.add_argument('--num_classes', type=int, default=8,
                    help='number of expressions(class)')

parser.add_argument('--num_regions', type=int, default=20,
                    help='number of regions(crops)')

args = parser.parse_args()
best_prec1 = 0
# f = open('/projects/joshi/projects/GACNN/logs/logger'+args.title+'.txt','a')
train_losses = []
validation_losses = []
test_losses = []

train_accu = []
validation_accu = []
test_accu = []

if args.DB == 'A':
    print('DB: A')
    from dataset_affectnet import ImageList
if args.DB == 'C':
    print('DB: C')
    from dataset_ckplus import ImageList
if args.DB == 'R':
    print('DB: R')
    from dataset_raf_db import ImageList

from model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        #print(self.indices)    
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        #print(self.num_samples)              
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            #print(label)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type)
        #pdb.set_trace()
        if dataset_type is ImageList:
            return dataset.imgList[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples  


def main():
    global args, best_prec1
    args = parser.parse_args()
    print('Device state:', device)
    print('\nFER on {} using GACNN\n\n'.format(args.DB))
    #print('img_dir:', args.root_path)

    CM = torch.zeros(args.num_classes,args.num_classes)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    the_last_loss = 1000
    patience = 3
    trigger_times = 0
    imagesize = args.imagesize
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),            
            transforms.Resize((args.imagesize, args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    
    valid_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    train_data = ImageList(root=args.root_path ,landmarksfile=args.train_landmarksfile, fileList=args.train_list,
                  transform=train_transform ,num_classes= args.num_classes)

    valid_data = ImageList(root=args.root_path ,landmarksfile=args.train_landmarksfile, fileList=args.valid_list,
                  transform=train_transform ,num_classes= args.num_classes)

    test_data = ImageList(root=args.root_path, landmarksfile=args.test_landmarksfile,fileList=args.test_list,
                  transform=valid_transform,num_classes= args.num_classes)

    train_sampler = ImbalancedDatasetSampler(train_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, drop_last = True , sampler=train_sampler)    
    
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_t, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)

    print('length of  train Database for training: ' + str(len(train_loader.dataset)))
    print('length of  valid Database for validation training: ' + str(len(valid_loader.dataset)))
    print('length of  test Database: ' + str(len(test_loader.dataset)))


    
    print("prepare model")
    basemodel = torch.nn.DataParallel(Net(num_classes=args.num_classes)).to(device)
    count_parameters(basemodel)
    
    criterion = nn.CrossEntropyLoss().to(device)

    criterion1 = criterion#MarginLoss(loss_lambda=0.5).to(device)
    
    optimizer1 =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            basemodel.load_state_dict(checkpoint['state_dict'])
            
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

      
    print('Training starting:\n')
    # print("Dont know what comes next")
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0:
           adjust_learning_rate(optimizer1, epoch)

        # train for one epoch
        # print("------checkhere------")
        model = train(train_loader, basemodel, criterion,  optimizer1,  epoch)
        
        the_current_loss = validation(valid_loader,basemodel,criterion,optimizer1,epoch)#model, device, valid_loader, loss_function)


        print('The current loss:', int(the_current_loss))
        print('The Last loss: ',int(the_last_loss))

        if int(the_current_loss) >= int(the_last_loss):
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break

        # else:
        #     print('trigger times: 0')
        #     trigger_times = 0

        the_last_loss = the_current_loss # return model
        
   
    print("Testing started")
    prec1, CM = test(test_loader, basemodel, criterion, optimizer1,  epoch, args.num_classes,args.epochs )
    print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
        
    ''' 
   # remember best prec@1 and save checkpoint
    f1 = plt.figure()
    f2 = plt.figure()

    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)

    ax1.plot(train_losses,'-o')
    ax1.plot(validation_losses,'-o')
    ax1.plot(test_losses,'-o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('losses')
    ax1.legend(['Train','Valid','Test'])
    ax1.set_title(args.title+'Train, Valid and Test Losses')
    os.chdir(args.main_folder)
    f1.savefig('Train, Valid and Test Losses')
    

    ax2.plot(train_accu,'-o')
    ax2.plot(validation_accu,'-o')
    ax2.plot(test_accu,'-o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Train','Test'])
    ax2.set_title(args.title+'Train and Test Accuracy')
    os.chdir(args.main_folder)
    f2.savefig('Train and Test Accuracy')
    '''

    confusion_matrix_new = confusion_matrix(CM)
    if args.num_classes == 7:
        squad = ['Neutral','Happy','Angry','Surprise','Sad','Fear','Disgust']
    if args.num_classes == 4: 
        squad = ['Neutral','Happy','Angry','Surprise']

    df_cm = pd.DataFrame(confusion_matrix_new, index = [i for i in squad],columns = [i for i in squad])

    plt.figure(figsize = (10,10))
    plt.title(args.title, fontsize =20)
    cm_plot = sn.heatmap(df_cm, annot=True)
    os.chdir(args.main_folder)
    figure = cm_plot.get_figure()    
    figure.savefig(args.title+'ConfusionMatrix', dpi=600)


    is_best = prec1 > best_prec1
    best_prec1 = max(prec1.to(device).item(), best_prec1)
    
    save_checkpoint({
        'epoch': epoch + 1,            
        'state_dict': basemodel.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer1.state_dict(),
    }, is_best.item())

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

        
def confusion_matrix(CM):
    stacks = []
    for i, row in enumerate(CM):
        new_row = []
        for j, item in enumerate(row):
            item = item / CM.sum(-1)[i]
            new_row.append(item)
        stacks.append(new_row)
    confusion_matrix_new = np.vstack(stacks)

    
    return confusion_matrix_new


def train(train_loader,  model,  criterion, optimizer1, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()

    # top1 = 0
    # loss = 0
    
    the_last_loss = 100
    
    # switch to train mode
    model.train()

    end = time.time()
    

    for i, (input, target, landmarks) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        
        target = target.to(device)
        # print(len(target))
        # compute output
        preds = model(input, landmarks)
        prec = accuracy(preds, target, topk=(1,))
        loss1 = criterion(preds, target) 
        loss.update(loss1.item(), input.size(0))
        top1.update(prec[0], input.size(0))

        running_loss =+ loss1.item() * input.size(0)
        
    
        # loss += loss1.item()
        # top1 += prec[0]
         # compute gradient and do SGD step
        optimizer1.zero_grad()     
        loss1.backward()
        optimizer1.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))
                   #data_time=data_time, loss=losses,  top1=top1))
    train_losses.append(running_loss / len(train_loader))
    train_accu.append(prec[0].detach().cpu().clone().numpy())

def validation(valid_loader,model,criterion,optimizer1,epoch):# device, , loss_function):
    # Settings
    model.eval()
    loss_total = 0
    loss_values = []
    # loss = AverageMeter()
    # Test validation data
    with torch.no_grad():
        # for data in valid_loader:
        #     inputs = data[0].to(device)
        #     labels = data[1].to(device)

        #     outputs = model(inputs.view(inputs.shape[0], -1))
        #     loss = loss_function(outputs, labels)
        #     loss_total += loss.item()

        for i, (input, target, landmarks) in enumerate(valid_loader):
            # measure data loading time
            
            input = input.to(device)
            
            target = target.to(device)
            
            preds = model(input, landmarks)
            
            loss = criterion(preds, target) 
            loss_total += loss.item()
            # loss.update(loss1.item(), input.size(0))
            running_loss =+ loss.item() * input.size(0)

        validation_losses.append(running_loss / len(valid_loader))

    
    
    return loss_total


def test(val_loader,  model, criterion,  optimizer1, epoch,classes,last_epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()
    
    # top1 = 0
    # loss = 0

    mode =  'Testing'
    # switch to evaluate mode
    model.eval()
    loss_values = []
    end = time.time()

    corrects = [0 for _ in range(2 + 1)] #2 predictions due to ce + wing +1(majority)
   
    CM = torch.zeros(args.num_classes,args.num_classes)
    with torch.no_grad():         
        for i, (input, target, landmarks) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            preds = model(input, landmarks)
            prec = accuracy(preds, target, topk=(1,))
            # if epoch == last_epoch-1:
            #     for t, p in zip(target.view(-1), preds.view(-1)):
            #         confusion_matrix[t.long(), p.long()] += 1
            
                
            loss1 = criterion(preds, target) 
            loss.update(loss1.item(), input.size(0))
            #print(prec)
            top1.update(prec[0], input.size(0))
            # loss += loss1.item()
            # top1 += prec[0]

            # measure elapsed time

            batch_time.update(time.time() - end)
            end = time.time()
            
            # if epoch == last_epoch-1:
            topk=(1,)
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = preds.topk(maxk, 1, True, True)
            pred = pred.t()
            for t, p in zip(target.view(-1), pred.view(-1)):
                CM[t.long(), p.long()] += 1
            
            running_loss =+ loss1.item() * input.size(0)
        
            if i % args.print_freq == 0:
               print('Testing Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(val_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))
        print('Testing Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(val_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))
        test_losses.append(running_loss / len(val_loader))
        test_accu.append(prec[0].detach().cpu().clone().numpy())

   
    print(CM)
    print(CM.diag()/CM.sum(1))
    return top1.avg , CM



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch_num = state['epoch']
    full_filename = os.path.join(args.model_dir, str(epoch_num)+'_'+ filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
   
    return res
    
def adjust_learning_rate(optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        

if __name__ == '__main__':
    main()
