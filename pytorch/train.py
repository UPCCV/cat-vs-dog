#coding=utf8

import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import torchvision.models as models
#import models
from data.dataset import DogCat
from config import opt

def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_lastest_model(modeldir="checkpoints"):
    files = os.listdir(modeldir)
    if len(files)==0:
        return None
    files.sort(key=lambda fn:os.path.getmtime(modeldir + "/" + fn))
    lastest = os.path.join(modeldir,files[-1])
    return lastest

def train(args):
    opt.parse(args)
    # step1: setup model
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 2)
    # try to resume from lastest checkpoint
    if opt.load_model_path is None:
        opt.load_model_path = get_lastest_model()
    if opt.load_model_path:
        print("Resuming from "+opt.load_model_path)
        model.load_state_dict(torch.load(opt.load_model_path))
    model.to(opt.device)
    # step2: data
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    # optimizer = model.get_optimizer(lr, opt.weight_decay)
    optimizer = torch.optim.Adam(model.fc.parameters(), opt.lr, weight_decay=opt.weight_decay)    
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    acc_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    best_acc = 0

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        acc_meter.reset()
        confusion_matrix.reset()
        pbar = tqdm(train_dataloader)
        for _,(data,label) in enumerate(pbar):
            # train model 
            input = data.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            y_pred = model(input)
            loss = criterion(y_pred,target)
            prec1 = accuracy(y_pred.data, target)
            loss.backward()
            optimizer.step()            
            # meters update
            loss_meter.add(loss.item())
            acc_meter.add(prec1[0].item())
            confusion_matrix.add(y_pred.detach(), target.detach()) 
            pbar.set_description("{epoch}: Loss:{loss.val:.5f} Acc:{acc.val:.3f}".format(epoch=epoch,loss=loss_meter, acc=acc_meter))
        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)
        if val_accuracy > best_acc:
            best_acc =val_accuracy
            prefix = 'checkpoints/' + args.model + '_'+"{acc:.2f}".format(acc=val_accuracy)
            name = time.strftime(prefix + '_%m%d_%H:%M:%S.pth')
            torch.save(model.state_dict(),name)
        print("{epoch}: Acc:{acc},loss:{loss},lr:{lr}".format(epoch = epoch,acc=val_accuracy,loss = loss_meter.value()[0],lr=lr))
        print("confusion_matrix:")
        print("{val_cm}".format(val_cm = str(val_cm.value())))
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

@torch.no_grad()
def val(model,dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for _, (val_input, label) in enumerate(tqdm(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use_gpu",type=bool,default=True,help='an integer for the accumulator')
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--model",type=str,default="ResNet50")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--load_model_path",type=str,default=None)
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    train(args)