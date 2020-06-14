#coding=utf8

import os
import time
import torch
from torch.utils.data import DataLoader
from torchnet import meter
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable,**kwargs):
        return iterable
import logging
import datasets
import models
from utils.util import accuracy,get_lastest_model,get_args
from utils.config import opt
from utils.focalloss import FocalLoss

def train(args):
    # step0: parse config
    new_config ={"model":args.model,"num_workers":args.num_workers,
        "batch_size":args.batch_size,"load_model_path":args.load_model_path}
    opt.parse(new_config)
    # step1:model
    model = getattr(models,opt.model)()
    if opt.load_model_path is None:
        opt.load_model_path = get_lastest_model(prefix=opt.model)
    if opt.load_model_path:
        print("Resuming from "+opt.load_model_path)
        model.load_state_dict(torch.load(opt.load_model_path))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(opt.device)
    model.train()
    # step2: data
    dataset = getattr(datasets,opt.dataset)
    train_data = dataset(opt.train_data_root,train=True)
    val_data = dataset(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,pin_memory=True,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,pin_memory=True,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),opt.lr,weight_decay=opt.weight_decay)
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
        nIters = len(train_dataloader)
        pbar = tqdm(train_dataloader)
        start = time.time()
        for iter,(data,label) in enumerate(pbar):
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
            if sys.stderr.isatty():
                log_str = "{epoch}: Loss:{loss.val:.5f} Acc:{acc.val:.3f}".format(epoch=epoch,loss=loss_meter, acc=acc_meter)
                pbar.set_description(log_str)
            else:
                if iter%opt.print_freq == 0:
                    log_str = "{iter}/{len}: Loss:{loss.val:.5f} Acc:{acc.val:.3f}".format(iter=iter,len=nIters,loss=loss_meter, acc=acc_meter)
                    print(log_str)
        # validate and visualize
        end = time.time()
        if not sys.stderr.isatty():
            print(str(epoch)+": time "+str(end-start)+"s")
        val_cm,val_accuracy = val(model,val_dataloader)
        if val_accuracy > best_acc:
            best_acc =val_accuracy
            prefix = 'checkpoints/' + opt.model + '_'+"{acc:.2f}".format(acc=val_accuracy)
            name = time.strftime(prefix + '_%m%d_%H:%M:%S.pth')
            torch.save(model.state_dict(),name)
        print("Val {epoch}: Loss: {loss},Acc: {acc},lr: {lr}".format(epoch = epoch,acc=val_accuracy,loss = loss_meter.value()[0],lr=lr))
        #print("confusion_matrix:{val_cm}".format(val_cm = str(val_cm.value())))
        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            if lr > 1e-5:         
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

if __name__=='__main__':
    args = get_args()
    train(args)