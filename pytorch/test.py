#coding=utf8

import os
import cv2
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from config import opt
import datasets
import models
from utils.util import accuracy,get_lastest_model,get_args

CLASSES=['cat','dog']

@torch.no_grad()
def test(args):
    new_config ={"model":args.model,"num_workers":args.num_workers,
    "batch_size":args.batch_size,"load_model_path":args.load_model_path}
    opt.parse(new_config)
    if not args.load_model_path:
        args.load_model_path = get_lastest_model()
    if not args.load_model_path:
        print("No pretrained model found")
        return
    # step1: configure model
    model = getattr(models, opt.model)()
    model.load_state_dict(torch.load(args.load_model_path))
    model.to(opt.device)
    dataset = getattr(datasets,opt.dataset)
    val_data = dataset(opt.train_data_root,train=False)
    val_dataloader = DataLoader(val_data,opt.batch_size,
            shuffle=False,num_workers=opt.num_workers)
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)

    for _, (val_input, label) in enumerate(tqdm(val_dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print("Acc:{acc}".format(acc=accuracy))
    print("confusion matrix:{val_cm}".format(val_cm = str(cm_value)))

def test_loader():
    dataset = getattr(datasets,opt.dataset)
    train_data = dataset(opt.train_data_root,train=False)
    dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=0)
    for _, (val_input, label) in enumerate(dataloader):
        for i in range(len(val_input.numpy())):
            img = val_input.numpy()[i].transpose(1,2,0).copy()
            cv2.putText(img,CLASSES[label[i].item()],(0,40),3,1,(0,0,255))
            cv2.imshow("img",img)
            cv2.waitKey()

if __name__=='__main__':
    args = get_args()
    test(args)
    #test_loader()