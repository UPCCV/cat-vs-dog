import os
import argparse
import torch
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import models
from utils.util import get_args,get_lastest_model

CLASSES=['cat','dog']

transformsImage = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def test_image(model, device, imgpath):
    img = cv2.imread(imgpath)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = transformsImage(img).unsqueeze(0)
    outputs = model(image.to(device))
    _, index = torch.max(outputs, 1)
    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    print(imgpath,CLASSES[index[0]], percentage[index[0]].item())

def test_dir(model, device, dir):
    files = os.listdir(dir)
    for file in tqdm(files):
        imgpath = dir+'/'+file
        test_image(model,device, imgpath)
def export_model(model):
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones(1, 3, 224, 224))
    traced_script_module.save("./mnasnet_dogcat.pt")

@torch.no_grad()
def demo():
    args = get_args()
    model = getattr(models,args.model)()
    if not args.load_model_path:
        args.load_model_path = get_lastest_model()
    if not args.load_model_path:
        print("No pretrained model found")
        return
    model.load_state_dict(torch.load(args.load_model_path,map_location=torch.device('cpu')))
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
    model.to(device)
    model.eval()
    #export_model(model)
    test_image(model, device, args.image_path)
    #test_dir(model, device, args.image_dir)

if __name__=="__main__":
    demo()