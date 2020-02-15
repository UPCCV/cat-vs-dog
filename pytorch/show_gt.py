import torch
from torch.utils.data import DataLoader
from data.dataset import DogCat
from config import opt
import cv2

def main():
    data = DogCat(opt.train_data_root,train=False)
    dataloader = DataLoader(data,4,
                        shuffle=False,num_workers=0)
    for _, (val_input, label) in enumerate(dataloader):
        for i in range(len(val_input.numpy())):
            img = val_input.numpy()[i].transpose(1,2,0)
            cv2.imshow("img",img)
            cv2.waitKey()

if __name__=="__main__":
    main()