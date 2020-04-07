from torch import nn
import torchvision.models as models
import torch
from torchsummary import summary

class SqueezeNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SqueezeNet,self).__init__()
        self.model_name = "SqueezeNet"
        model = models.squeezenet1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048,2)
        self.model = model        
    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    mrnet = SqueezeNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mrnet.to(device)
    print(summary(mrnet,(3,224,224)))