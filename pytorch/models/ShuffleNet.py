from torch import nn
import torchvision.models as models
import torch

class ShuffleNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ShuffleNet,self).__init__()
        self.model_name = "ShuffleNet"
        model = models.shufflenet_v2_x0_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features,num_classes)
        self.model = model        
    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    net = ShuffleNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    from torchsummary import summary
    print(summary(net,(3,224,224)))