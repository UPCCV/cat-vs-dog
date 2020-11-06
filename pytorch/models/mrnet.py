from torch import nn
import torch
import math
import torch.nn.functional as F
class MRNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(MRNet,self).__init__()
        self.model_name = "MRNet"
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(self.conv1.out_channels,16,3)
        self.conv3 = nn.Conv2d(self.conv2.out_channels,24,3)
        self.conv4 = nn.Conv2d(self.conv3.out_channels,32,3)
        self.conv5 = nn.Conv2d(self.conv4.out_channels,64,3)
        self.conv6 = nn.Conv2d(self.conv5.out_channels,128,3)
        self.fc = nn.Linear(self.conv6.out_channels,num_classes)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = F.max_pool2d(F.relu(self.conv4(x)),2)
        x = F.max_pool2d(F.relu(self.conv5(x)),2)
        x = F.max_pool2d(F.relu(self.conv6(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

if __name__=="__main__":
    net = MRNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    from torchsummary import summary
    print(summary(net,(3,224,224)))
<<<<<<< HEAD
    dummy_input = torch.rand(1,3,224,224).to(device)
=======
    dummy_input = torch.rand(1,3,224,224).to(device)
>>>>>>> df55394a74005d3d1b17654c1b4d31c41d4e2bb6
