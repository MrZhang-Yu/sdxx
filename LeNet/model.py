import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):#网络层
        super(LeNet, self).__init__()#解决多继承的问题
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)#最后的10代表有多少分类
    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(1, 28, 28) output(6, 24, 24)
        x = self.pool1(x)            # output(6, 12, 12)
        x = F.relu(self.conv2(x))    # output(16, 8, 8)
        x = self.pool2(x)            # output(16, 4, 4)
        x = x.view(-1, 16*4*4)       # output(16*4*4)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x