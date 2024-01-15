import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.sh = []
    self.bn = nn.BatchNorm2d(0)
    self.act = nn.LeakyReLU(0.2)
    # CNN layer
    self.bn1 = nn.BatchNorm2d(3)
    self.conv1 = nn.Conv2d(3, 3, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool1 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 2)
    
    self.bn2 = nn.BatchNorm2d(3)
    self.conv2 = nn.Conv2d(3, 1, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool2 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 1)

    self.bn3 = nn.BatchNorm2d(1)
    self.conv3 = nn.Conv2d(1, 1, kernel_size=(5,9), padding = (2,4), stride = 1,  bias=False)
    self.pool3 = nn.MaxPool2d(kernel_size=(5,9), padding = (2,4), stride = 2)

    # LSTM layer
    self.lstm = nn.LSTM(input_size= 32, hidden_size= 100, num_layers=3, batch_first=True)
    self.fc5 = nn.Linear(100,2)
    self.sm = nn.Softmax(dim = -1)
  def forward(self, input):
    x = self.act(self.conv1(input))
    x = self.pool1(x)
#     print(x.shape)

    x = self.act(self.conv2(x))
    x = self.pool2(x)
#     print(x.shape)

    x = self.act(self.conv3(x))
    x = self.pool3(x)
#     print(x.shape)

    x = x.transpose(2,3)
    x = torch.flatten(x, start_dim=1, end_dim=2)
#     print(x.shape)
    self.sh = x
  
    out,(h,c) = self.lstm(x)
    
    x = out[:,-1]
#     print(x.shape)
    
    x = self.sm(self.fc5(x))
#     print(x.shape)

    return x