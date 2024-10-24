import torch.nn as nn


class CNN_model_1(nn.Module):
    def __init__(self):
        super(CNN_model_1,self).__init__()
        self.cnn_model_1 = nn.Sequential(
                
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), #96
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), #48
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(), 
        nn.AdaptiveMaxPool2d(1),
        
        nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1, padding=0), #1 
        nn.Flatten())
        
    def forward(self,x):
        return self.cnn_model_1(x)


class CNN_model_2(nn.Module):
    def __init__(self):
        super(CNN_model_2,self).__init__()
        self.cnn_model_2 = nn.Sequential(
                
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), #96
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), #48
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(), 
        nn.AdaptiveMaxPool2d(1),
        
        nn.Conv2d(in_channels=64, out_channels=5, kernel_size=1, stride=1, padding=0), #1 
        nn.Flatten(),
        nn.Sigmoid())
        
    def forward(self,x):
        return self.cnn_model_2(x)
    

class LNN_model_1(nn.Module):
    def __init__(self):
        super(LNN_model_1,self).__init__()
        self.lnn_model_1 = nn.Sequential(
                
        nn.Flatten(),
        nn.Linear(96*96*3, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 6))
        
    def forward(self,x):
        return self.lnn_model_1(x)


class LNN_model_2(nn.Module):
    def __init__(self):
        super(LNN_model_2,self).__init__()
        self.lnn_model_1 = nn.Sequential(
                
        nn.Flatten(),
        nn.Linear(96*96*3, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.Sigmoid())
        
    def forward(self,x):
        return self.lnn_model_1(x)