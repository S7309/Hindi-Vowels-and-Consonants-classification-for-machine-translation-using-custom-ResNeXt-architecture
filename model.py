import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class BasicBlock(nn.Module):
    def __init__(self, channels = 256, stride = 1, padding = 1):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        
        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,
                                kernel_size = 3, stride = self.stride, padding = self.padding)
        self.bn_1 = nn.BatchNorm2d(self.channels)
        self.prelu_1 = nn.PReLU()
        
        self.conv_2 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,
                                kernel_size = 3, stride = self.stride, padding = self.padding)
        self.bn_2 = nn.BatchNorm2d(self.channels)
        self.prelu_2 = nn.PReLU()
        
        self.conv_3 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,
                                kernel_size = 5, stride = self.stride, padding = self.padding + 1)
        self.bn_3 = nn.BatchNorm2d(self.channels)
        
    def forward(self, x):
        identity = x
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x)) + self.bn_3(self.conv_3(identity))
        x = self.prelu_2(x)        
        return x

class ModInception(nn.Module):
    def __init__(self, channels = 256, stride = 1, padding = 1):
        super(ModInception, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        
        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = 60, kernel_size = 1,
                                stride = self.stride, padding = 0)
        self.conv_2 = nn.Sequential(
                      nn.Conv2d(in_channels = self.channels, out_channels = 70, kernel_size = 1, stride = self.stride,
                                padding = 0),
                      nn.BatchNorm2d(70),
                      nn.PReLU(),
            
                      nn.Conv2d(in_channels = 70, out_channels = 70, kernel_size = 3,stride = self.stride,
                                padding = 1)
                      )
        self.conv_3 = nn.Sequential(
                      nn.Conv2d(in_channels = self.channels, out_channels = 126, kernel_size = 1, stride = self.stride,
                                padding = 0),
                      nn.BatchNorm2d(126),
                      nn.PReLU(),
            
                      nn.Conv2d(in_channels = 126, out_channels = 126, kernel_size = 5, stride = self.stride,
                                padding = 2)
                      )
        self.bn = nn.BatchNorm2d(self.channels)
        self.prelu = nn.PReLU() 
        
    def forward(self, x):
        x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x)], dim=1)
        x = self.prelu(self.bn(x))
        return x

class ResNet(nn.Module):
    def __init__(self, block, incp_block):
        super(ResNet, self).__init__()
        self.block = block
        self.incp_block = incp_block
        self.input_conv = nn.Sequential(
                       nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 3, padding = 1),
                       nn.BatchNorm2d(128),
                       nn.PReLU(),

                       nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
                       nn.BatchNorm2d(256),
                       nn.PReLU(),
                       )
        
        self.layer_64x64 = self.make_layers(2)
        self.layer_32x32 = self.make_layers(2)
        self.layer_16x16 = self.make_layers(2)
        self.layer_8x8 = self.make_layers(2)
        self.layer_4x4 = self.make_layers(2)
        
        self.downsample_conv_1 = nn.Sequential(
                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),
                                 nn.BatchNorm2d(256),
                                 nn.PReLU()
                                 )
        
        self.downsample_conv_2 = nn.Sequential(
                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),
                                 nn.BatchNorm2d(256),
                                 nn.PReLU()
                                 )
        self.downsample_conv_3 = nn.Sequential(
                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),
                                 nn.BatchNorm2d(256),
                                 nn.PReLU()
                                 )
        self.downsample_conv_4 = nn.Sequential(
                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),
                                 nn.BatchNorm2d(256),
                                 nn.PReLU()
                                 )
        self.downsample_conv_5 = nn.Sequential(
                                 nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),
                                 nn.BatchNorm2d(128),
                                 nn.PReLU()
                                 )
            
        self.V_classifier = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.PReLU(),
            
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.PReLU(),
            
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.PReLU(),
                
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.PReLU(),
            
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.PReLU(),
            
                    nn.Linear(16, 10)
                    )
        
        self.C_classifier = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.PReLU(),
            
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.PReLU(),
            
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.PReLU(),
                
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.PReLU(),
            
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.PReLU(),
            
                    nn.Linear(16, 10)
                    )
    
    def make_layers(self, layers):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block())
            res_layers.append(self.incp_block())
        return nn.Sequential(*res_layers)
    
    def forward(self, x):
        
        x = self.input_conv(x)
        x = self.layer_64x64(x)
        x = self.downsample_conv_1(x)
        x = self.layer_32x32(x)
        x = self.downsample_conv_2(x)
        x = self.layer_16x16(x)
        x = self.downsample_conv_3(x)
        x = self.layer_8x8(x)
        x = self.downsample_conv_4(x)
        x = self.layer_4x4(x)
        x = self.downsample_conv_5(x)
        x = x.view(x.shape[0], -1)
        out_1 = self.V_classifier(x)
        out_2 = self.C_classifier(x)
        return out_1, out_2

