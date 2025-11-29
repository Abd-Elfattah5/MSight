import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import torch.nn.init as init

import torchio as tio
import random
import scipy.ndimage


# Unet with attention mechanism
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.last_attention_map = None  

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        self.last_attention_map = psi  
        return x * psi

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        diffZ = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffZ // 2, diffZ - diffZ // 2,
                      diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.down1 = DownBlock(in_channels, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)

        self.up1 = UpBlock(128, 64, skip_channels=64)
        self.up2 = UpBlock(64, 32, skip_channels=32)
        self.up3 = UpBlock(32, 16, skip_channels=16)

        self.out = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.out(x)
        return x