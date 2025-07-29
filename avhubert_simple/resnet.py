"""
ResNet implementation for visual feature extraction in AVHuBERT.
Simplified version without fairseq dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    """Downsample block for ResNet."""
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    """Downsample block v2 for ResNet."""
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='relu'):
        super(BasicBlock, self).__init__()
        
        assert relu_type in ['relu', 'prelu']
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu2(out)
        
        return out


class ResNet(nn.Module):
    """ResNet model."""
    
    def __init__(self, block, layers, num_classes=1000, relu_type='relu', 
                 gamma_zero=False, avg_pool_downsample=False):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(self.inplanes, planes * block.expansion, stride)
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.relu_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResEncoder(nn.Module):
    """ResNet encoder for visual features."""
    
    def __init__(self, relu_type, weights):
        super(ResEncoder, self).__init__()
        
        # Create ResNet-18
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        
        # Load pretrained weights if provided
        if weights:
            checkpoint = torch.load(weights, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove module prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # remove 'module.' prefix
                new_state_dict[k] = v
            
            self.resnet.load_state_dict(new_state_dict, strict=False)
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B*T, C, H, W]
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # Extract features
        features = self.resnet(x)  # [B*T, 512]
        
        # Reshape back to [B, T, 512]
        features = features.view(batch_size, time_steps, -1)
        
        return features

    def threeD_to_2D_tensor(self, x):
        """Convert 3D tensor to 2D for processing."""
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        return x