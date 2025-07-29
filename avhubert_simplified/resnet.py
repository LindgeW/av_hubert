"""
ResNet-based visual encoder for AVHuBERT.
Simplified version without fairseq dependencies.
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
    """Basic downsampling block"""
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
    """Alternative downsampling block with average pooling"""
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, relu_type: str = 'relu'):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """ResNet backbone for visual feature extraction"""
    
    def __init__(self, block: nn.Module, layers: list, num_classes: int = 1000,
                 relu_type: str = 'relu', gamma_zero: bool = False, 
                 avg_pool_downsample: bool = False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if relu_type == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
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

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


class ResEncoder(nn.Module):
    """ResNet encoder for visual features in AVHuBERT"""
    
    def __init__(self, relu_type: str = 'prelu', weights: Optional[str] = None):
        super(ResEncoder, self).__init__()
        
        # Create ResNet-18 backbone
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        
        # Remove the final classification layer
        self.backend_out = 512  # ResNet-18 feature dimension
        
        # Load pretrained weights if provided
        if weights is not None:
            self.load_pretrained_weights(weights)
    
    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out fc layer weights if they exist
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            self.backbone.load_state_dict(filtered_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through visual encoder
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, H, W)
            
        Returns:
            Visual features of shape (B, T, D) or (B, D)
        """
        original_shape = x.shape
        
        # Handle video input (B, T, C, H, W)
        if len(original_shape) == 5:
            B, T, C, H, W = original_shape
            x = x.view(B * T, C, H, W)
            
        # Extract features using ResNet backbone (without final FC layer)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)  # (B*T, 512) or (B, 512)
        
        # Reshape back to video format if needed
        if len(original_shape) == 5:
            x = x.view(B, T, -1)  # (B, T, 512)
            
        return x


def create_resnet_encoder(relu_type: str = 'prelu', weights: Optional[str] = None) -> ResEncoder:
    """Factory function to create ResNet encoder"""
    return ResEncoder(relu_type=relu_type, weights=weights)