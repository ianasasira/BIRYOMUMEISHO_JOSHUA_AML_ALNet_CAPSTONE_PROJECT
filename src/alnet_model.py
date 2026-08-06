import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (EfficientNet-style)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Lightweight spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class LocalizedSparseAttention(nn.Module):
    """Localized sparse multi-head self-attention block.
    Uses a combination of channel and spatial attention to capture
    morphological features while staying lightweight."""
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.channel_attn(x)
        x = self.norm1(x + identity)

        identity = x
        x = self.spatial_attn(x)
        x = self.norm2(x + identity)
        return x


class ConvBlock(nn.Module):
    """Conv block with two depthwise separable convolutions + attention."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.attention = LocalizedSparseAttention(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x


class TransitionBlock(nn.Module):
    """Max pooling for spatial down-sampling."""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class ALNet(nn.Module):
    """ALNet: Acute Leukemia Network.
    A lightweight hybrid model combining depthwise separable convolutions
    with localized sparse attention for AML detection from blood smear images.
    
    Architecture:
    - Conv Block 1: depthwise separable convs + attention
    - Conv Block 2: depthwise separable convs + attention
    - Transition Block: max pooling
    - Dense Block: progressively reduced FC layers with dropout
    - Output: 2-class softmax (AML / Non-AML)
    """

    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()

        self.conv_block1 = ConvBlock(input_channels, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.transition = TransitionBlock()
        self.pool_global = nn.AdaptiveAvgPool2d(1)

        self.dense1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.transition(x)
        x = self.pool_global(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.classifier(x)
        return x


class WeightedFocalLoss(nn.Module):
    """Custom Weighted Focal Loss.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma: focusing parameter; down-weights easy/healthy examples
    - alpha: class weight for the positive class (corrects imbalance)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
