import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    """
    Depthwise Separable Convolution module
    (深度可分离卷积模块)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 确保输入格式正确 [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != self.depthwise.in_channels:
            # 如果通道维度不对，尝试调整
            if x.shape[-1] == self.depthwise.in_channels:
                x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention module
    (通道注意力模块)
    """
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att

class FeatureAggregator(nn.Module):
    """
    Feature Aggregator module that processes selected layers from SAM encoder
    (特征聚合器模块, 处理从SAM编码器中选择的层)
    """
    def __init__(self, in_channels, out_channels, feature_layers):
        super().__init__()
        self.feature_layers = feature_layers
        
        # Shared projection module
        self.projection = DepthwiseConv(in_channels, out_channels)
        
        # Channel attention module
        self.attention = ChannelAttention(out_channels)
        
        # Final fusion
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def _reshape_vit_feature(self, x):
        """将ViT特征转换为卷积层所需的形状"""
        # 处理不同可能的输入形状
        if len(x.shape) == 3:  # [B, L, C]
            B, L, C = x.shape
            H = W = int(L**0.5)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        elif len(x.shape) == 4:
            B, H, W, C = x.shape
            if C == H or C == W:  # 判断是否通道已经在正确位置
                return x
            else:
                return x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x
        
    def forward(self, features):
        # 直接使用钩子收集的特征，不需要选择
        projected_features = [self.projection(feat) for feat in features]
        
        # 其余代码保持不变...
        # Skip connection from first feature
        skip_feature = projected_features[0]
        
        # Fuse features with element-wise addition
        fused_feature = projected_features[0]
        for feat in projected_features[1:]:
            # Resize feature to match the first one if needed
            if feat.shape[-2:] != fused_feature.shape[-2:]:
                feat = F.interpolate(
                    feat, size=fused_feature.shape[-2:], mode='bilinear', align_corners=False)
            fused_feature = fused_feature + feat
        
        # Apply channel attention
        attended_feature = self.attention(fused_feature)
        
        # Apply final fusion with skip connection
        agg_feature = self.fusion(attended_feature) + skip_feature
        
        return agg_feature

class MultiScaleGenerator(nn.Module):
    """
    Multi-scale feature generator using transpose conv and max pooling
    (使用转置卷积和最大池化的多尺度特征生成器)
    """
    def __init__(self, channels):
        super().__init__()
        
        # 转置卷积用于上采样x2
        self.transpose_conv = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1)
        
        # 用于精调每个尺度的特征
        self.conv_up2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_orig = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_down2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_down4 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_down8 = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        multi_scale_features = []
        
        # 上采样x2
        feat_up2 = self.transpose_conv(x)
        feat_up2 = self.conv_up2(feat_up2)
        multi_scale_features.append(feat_up2)
        
        # 原尺寸（恒等映射）
        feat_orig = self.conv_orig(x)
        multi_scale_features.append(feat_orig)
        
        # 下采样x2
        feat_down2 = F.max_pool2d(x, kernel_size=2, stride=2)
        feat_down2 = self.conv_down2(feat_down2)
        multi_scale_features.append(feat_down2)
        
        # 下采样x4
        feat_down4 = F.max_pool2d(x, kernel_size=4, stride=4)
        feat_down4 = self.conv_down4(feat_down4)
        multi_scale_features.append(feat_down4)
        
        # 下采样x8
        feat_down8 = F.max_pool2d(x, kernel_size=8, stride=8)
        feat_down8 = self.conv_down8(feat_down8)
        multi_scale_features.append(feat_down8)
        
        return multi_scale_features