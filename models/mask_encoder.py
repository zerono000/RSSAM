import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseMaskEncoder(nn.Module):
    """
    Encodes coarse masks into dense prompt embeddings for SAM
    (将粗掩码编码为SAM的密集提示嵌入)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 修改最后输出通道为SAM期望的hidden_dim
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=1)  # 确保输出通道是hidden_dim
        )
    
    def forward(self, coarse_masks):
        """处理每张图像中所有实例掩码"""
        B, N, H, W = coarse_masks.shape
        
        # 合并实例为单一掩码
        combined_masks = torch.zeros((B, 1, H, W), device=coarse_masks.device)
        for b in range(B):
            combined_masks[b] = (coarse_masks[b].sum(dim=0, keepdim=True) > 0).float()
        
        # 编码掩码
        encoded = self.mask_encoder(combined_masks)
        
        return encoded  # [B, 1, H, W]