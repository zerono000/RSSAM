import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

# class GradientExtractor(nn.Module):
#     """
#     Extracts image gradients using Sobel filter
#     (使用Sobel滤波器提取图像梯度)
#     """
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, x):
#         # Convert to grayscale if RGB
#         if x.shape[1] > 1:
#             gray = x.mean(dim=1, keepdim=True)
#         else:
#             gray = x
        
#         # Calculate gradients using Sobel filters
#         grad_x = kf.sobel(gray, normalized=True, direction='x')
#         grad_y = kf.sobel(gray, normalized=True, direction='y')
        
#         # Compute gradient magnitude
#         grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
#         # Stack horizontal and vertical gradients
#         grad_features = torch.cat([grad_x, grad_y, grad_mag], dim=1)
        
#         return grad_features

class GradientExtractor(nn.Module):
    """
    Extracts image gradients using Sobel filter
    (使用Sobel滤波器提取图像梯度)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Convert to grayscale if RGB
        if x.shape[1] > 1:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x
        
        # 方法1：尝试使用kornia的sobel_x和sobel_y函数
        try:
            # 新版Kornia可能有专门的水平和垂直滤波器函数
            if hasattr(kf, 'sobel_x') and hasattr(kf, 'sobel_y'):
                grad_x = kf.sobel_x(gray, normalized=True)
                grad_y = kf.sobel_y(gray, normalized=True)
            # 方法2：如果sobel函数不支持direction参数，尝试使用PyTorch手动实现
            else:
                # 定义水平和垂直Sobel滤波器
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
                
                # 复制到与输入通道匹配的维度
                sobel_x = sobel_x.repeat(gray.shape[1], 1, 1, 1)
                sobel_y = sobel_y.repeat(gray.shape[1], 1, 1, 1)
                
                # 应用滤波器
                padding = nn.ReflectionPad2d(1)
                gray_padded = padding(gray)
                grad_x = F.conv2d(gray_padded, sobel_x, groups=gray.shape[1])
                grad_y = F.conv2d(gray_padded, sobel_y, groups=gray.shape[1])
                
                # 可选择归一化
                grad_x = grad_x / 8.0
                grad_y = grad_y / 8.0
        
        except Exception as e:
            print(f"Error in gradient extraction: {e}")
            print("Falling back to basic implementation")
            
            # 降级方案：使用简单差分
            padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
            grad_x = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
            grad_y = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
        
        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # Stack horizontal and vertical gradients
        grad_features = torch.cat([grad_x, grad_y, grad_mag], dim=1)
        
        return grad_features

class GradientEnhancer(nn.Module):
    """
    Enhances cross-attention with gradient information
    (使用梯度信息增强交叉注意力)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim  # 保存hidden_dim作为类属性
        
        # Project gradient features to hidden dimension
        self.gradient_proj = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=1)
        )
        
        # Attention fusion
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, decoder_features, gradient_features, memory=None):
        """
        Args:
            decoder_features: [B, N, C] from transformer decoder
            gradient_features: [B, 3, H, W] gradient features
            memory: Optional additional context
        Returns:
            Enhanced features with gradient information
        """
        # Project and reshape gradient features
        B = gradient_features.shape[0]
        proj_grad = self.gradient_proj(gradient_features)  # [B, C, H, W]
        h, w = proj_grad.shape[2:]
        proj_grad = proj_grad.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Query from decoder, Key/Value from gradients
        q = self.query_proj(decoder_features)  # [B, N, C]
        k = self.key_proj(proj_grad)  # [B, HW, C]
        v = self.value_proj(proj_grad)  # [B, HW, C]
        
        # 使用self.hidden_dim代替未定义的hidden_dim
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [B, N, HW]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)  # [B, N, C]
        enhanced = self.out_proj(context)
        
        # Add & norm (residual connection)
        enhanced = decoder_features + enhanced
        enhanced = self.norm(enhanced)
        
        return enhanced