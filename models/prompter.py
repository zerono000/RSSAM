import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class PositionLevelEncoding(nn.Module):
    """
    Adds position and level encoding to multi-scale features
    (为多尺度特征添加位置和层级编码)
    """
    def __init__(self, feature_dim, num_scales):  # 修改参数名为feature_dim
        super().__init__()
        
        # Position encodings for each scale - 使用feature_dim
        self.position_encodings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim, 1, 1))  # 使用feature_dim替代固定值
            for _ in range(num_scales)
        ])
        
        # Level encodings for each scale - 使用feature_dim
        self.level_encodings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim, 1, 1))  # 使用feature_dim替代固定值
            for _ in range(num_scales)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        encoded_features = []
        
        for i, feature in enumerate(features):
            # 确保索引在范围内
            pos_idx = min(i, len(self.position_encodings) - 1)
            lvl_idx = min(i, len(self.level_encodings) - 1)
            
            # 获取编码
            pos_encoding = self.position_encodings[pos_idx]
            lvl_encoding = self.level_encodings[lvl_idx]
            
            # 打印形状以进行调试
            # print(f"Feature shape: {feature.shape}, Pos encoding shape: {pos_encoding.shape}")
            
            # 确保维度匹配
            if pos_encoding.shape[1] != feature.shape[1]:
                raise ValueError(f"Encoding channels ({pos_encoding.shape[1]}) don't match "
                               f"feature channels ({feature.shape[1]}) at index {i}")
            
            # Add position encoding (broadcast to feature map size)
            pos_encoding = pos_encoding.expand(-1, -1, feature.shape[2], feature.shape[3])
            
            # Add level encoding
            lvl_encoding = lvl_encoding.expand(-1, -1, feature.shape[2], feature.shape[3])
            
            # Combine encodings with feature
            encoded = feature + pos_encoding + lvl_encoding
            encoded_features.append(encoded)
            
        return encoded_features


class QueryPrompter(nn.Module):
    """
    Query-based prompter that generates SAM-compatible prompts
    (基于查询的提示器, 生成SAM兼容的提示)
    """
    def __init__(self, hidden_dim, num_queries, num_classes,
                 feature_dim=32, num_points_per_query=5, # 添加新参数，默认值为32
                 num_encoder_layers=3, num_decoder_layers=6, dropout=0.1,
                 use_gradient=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim  # 保存feature_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.use_gradient = use_gradient
        self.num_points_per_query = num_points_per_query
        
        # Position and level encoding - 使用feature_dim
        self.encoding = PositionLevelEncoding(feature_dim, num_scales=5)
        
        # Projection to align feature dimensions
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_head = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_head = nn.Linear(hidden_dim, hidden_dim * self.num_points_per_query)  # 5 points per instance
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights
        (初始化权重)
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, multi_scale_features, gradient_module=None, original_image=None):
        # Add position and level encoding
        encoded_features = self.encoding(multi_scale_features)
        
        # Project and flatten features for transformer
        projected_features = []
        largest_feature = None
        
        for i, feature in enumerate(encoded_features):
            proj = self.input_proj(feature)
            if i == 0:  # 保存最大尺寸特征用于掩码生成
                largest_feature = proj
            
            # Flatten spatial dimensions
            b, c, h, w = proj.shape
            flattened = proj.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            projected_features.append(flattened)
        
        # Concatenate all flattened features
        concat_features = torch.cat(projected_features, dim=1)  # [B, sum(HW), C]
        
        # Apply transformer encoder
        memory = self.transformer_encoder(concat_features)
        
        # Prepare query embeddings
        batch_size = memory.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, C]
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(query_embed, memory)
        
        # Apply gradient enhancement if enabled
        if self.use_gradient and gradient_module is not None and original_image is not None:
            gradient_features = gradient_module[0](original_image)
            decoder_output = gradient_module[1](decoder_output, gradient_features)
        
        # Apply output heads
        class_logits = self.class_head(decoder_output)
        mask_tokens = self.mask_head(decoder_output)
        prompt_embeddings = self.prompt_head(decoder_output)
        
        # Reshape prompt embeddings to get 5 points per instance
        b, n, _ = prompt_embeddings.shape
        prompt_embeddings = prompt_embeddings.reshape(b, n, self.num_points_per_query, self.hidden_dim)
        
        # Generate coarse masks using mask tokens and largest feature
        coarse_masks = self._generate_coarse_masks(mask_tokens, largest_feature)
        
        return {
            'class_logits': class_logits,  # [B, N, num_classes+1]
            'prompt_embeddings': prompt_embeddings,  # [B, N, 5, hidden_dim]
            'coarse_masks': coarse_masks,  # [B, N, H, W]
            'decoder_output': decoder_output  # 用于需要的额外处理
        }
    
    def _generate_coarse_masks(self, mask_tokens, feature_map):
        """
        Generate coarse masks by applying mask tokens to feature map
        (通过将掩码令牌应用于特征图来生成粗掩码)
        """
        b, c, h, w = feature_map.shape
        flattened_features = feature_map.flatten(2)  # [B, C, HW]
        
        # 矩阵乘法和归一化
        masks = torch.matmul(mask_tokens, flattened_features)  # [B, N, HW]
        masks = masks.reshape(b, -1, h, w)  # [B, N, H, W]
        
        # Sigmoid activation
        masks = masks.sigmoid()
        
        return masks