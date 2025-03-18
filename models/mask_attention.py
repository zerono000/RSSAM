import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskAttention(nn.Module):
    """
    掩码引导的交叉注意力机制，限制注意力计算在前景区域
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 查询、键、值的线性投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 残差连接前的层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attention_mask=None):
        """
        参数:
            query: 形状为 [batch_size, num_queries, hidden_dim] 的查询张量
            key: 形状为 [batch_size, seq_len, hidden_dim] 的键张量
            value: 形状为 [batch_size, seq_len, hidden_dim] 的值张量
            attention_mask: 可选，形状为 [batch_size, num_queries, seq_len] 的二值注意力掩码
        
        返回:
            output: 形状为 [batch_size, num_queries, hidden_dim] 的输出张量
        """
        # 添加残差连接的预处理
        residual = query
        query = self.norm1(query)
        
        batch_size, num_queries, _ = query.shape
        seq_len = key.shape[1]
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头注意力格式 [batch_size, num_heads, seq_len, head_dim]
        q = q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_queries, seq_len]
        
        # 应用掩码（如果提供）
        if attention_mask is not None:
            # 确保掩码维度正确
            if attention_mask.dim() == 3:  # [B, num_queries, seq_len]
                # 扩展掩码到所有注意力头
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, num_queries, seq_len]
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)  # [B, num_heads, num_queries, seq_len]
                
                # 应用掩码
                attn_weights = attn_weights.masked_fill(~attention_mask, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        output = torch.matmul(attn_weights, v)  # [B, num_heads, num_queries, head_dim]
        
        # 重塑回原始格式
        output = output.permute(0, 2, 1, 3).reshape(batch_size, num_queries, self.hidden_dim)
        
        # 最终投影
        output = self.out_proj(output)
        
        # 应用残差连接
        output = output + residual
        
        return output