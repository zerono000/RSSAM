import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .prompter import QueryPrompter
from .mask_encoder import CoarseMaskEncoder
from .aggregator import FeatureAggregator, MultiScaleGenerator
from .gradient_module import GradientExtractor, GradientEnhancer

class RSPrompter(nn.Module):
    """
    Full RSPrompter model with improved components
    (具有改进组件的完整RSPrompter模型)
    """
    def __init__(self, config):
        super().__init__()
        
        # Load SAM model
        self.sam = sam_model_registry[config.model.sam_type](checkpoint=config.model.checkpoint)
        
        # Freeze image encoder if specified
        self.sam.image_encoder.eval()
        if config.model.freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # Get encoder dimensions - fixed for SAM model variants
        if config.model.sam_type == "vit_h":
            in_channels = 1280  # ViT-H hidden dimension
        elif config.model.sam_type == "vit_l":
            in_channels = 1024  # ViT-L hidden dimension
        else:  # vit_b
            in_channels = 768   # ViT-B hidden dimension
            
        hidden_dim = config.model.hidden_dim
        
        # Feature aggregator
        self.aggregator = FeatureAggregator(
            in_channels=in_channels,
            out_channels=config.model.out_channels,
            feature_layers=config.model.feature_layers
        )
        
        # Multi-scale feature generator
        self.multi_scale_generator = MultiScaleGenerator(
            channels=config.model.out_channels
        )
        
        # Gradient modules if enabled
        self.use_gradient = config.model.use_gradient
        if self.use_gradient:
            self.gradient_extractor = GradientExtractor()
            self.gradient_enhancer = GradientEnhancer(hidden_dim=hidden_dim)
        
        # Query-based prompter
        self.prompter = QueryPrompter(
            hidden_dim=hidden_dim,
            feature_dim=config.model.out_channels,
            num_queries=config.model.num_queries,
            num_classes=config.dataset.num_classes,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dropout=config.model.dropout,
            use_gradient=config.model.use_gradient
        )
        
        # Coarse mask encoder if enabled
        self.use_coarse_mask = config.model.use_coarse_mask
        if self.use_coarse_mask:
            self.mask_encoder = CoarseMaskEncoder(hidden_dim=hidden_dim)

        # 用于存储特征的列表
        self.intermediate_features = []
        
        # 特征层索引
        self.feature_layers = config.model.feature_layers
        
        # 配置参数
        # self.per_query_mask = getattr(config.model, 'per_query_mask', True)
        # self.max_queries_per_batch = getattr(config.model, 'max_queries_per_batch', 10)
        # 设置最大前景查询数量，避免内存不足
        self.max_foreground_queries = getattr(config.model, 'max_foreground_queries', 10)
        self.memory_efficient = getattr(config.model, 'memory_efficient', True)

        # 如果配置了同步BN并处于分布式训练模式
        if hasattr(config, 'distributed') and hasattr(config.distributed, 'sync_bn') and config.distributed.sync_bn:
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        
    def _hook_fn(self, module, input, output):
        """钩子函数，用于捕获中间层特征"""
        self.intermediate_features.append(output)
    
    def forward(self, images):
        """
        RSPrompter的端到端前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]
            
        Returns:
            包含预测结果的字典：类别预测、掩码预测、提示嵌入和可选的粗略掩码
        """
        # 保存原始图像尺寸用于后处理
        original_size = (images.shape[-2], images.shape[-1])
        input_size = original_size  # 假设不裁剪
        
        # 清空存储的特征
        self.intermediate_features = []
        
        # 注册钩子收集特征
        hooks = []
        for i, block in enumerate(self.sam.image_encoder.blocks):
            if i in self.feature_layers:
                hooks.append(block.register_forward_hook(self._hook_fn))
        
        # 处理图像
        with torch.no_grad() if not self.sam.image_encoder.training else torch.enable_grad():
            image_embeddings = self.sam.image_encoder(images)
        
        # 移除钩子
        for h in hooks:
            h.remove()
        
        # 特征聚合
        aggregated_feature = self.aggregator(self.intermediate_features)
        
        # 生成多尺度特征
        multi_scale_features = self.multi_scale_generator(aggregated_feature)
        
        # 准备梯度增强模块
        gradient_module = None
        if self.use_gradient:
            gradient_module = (self.gradient_extractor, self.gradient_enhancer)
        
        # 通过查询提示器生成提示
        prompter_outputs = self.prompter(
            multi_scale_features, 
            gradient_module, 
            images
        )
        
        # 获取类别预测和提示嵌入
        class_logits = prompter_outputs['class_logits']  # [B, N, C]
        prompt_embeddings = prompter_outputs['prompt_embeddings']  # [B, N, 5, 256]
        coarse_masks = prompter_outputs['coarse_masks'] if self.use_coarse_mask else None  # [B, N, H, W]
        
        # 从SAM获取位置编码
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        
        # 批次大小和查询数量
        batch_size, num_queries = class_logits.shape[:2]
        points_per_query = prompt_embeddings.shape[2]
        hidden_dim = prompt_embeddings.shape[3]
        
        # 准备密集嵌入
        if self.use_coarse_mask and coarse_masks is not None:
            try:
                dense_embeddings = self.mask_encoder(coarse_masks)  # [B, 256, H, W]
            except Exception as e:
                print(f"警告: 粗掩码编码失败 {e}, 使用零张量替代")
                h, w = image_embeddings.shape[-2:]
                dense_embeddings = torch.zeros((batch_size, hidden_dim, h, w), device=image_embeddings.device)
        else:
            # 创建空的密集嵌入
            h, w = image_embeddings.shape[-2:]
            dense_embeddings = torch.zeros((batch_size, hidden_dim, h, w), device=image_embeddings.device)
        
        # 为每个查询生成单独的掩码 - 内存优化版本
        all_masks = []
        
        for b in range(batch_size):
            # 预测类别，找出前景查询
            if hasattr(self, 'memory_efficient') and self.memory_efficient:
                # 只处理预测为前景的查询
                foreground_queries = (class_logits[b].argmax(-1) == 0).nonzero().squeeze(-1)
                
                # 如果是0维张量，转为1维
                if foreground_queries.ndim == 0 and foreground_queries.numel() == 1:
                    foreground_queries = foreground_queries.unsqueeze(0)
                
                # 如果没有前景查询，使用空查询集
                if foreground_queries.numel() == 0:
                    query_indices = torch.tensor([], dtype=torch.long, device=class_logits.device)
                else:
                    # 限制处理的前景查询数量
                    max_queries = min(foreground_queries.shape[0], self.max_foreground_queries)
                    query_indices = foreground_queries[:max_queries]
            else:
                # 处理所有查询
                query_indices = torch.arange(num_queries, device=class_logits.device)
            
            # 为该批次创建掩码存储
            batch_masks = torch.zeros((1, num_queries, 1, original_size[0], original_size[1]), 
                                    device=images.device)
            
            # 如果有查询要处理
            if len(query_indices) > 0:
                current_image_embedding = image_embeddings[b:b+1]  # [1, 256, H, W]
                current_dense_embedding = dense_embeddings[b:b+1]  # [1, 256, H, W]
                
                # 批量处理查询以节省内存 (每次处理4个查询)
                batch_size_q = min(4, len(query_indices))
                for i in range(0, len(query_indices), batch_size_q):
                    end_idx = min(i + batch_size_q, len(query_indices))
                    batch_indices = query_indices[i:end_idx]
                    
                    # 为批量查询生成掩码
                    for q_idx in batch_indices:
                        # 获取当前查询的提示嵌入
                        current_prompt = prompt_embeddings[b, q_idx].unsqueeze(0)  # [1, 5, 256]
                        
                        # 运行SAM掩码解码器
                        mask_pred, _ = self.sam.mask_decoder(
                            image_embeddings=current_image_embedding,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=current_prompt,
                            dense_prompt_embeddings=current_dense_embedding,
                            multimask_output=False,
                        )  # [1, 1, H, W]
                        
                        # 将掩码上采样到原始尺寸
                        upsampled_mask = self.sam.postprocess_masks(
                            mask_pred,
                            input_size=input_size,
                            original_size=original_size
                        )  # [1, 1, H, W]
                        
                        # 存储掩码
                        batch_masks[0, q_idx] = upsampled_mask
                        
                        # 立即清理不需要的变量
                        del mask_pred
                    
                    # 显式清理缓存
                    if i + batch_size_q < len(query_indices):
                        torch.cuda.empty_cache()
            
            all_masks.append(batch_masks)
        
        # 合并所有批次的掩码
        pred_masks = torch.cat(all_masks, dim=0)  # [B, N, 1, H, W]
        
        return {
            'pred_logits': class_logits,  # [B, N, C]
            'pred_masks': pred_masks,  # [B, N, 1, H, W]
            'prompt_embeddings': prompt_embeddings,  # [B, N, 5, 256] 用于计算提示损失
            'coarse_masks': coarse_masks if self.use_coarse_mask else None  # [B, N, H, W] 可选
        }