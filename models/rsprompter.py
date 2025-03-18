import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .prompter import QueryPrompter
from .mask_encoder import CoarseMaskEncoder
from .aggregator import FeatureAggregator, MultiScaleGenerator
from .gradient import GradientExtractor, GradientEnhancer
from .postprocess import build_postprocessor

class RSPrompter(nn.Module):
    """
    Full RSPrompter model with improved components
    (具有改进组件的完整RSPrompter模型)
    """
    def __init__(self, config):
        super().__init__()

        # 添加分割类型参数
        self.segmentation_type = getattr(config.model, 'segmentation_type', 'instance')
        
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

        # 确保掩码注意力设置正确传递
        self.use_mask_attention = getattr(config.model, 'use_mask_attention', True)
        
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
            use_gradient=config.model.use_gradient,
            use_mask_attention=self.use_mask_attention
        )

        # 创建后处理器
        self.postprocessor = build_postprocessor(config)
        
        # Coarse mask encoder if enabled
        self.use_coarse_mask = config.model.use_coarse_mask
        if self.use_coarse_mask:
            self.mask_encoder = CoarseMaskEncoder(hidden_dim=hidden_dim)

        # 用于存储特征的列表
        self.intermediate_features = []
        
        # 特征层索引
        self.feature_layers = config.model.feature_layers
        
        # 设置最大前景查询数量，避免内存不足
        self.max_foreground_queries = getattr(config.model, 'max_foreground_queries', 10)
        self.memory_efficient = getattr(config.model, 'memory_efficient', True)

        # 如果配置了同步BN并处于分布式训练模式
        if hasattr(config, 'distributed') and hasattr(config.distributed, 'sync_bn') and config.distributed.sync_bn:
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        
    def _hook_fn(self, module, input, output):
        """钩子函数，用于捕获中间层特征"""
        self.intermediate_features.append(output)

    def get_trainable_and_encoder_state_dict(self):
        """
        将模型状态字典分离为可训练部分和编码器部分
        
        Returns:
            tuple: (可训练参数的状态字典, 编码器参数的状态字典)
        """
        full_state_dict = self.state_dict()
        encoder_prefix = 'sam.image_encoder.'
        
        encoder_state_dict = {}
        trainable_state_dict = {}
        
        for key, value in full_state_dict.items():
            if key.startswith(encoder_prefix):
                # 提取编码器相关参数（移除前缀）
                encoder_key = key[len(encoder_prefix):]
                encoder_state_dict[encoder_key] = value
            else:
                # 提取所有非编码器参数
                trainable_state_dict[key] = value
        
        return trainable_state_dict, encoder_state_dict

    def load_weights(self, trainable_weights_path, encoder_weights_path=None):
        """
        加载分离的权重文件
        
        Args:
            trainable_weights_path: 可训练参数权重文件路径
            encoder_weights_path: 编码器参数权重文件路径（可选）
            
        Returns:
            self: 加载权重后的模型实例
        """
        # 加载可训练部分权重
        trainable_checkpoint = torch.load(trainable_weights_path, map_location='cpu')
        
        # 检查是否存在model键
        if 'model' in trainable_checkpoint:
            trainable_state_dict = trainable_checkpoint['model']
        else:
            # 兼容直接保存state_dict的情况
            trainable_state_dict = trainable_checkpoint
        
        # 获取当前模型状态字典
        current_state_dict = self.state_dict()
        
        # 更新可训练参数
        for key, value in trainable_state_dict.items():
            if key in current_state_dict:
                current_state_dict[key] = value
        
        # 如果提供了编码器权重，加载编码器部分
        if encoder_weights_path is not None and os.path.exists(encoder_weights_path):
            encoder_checkpoint = torch.load(encoder_weights_path, map_location='cpu')
            
            # 检查编码器权重格式
            if 'model_encoder' in encoder_checkpoint:
                encoder_state_dict = encoder_checkpoint['model_encoder']
            else:
                encoder_state_dict = encoder_checkpoint
            
            # 将编码器参数添加到状态字典，添加前缀
            encoder_prefix = 'sam.image_encoder.'
            for key, value in encoder_state_dict.items():
                full_key = encoder_prefix + key
                if full_key in current_state_dict:
                    current_state_dict[full_key] = value
        
        # 加载组合后的状态字典
        missing_keys, unexpected_keys = self.load_state_dict(current_state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"警告: 缺少的键: {missing_keys[:5]}...")
        if len(unexpected_keys) > 0:
            print(f"警告: 意外的键: {unexpected_keys[:5]}...")
        
        return self
    
    def forward(self, images, segmentation_type=None):
        """
        RSPrompter的端到端前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]
            segmentation_type: 可选, 覆盖当前分割类型设置
            
        Returns:
            包含预测结果的字典
        """
        # 确定使用的分割类型
        seg_type = segmentation_type if segmentation_type is not None else self.segmentation_type
        
        # 保存原始图像尺寸用于后处理
        original_size = (images.shape[-2], images.shape[-1])
        input_size = original_size
        
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
        
        # 特征聚合和生成多尺度特征
        aggregated_feature = self.aggregator(self.intermediate_features)
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
        coarse_masks = prompter_outputs.get('coarse_masks', None)  # [B, N, H, W]
        
        # 从SAM获取位置编码
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        
        # 批次大小和查询数量
        batch_size, num_queries = class_logits.shape[:2]
        
        # 准备密集嵌入
        if self.use_coarse_mask and coarse_masks is not None:
            try:
                dense_embeddings = self.mask_encoder(coarse_masks)  # [B, 256, H, W]
            except Exception as e:
                print(f"警告: 粗掩码编码失败 {e}, 使用零张量替代")
                h, w = image_embeddings.shape[-2:]
                dense_embeddings = torch.zeros((batch_size, self.hidden_dim, h, w), device=image_embeddings.device)
        else:
            # 创建空的密集嵌入
            h, w = image_embeddings.shape[-2:]
            dense_embeddings = torch.zeros((batch_size, self.hidden_dim, h, w), device=image_embeddings.device)
        
        # 根据分割类型选择不同的处理逻辑
        if seg_type == 'semantic' and hasattr(self, 'memory_efficient') and self.memory_efficient:
            # 语义分割优化: 按类别处理查询
            all_masks = []
            
            for b in range(batch_size):
                # 创建存储掩码的张量
                batch_masks = torch.zeros((1, num_queries, 1, original_size[0], original_size[1]), 
                                        device=images.device)
                
                # 获取所有唯一类别
                pred_classes = class_logits[b].argmax(-1)
                unique_classes = pred_classes.unique()
                
                # 按类别批量处理
                for cls_id in unique_classes:
                    if cls_id == class_logits.shape[-1] - 1:  # 跳过背景类
                        continue
                        
                    # 找出当前类别的所有查询
                    cls_indices = (pred_classes == cls_id).nonzero().squeeze(-1)
                    
                    if cls_indices.numel() == 0:
                        continue
                        
                    # 分批处理以节省内存
                    batch_size_q = min(4, len(cls_indices))
                    for i in range(0, len(cls_indices), batch_size_q):
                        end_idx = min(i + batch_size_q, len(cls_indices))
                        batch_indices = cls_indices[i:end_idx]
                        
                        for q_idx in batch_indices:
                            # 获取当前查询的提示嵌入
                            current_prompt = prompt_embeddings[b, q_idx].unsqueeze(0)  # [1, 5, 256]
                            
                            # 运行SAM掩码解码器
                            mask_pred, _ = self.sam.mask_decoder(
                                image_embeddings=image_embeddings[b:b+1],
                                image_pe=image_pe,
                                sparse_prompt_embeddings=current_prompt,
                                dense_prompt_embeddings=dense_embeddings[b:b+1],
                                multimask_output=False,
                            )
                            
                            # 将掩码上采样到原始尺寸
                            upsampled_mask = self.sam.postprocess_masks(
                                mask_pred,
                                input_size=input_size,
                                original_size=original_size
                            )
                            
                            # 存储掩码
                            batch_masks[0, q_idx] = upsampled_mask
                            
                            # 清理变量
                            del mask_pred
                        
                        # 清理缓存
                        if i + batch_size_q < len(cls_indices):
                            torch.cuda.empty_cache()
                
                all_masks.append(batch_masks)
            
            # 合并所有批次的掩码
            pred_masks = torch.cat(all_masks, dim=0)  # [B, N, 1, H, W]
        else:
            # 实例分割处理（保持原有逻辑）
            all_masks = []
            
            for b in range(batch_size):
                # 预测类别，找出前景查询
                if seg_type == 'instance' and hasattr(self, 'memory_efficient') and self.memory_efficient:
                    # 只处理前景类别的查询
                    foreground_queries = (class_logits[b].argmax(-1) < class_logits.shape[-1] - 1).nonzero().squeeze(-1)
                    
                    # 确保维度正确
                    if foreground_queries.ndim == 0 and foreground_queries.numel() == 1:
                        foreground_queries = foreground_queries.unsqueeze(0)
                    
                    # 如果没有前景查询，使用空集
                    if foreground_queries.numel() == 0:
                        query_indices = torch.tensor([], dtype=torch.long, device=class_logits.device)
                    else:
                        # 限制处理的前景查询数量
                        max_queries = min(foreground_queries.shape[0], self.max_foreground_queries)
                        query_indices = foreground_queries[:max_queries]
                else:
                    # 处理所有查询
                    query_indices = torch.arange(num_queries, device=class_logits.device)
                
                # 创建掩码存储
                batch_masks = torch.zeros((1, num_queries, 1, original_size[0], original_size[1]), 
                                        device=images.device)
                
                # 处理有效查询
                if len(query_indices) > 0:
                    current_image_embedding = image_embeddings[b:b+1]
                    current_dense_embedding = dense_embeddings[b:b+1]
                    
                    # 分批处理查询
                    batch_size_q = min(4, len(query_indices))
                    for i in range(0, len(query_indices), batch_size_q):
                        end_idx = min(i + batch_size_q, len(query_indices))
                        batch_indices = query_indices[i:end_idx]
                        
                        for q_idx in batch_indices:
                            # 获取当前查询的提示嵌入
                            current_prompt = prompt_embeddings[b, q_idx].unsqueeze(0)
                            
                            # 运行SAM掩码解码器
                            mask_pred, _ = self.sam.mask_decoder(
                                image_embeddings=current_image_embedding,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=current_prompt,
                                dense_prompt_embeddings=current_dense_embedding,
                                multimask_output=False,
                            )
                            
                            # 将掩码上采样到原始尺寸
                            upsampled_mask = self.sam.postprocess_masks(
                                mask_pred,
                                input_size=input_size,
                                original_size=original_size
                            )
                            
                            # 存储掩码
                            batch_masks[0, q_idx] = upsampled_mask
                            
                            # 清理变量
                            del mask_pred
                        
                        # 清理缓存
                        if i + batch_size_q < len(query_indices):
                            torch.cuda.empty_cache()
                
                all_masks.append(batch_masks)
            
            # 合并所有批次的掩码
            pred_masks = torch.cat(all_masks, dim=0)  # [B, N, 1, H, W]
        
        # 使用后处理器进行处理（如果需要）
        if hasattr(self, 'postprocessor') and self.training is False:
            processed_results = self.postprocessor.process(
                {"pred_logits": class_logits, "pred_masks": pred_masks},
                original_size
            )
            
            # 在输出中添加处理后的结果
            return {
                'pred_logits': class_logits,  # [B, N, C]
                'pred_masks': pred_masks,  # [B, N, 1, H, W]
                'prompt_embeddings': prompt_embeddings,  # [B, N, 5, 256]
                'coarse_masks': coarse_masks,  # [B, N, H, W] 可选
                'segmentation_type': seg_type,  # 分割类型
                'processed_results': processed_results  # 后处理结果
            }
        else:
            # 返回原始预测结果
            return {
                'pred_logits': class_logits,  # [B, N, C]
                'pred_masks': pred_masks,  # [B, N, 1, H, W]
                'prompt_embeddings': prompt_embeddings,  # [B, N, 5, 256]
                'coarse_masks': coarse_masks,  # [B, N, H, W] 可选
                'segmentation_type': seg_type  # 分割类型
            }