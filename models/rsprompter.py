import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .aggregator import FeatureAggregator, MultiScaleGenerator
from .prompter import QueryPrompter
from .gradient_module import GradientExtractor, GradientEnhancer
from .mask_encoder import CoarseMaskEncoder

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
            feature_dim=config.model.out_channels,  # 传递实际特征通道数
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

    def _hook_fn(self, module, input, output):
        """钩子函数，用于捕获中间层特征"""
        self.intermediate_features.append(output)
    
    def forward(self, images):
        # 保存原始图像尺寸用于后处理
        original_size = (images.shape[-2], images.shape[-1])
        input_size = original_size  # 假设不裁剪
        # 清空存储的特征
        self.intermediate_features = []
        
        # 注册钩子
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
        
        # 处理粗掩码 - 修改这部分逻辑
        dense_embeddings = None
        if self.use_coarse_mask and prompter_outputs['coarse_masks'] is not None:
            coarse_masks = prompter_outputs['coarse_masks']
            # print(coarse_masks.shape)
            dense_embeddings = self.mask_encoder(coarse_masks)
            
            # 打印形状以进行调试
            print(f"Dense embeddings shape: {dense_embeddings.shape}")
        
        # 获取提示嵌入
        prompt_embeddings = prompter_outputs['prompt_embeddings']
        
        # 打印关键形状信息
        print(f"Prompt embeddings shape: {prompt_embeddings.shape}")
        
        # 重塑提示嵌入以用于SAM
        b, n, points, c = prompt_embeddings.shape
        print(f"Batch size: {b}, Queries: {n}, Points per query: {points}")
        sparse_embeddings = prompt_embeddings.reshape(b, n * points, c)
        
        # ----- 关键修复部分开始 -----
        # 从SAM获取位置编码
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        
        # 准备空的密集嵌入
        if self.use_coarse_mask and 'coarse_masks' in prompter_outputs:
            try:
                dense_embeddings = self.mask_encoder(prompter_outputs['coarse_masks'])
            except Exception as e:
                print(f"警告: 粗掩码编码失败 {e}, 使用零张量替代")
                h, w = image_embeddings.shape[-2:]
                dense_embeddings = torch.zeros((b, 1, h, w), device=image_embeddings.device)
        else:
            # 创建空的密集嵌入
            h, w = image_embeddings.shape[-2:]
            dense_embeddings = torch.zeros((b, 1, h, w), device=image_embeddings.device)
        
        # 打印掩码解码器输入的形状
        print(f"Image embeddings: {image_embeddings.shape}")
        print(f"Sparse embeddings: {sparse_embeddings.shape}")
        print(f"Dense embeddings: {dense_embeddings.shape}")
        
        # 运行SAM掩码解码器 - 使用稀疏嵌入的前n_per_image个
        # 这里我们选择每张图像只使用前100个查询点
        n_per_image = min(100, n * points)  # 限制每张图像的提示数量
        
        all_mask_predictions = []
        all_class_logits = []
        
        for i in range(b):
            # 为每张图像处理最多n_per_image个提示
            current_sparse = sparse_embeddings[i, :n_per_image].unsqueeze(0)  # [1, n_per_image, c]
            current_image = image_embeddings[i:i+1]  # [1, C, H, W]
            current_dense = dense_embeddings[i:i+1]  # [1, 1, H, W]
            
            # 对当前图像运行掩码解码器
            mask_pred, _ = self.sam.mask_decoder(
                image_embeddings=current_image,
                image_pe=image_pe,
                sparse_prompt_embeddings=current_sparse,
                dense_prompt_embeddings=current_dense,
                multimask_output=False,
            )

            # 使用SAM的后处理方法将掩码上采样到原始尺寸
            upsampled_masks = self.sam.postprocess_masks(
                mask_pred,
                input_size=input_size,
                original_size=original_size
            )
            
            # 保存当前图像的预测和类别
            all_mask_predictions.append(upsampled_masks)  # [1, n_per_image, 1, H, W]
            
            # 保存类别逻辑（如果需要）
            current_class = prompter_outputs['class_logits'][i, :n_per_image//points].unsqueeze(0)
            all_class_logits.append(current_class)
        
        # 合并所有预测
        if all_mask_predictions:
            mask_predictions = torch.cat(all_mask_predictions, dim=0)  # [B, n_per_image, 1, H, W]
            class_logits = torch.cat(all_class_logits, dim=0)  # [B, n_per_image//points, num_classes]
            
            # 打印实际的掩码形状
            print(f"Raw mask_predictions shape: {mask_predictions.shape}")
            
            # 重新组织掩码以匹配外部预期格式 - 不要过度重塑
            # 只保留我们能处理的实例数量
            instances_to_keep = n_per_image // points
            mask_predictions = mask_predictions[:, :instances_to_keep]  # [B, instances_to_keep, 1, H, W]
        else:
            # 创建空的掩码预测
            mask_predictions = torch.zeros((b, 1, 1, image_embeddings.shape[-2], image_embeddings.shape[-1]), 
                                        device=image_embeddings.device)
            class_logits = prompter_outputs['class_logits']

        # 在返回结果前调整pred_masks的形状以匹配pred_logits的批次结构
        class_logits = prompter_outputs['class_logits']  # [B, N, C]
        pred_masks = mask_predictions
        
        # 调整pred_masks以匹配class_logits的查询数量
        B, N, C = class_logits.shape
        if pred_masks.shape[1] != N:
            # 如果pred_masks每个样本只有1个掩码，但class_logits有N个查询
            # 复制掩码N次以匹配
            pred_masks = pred_masks.repeat(1, N, 1, 1, 1)
    

        # 打印最终掩码尺寸
        print(f"Final mask_predictions shape: {mask_predictions.shape}")
        
        return {
            'pred_logits': class_logits,  # 使用与掩码匹配的类别
            'pred_masks': mask_predictions,
            'coarse_masks': prompter_outputs['coarse_masks'] if self.use_coarse_mask else None
        }
