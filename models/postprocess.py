import torch
import torch.nn.functional as F

class SegmentationPostProcessor:
    """分割后处理基类"""
    def __init__(self, config):
        self.config = config
    
    def process(self, outputs, image_size):
        raise NotImplementedError("子类必须实现此方法")

class InstanceSegmentationPostProcessor(SegmentationPostProcessor):
    """实例分割后处理"""
    def __init__(self, config):
        super().__init__(config)
        self.score_threshold = getattr(config.postprocessing, 'score_threshold', 0.5)
        self.mask_threshold = getattr(config.postprocessing, 'mask_threshold', 0.5)
    
    def process(self, outputs, image_size):
        # 获取预测
        pred_logits = outputs["pred_logits"]  # [B, N, C]
        pred_masks = outputs["pred_masks"]    # [B, N, 1, H, W]
        
        batch_size = pred_logits.shape[0]
        processed_results = []
        
        for b in range(batch_size):
            # 获取类别预测和置信度
            scores, labels = F.softmax(pred_logits[b], dim=-1).max(-1)
            
            # 过滤低置信度预测和背景类
            keep = (scores > self.score_threshold) & (labels < pred_logits.shape[-1] - 1)
            scores_per_image = scores[keep]
            labels_per_image = labels[keep]
            
            # 如果没有有效预测，返回空结果
            if keep.sum() == 0:
                processed_results.append({
                    "masks": torch.zeros((0, image_size[0], image_size[1]), device=pred_masks.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=pred_logits.device),
                    "scores": torch.zeros(0, device=pred_logits.device)
                })
                continue
            
            # 提取有效掩码
            if pred_masks.dim() == 5:
                masks_per_image = pred_masks[b, keep, 0]  # [K, H, W]
            else:
                masks_per_image = pred_masks[b, keep]  # [K, H, W]
            
            # 将掩码上采样到原始图像尺寸（如有必要）
            if masks_per_image.shape[-2:] != image_size:
                masks_per_image = F.interpolate(
                    masks_per_image.unsqueeze(1),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            
            # 二值化掩码
            masks_per_image = masks_per_image.sigmoid() > self.mask_threshold
            
            processed_results.append({
                "masks": masks_per_image,
                "labels": labels_per_image,
                "scores": scores_per_image
            })
        
        return processed_results

class SemanticSegmentationPostProcessor(SegmentationPostProcessor):
    """语义分割后处理"""
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.dataset.num_classes
        self.mask_threshold = getattr(config.postprocessing, 'mask_threshold', 0.5)
    
    def process(self, outputs, image_size):
        # 获取预测
        pred_logits = outputs["pred_logits"]  # [B, N, C]
        pred_masks = outputs["pred_masks"]    # [B, N, 1, H, W]
        
        batch_size = pred_logits.shape[0]
        processed_results = []
        
        for b in range(batch_size):
            # 获取类别预测
            scores, labels = F.softmax(pred_logits[b], dim=-1).max(-1)
            
            # 过滤背景类
            valid_indices = labels < (self.num_classes)
            
            # 准备语义分割结果
            semantic_map = torch.zeros(image_size, device=pred_masks.device, dtype=torch.long)
            
            # 提取掩码
            if pred_masks.dim() == 5:
                all_masks = pred_masks[b, :, 0]  # [N, H, W]
            else:
                all_masks = pred_masks[b]  # [N, H, W]
            
            # 将掩码上采样到原始图像尺寸（如有必要）
            if all_masks.shape[-2:] != image_size:
                all_masks = F.interpolate(
                    all_masks.unsqueeze(1),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            
            # 将掩码转换为概率
            all_masks = all_masks.sigmoid()
            
            # 为每个类别合并掩码
            for class_id in range(self.num_classes):
                # 找出属于当前类别的所有掩码
                class_indices = (labels == class_id) & valid_indices
                
                if class_indices.sum() > 0:
                    # 获取当前类别的所有掩码
                    class_masks = all_masks[class_indices]  # [K, H, W]
                    class_scores = scores[class_indices]  # [K]
                    
                    # 将所有掩码合并为一个掩码，使用最大概率
                    if class_masks.shape[0] > 1:
                        # 按照置信度排序掩码
                        sorted_indices = torch.argsort(class_scores, descending=True)
                        class_masks = class_masks[sorted_indices]
                        
                        # 创建合并掩码
                        combined_mask = torch.zeros_like(class_masks[0])
                        
                        # 依次合并掩码，优先级按置信度
                        for mask in class_masks:
                            # 将当前掩码添加到未分配区域
                            unassigned = (combined_mask <= self.mask_threshold)
                            combined_mask[unassigned] = mask[unassigned]
                    else:
                        combined_mask = class_masks[0]
                    
                    # 将类别添加到语义图
                    class_pixels = combined_mask > self.mask_threshold
                    semantic_map[class_pixels] = class_id + 1  # +1因为背景是0
            
            processed_results.append({
                "semantic_map": semantic_map
            })
        
        return processed_results

def build_postprocessor(config):
    """根据配置构建后处理器"""
    segmentation_type = getattr(config.model, 'segmentation_type', 'instance')
    
    if segmentation_type == 'instance':
        return InstanceSegmentationPostProcessor(config)
    elif segmentation_type == 'semantic':
        return SemanticSegmentationPostProcessor(config)
    else:
        raise ValueError(f"不支持的分割类型: {segmentation_type}")