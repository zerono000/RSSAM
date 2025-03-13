import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment

def dice_loss(inputs, targets, num_boxes):
    """
    计算Dice损失
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    # 确保输入是sigmoid结果
    inputs = inputs.sigmoid()
    
    # 避免空批次
    if inputs.shape[0] == 0:
        return inputs.sum() * 0.0
    
    # 计算Dice系数
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1) + 1e-6
    
    # 返回Loss = 1 - Dice
    loss = 1 - (numerator / denominator)
    
    # 返回所有实例的平均损失
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """
    Focal loss for dense prediction
    FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)
    """
    # 确保输入不为空
    if inputs.shape[0] == 0:
        return inputs.sum() * 0.0
    
    # 使用PyTorch实现
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # 返回所有实例的平均损失
    return loss.mean(1).sum() / num_boxes

class HungarianMatcher(nn.Module):
    """
    基于查询的RSPrompter匹配器，匹配预测与目标的对应关系
    """
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """执行匈牙利匹配"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 存储所有批次的匹配索引
        indices = []
        
        # 逐批次进行匹配
        for b in range(bs):
            # 提取当前批次的预测
            # 类别预测 [num_queries, num_classes]
            out_prob = outputs["pred_logits"][b].softmax(-1)
            
            # 如果当前批次没有目标，添加空匹配
            if b >= len(targets) or "labels" not in targets[b] or len(targets[b]["labels"]) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=out_prob.device),
                    torch.tensor([], dtype=torch.int64, device=out_prob.device)
                ))
                continue
            
            # 提取目标类别 [num_targets]
            tgt_ids = targets[b]["labels"]
            
            # 计算类别匹配成本 [num_queries, num_targets]
            cost_class = -out_prob[:, tgt_ids]
            
            # 提取掩码预测和目标
            # 掩码预测 [num_queries, 1, H, W] 或 [num_queries, H, W]
            if outputs["pred_masks"].ndim == 5:
                pred_masks = outputs["pred_masks"][b, :, 0]  # [num_queries, H, W]
            else:
                pred_masks = outputs["pred_masks"][b]  # [num_queries, H, W]
            
            # 目标掩码 [num_targets, H, W]
            tgt_masks = targets[b]["masks"]
            
            # 确保掩码形状匹配
            if pred_masks.shape[-2:] != tgt_masks.shape[-2:]:
                # 调整预测掩码大小以匹配目标
                pred_masks = F.interpolate(
                    pred_masks.unsqueeze(1).float(),  # 确保浮点类型
                    size=tgt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            
            # 展平掩码用于BCE计算 [num_queries, H*W], [num_targets, H*W]
            pred_masks_flat = pred_masks.flatten(1).float()  # 确保浮点类型
            tgt_masks_flat = tgt_masks.flatten(1).float()    # 确保浮点类型
            
            # 计算BCE成本
            cost_mask = F.binary_cross_entropy_with_logits(
                pred_masks_flat.unsqueeze(1).repeat(1, tgt_masks_flat.shape[0], 1),
                tgt_masks_flat.unsqueeze(0).repeat(pred_masks_flat.shape[0], 1, 1),
                reduction="none"
            ).mean(dim=2)  # [num_queries, num_targets]
            
            # 计算Dice成本
            pred_masks_sigmoid = pred_masks_flat.sigmoid()
            cost_dice = []
            
            for i in range(pred_masks_flat.shape[0]):
                dice_cost_i = []
                for j in range(tgt_masks_flat.shape[0]):
                    numerator = 2 * (pred_masks_sigmoid[i] * tgt_masks_flat[j]).sum()
                    denominator = pred_masks_sigmoid[i].sum() + tgt_masks_flat[j].sum() + 1e-6
                    dice_cost = 1 - numerator / denominator
                    dice_cost_i.append(dice_cost)
                cost_dice.append(torch.stack(dice_cost_i))
            
            cost_dice = torch.stack(cost_dice)  # [num_queries, num_targets]
            
            # 计算总成本矩阵
            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            
            # 使用匈牙利算法计算最优匹配
            C_np = C.cpu().numpy()
            indices_np = linear_sum_assignment(C_np)
            indices.append((
                torch.as_tensor(indices_np[0], dtype=torch.int64, device=out_prob.device),
                torch.as_tensor(indices_np[1], dtype=torch.int64, device=out_prob.device)
            ))
        
        return indices

class SetCriterion(nn.Module):
    """
    RSPrompter-query的损失计算器，包括匹配后的监督损失计算
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, losses=('labels', 'masks', 'prompt')):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 类别权重设置，背景类权重较低
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        """类别分类损失"""
        # 获取类别预测
        src_logits = outputs['pred_logits']  # [B, N, C+1]
        
        # 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        
        # 获取匹配的目标类别
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J) > 0:  # 确保有匹配的目标
                target_classes_o.append(t["labels"][J])
        
        # 如果没有匹配的目标，返回连接到计算图的零损失
        if len(target_classes_o) == 0:
            return {'loss_ce': src_logits.sum() * 0.0}
        
        target_classes_o = torch.cat(target_classes_o)
        
        # 初始化所有目标类别为背景类(num_classes)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, 
            dtype=torch.int64, device=src_logits.device
        )
        
        # 将匹配的位置设置为目标类别
        target_classes[idx] = target_classes_o
        
        # 计算交叉熵损失
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, 
            self.empty_weight.to(src_logits.device)
        )
        
        return {'loss_ce': loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """掩码损失：BCE + Dice"""
        # 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        
        # 如果没有匹配，返回零损失
        if len(idx[0]) == 0:
            return {
                'loss_mask': outputs['pred_masks'].sum() * 0.0, 
                'loss_dice': outputs['pred_masks'].sum() * 0.0
            }
        
        # 提取匹配的掩码预测
        if outputs['pred_masks'].ndim == 5:
            # [B, N, 1, H, W] -> [N_matched, H, W]
            src_masks = outputs['pred_masks'][idx[0], idx[1], 0]
        else:
            # [B, N, H, W] -> [N_matched, H, W]
            src_masks = outputs['pred_masks'][idx[0], idx[1]]
        
        # 提取匹配的目标掩码
        target_masks = []
        for b, (t, (_, J)) in enumerate(zip(targets, indices)):
            if len(J) > 0:
                # 直接使用掩码，稍后转换类型
                target_masks.append(t['masks'][J])
        
        # 如果没有有效目标掩码，返回零损失
        if not target_masks:
            return {
                'loss_mask': outputs['pred_masks'].sum() * 0.0, 
                'loss_dice': outputs['pred_masks'].sum() * 0.0
            }
        
        # 堆叠目标掩码
        target_masks = torch.cat(target_masks)
        
        # 确保掩码形状匹配
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            # 调整预测掩码大小以匹配目标
            src_masks = F.interpolate(
                src_masks.unsqueeze(1).float(), 
                size=target_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # 展平掩码用于损失计算 - 明确转换为浮点类型
        src_masks = src_masks.flatten(1).float()
        target_masks = target_masks.flatten(1).float()
        
        # 计算BCE损失
        loss_mask = F.binary_cross_entropy_with_logits(
            src_masks, target_masks, reduction='none'
        ).mean(1).sum() / num_masks
        
        # 计算Dice损失
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        
        # 处理粗掩码损失
        losses = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice
        }
        
        if 'coarse_masks' in outputs and outputs['coarse_masks'] is not None:
            if outputs['coarse_masks'].shape[0] > 0:  # 确保有粗掩码
                try:
                    # 提取匹配的粗掩码
                    coarse_masks = outputs['coarse_masks'][idx[0], idx[1]]
                    coarse_masks = coarse_masks.flatten(1).float()
                    
                    # 确保形状匹配
                    if coarse_masks.shape[-1] != target_masks.shape[-1]:
                        h_src = w_src = int(math.sqrt(coarse_masks.shape[-1]))
                        h_tgt = w_tgt = int(math.sqrt(target_masks.shape[-1]))
                        coarse_masks = F.interpolate(
                            coarse_masks.reshape(-1, 1, h_src, h_src),
                            size=(h_tgt, w_tgt),
                            mode='bilinear',
                            align_corners=False
                        ).reshape(-1, h_tgt*w_tgt)
                    
                    # 计算粗掩码损失
                    loss_coarse_mask = F.binary_cross_entropy_with_logits(
                        coarse_masks, target_masks, reduction='none'
                    ).mean(1).sum() / num_masks
                    
                    loss_coarse_dice = dice_loss(coarse_masks, target_masks, num_masks)
                    
                    losses.update({
                        'loss_coarse_mask': loss_coarse_mask,
                        'loss_coarse_dice': loss_coarse_dice
                    })
                except Exception as e:
                    print(f"粗掩码损失计算失败: {e}")
                    losses.update({
                        'loss_coarse_mask': outputs['pred_masks'].sum() * 0.0,
                        'loss_coarse_dice': outputs['pred_masks'].sum() * 0.0
                    })
        
        return losses
    
    def loss_prompt(self, outputs, targets, indices, num_masks):
        """提示嵌入正则化损失"""
        if 'prompt_embeddings' in outputs and outputs['prompt_embeddings'] is not None:
            prompt_embeddings = outputs['prompt_embeddings']
            
            # 如果提示嵌入为空，返回零损失
            if prompt_embeddings.numel() == 0:
                return {'loss_prompt': outputs['pred_logits'].sum() * 0.0}
            
            # 计算L2正则化损失
            loss_prompt = torch.norm(prompt_embeddings, p=2, dim=-1).mean()
            return {'loss_prompt': loss_prompt}
        
        return {'loss_prompt': outputs['pred_logits'].sum() * 0.0}
    
    def _get_src_permutation_idx(self, indices):
        """获取源(预测)索引的批次和查询位置"""
        batch_idx = []
        src_idx = []
        for i, (src, _) in enumerate(indices):
            if len(src) > 0:
                batch_idx.append(torch.full_like(src, i))
                src_idx.append(src)
        
        if not batch_idx:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        
        batch_idx = torch.cat(batch_idx)
        src_idx = torch.cat(src_idx)
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        """获取目标索引的批次和位置"""
        batch_idx = []
        tgt_idx = []
        for i, (_, tgt) in enumerate(indices):
            if len(tgt) > 0:
                batch_idx.append(torch.full_like(tgt, i))
                tgt_idx.append(tgt)
        
        if not batch_idx:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        
        batch_idx = torch.cat(batch_idx)
        tgt_idx = torch.cat(tgt_idx)
        return batch_idx, tgt_idx
    
    def forward(self, outputs, targets):
        """计算总损失"""
        # 执行匹配
        indices = self.matcher(outputs, targets)
        
        # 计算目标掩码数量用于损失标准化
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, 
            device=next(iter(outputs.values())).device
        ).clamp(min=1)  # 确保不为零
        
        # 计算各种损失
        losses = {}
        for loss in self.losses:
            loss_func = getattr(self, f"loss_{loss}")
            losses.update(loss_func(outputs, targets, indices, num_masks))
        
        # 应用权重
        weighted_losses = {}
        for k, v in losses.items():
            if k in self.weight_dict:
                weighted_losses[k] = v * self.weight_dict[k]
            else:
                # 保留未加权的损失供调试使用
                weighted_losses[k] = v
        
        return weighted_losses