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
    优化的匈牙利匹配器，提高匹配效率和成功率
    """
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        
        # 调试模式 - 设置为True输出更多日志
        self.debug = False
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 存储所有批次的匹配索引
        indices = []
        
        for b in range(bs):
            # 提取当前批次的预测
            out_prob = outputs["pred_logits"][b].softmax(-1)
            
            # 如果当前批次没有目标，添加空匹配
            if b >= len(targets) or "labels" not in targets[b] or len(targets[b]["labels"]) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=out_prob.device),
                    torch.tensor([], dtype=torch.int64, device=out_prob.device)
                ))
                continue
            
            # 提取目标类别
            tgt_ids = targets[b]["labels"]
            
            # 计算类别匹配成本 - 关键优化点，只考虑目标中存在的类别
            cost_class = -out_prob[:, tgt_ids]
            
            # 提取掩码预测和目标
            if outputs["pred_masks"].ndim == 5:
                pred_masks = outputs["pred_masks"][b, :, 0]  # [num_queries, H, W]
            else:
                pred_masks = outputs["pred_masks"][b]  # [num_queries, H, W]
            
            tgt_masks = targets[b]["masks"]
            
            # 确保掩码形状匹配
            if pred_masks.shape[-2:] != tgt_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks.unsqueeze(1).float(),
                    size=tgt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            
            # 展平掩码用于更高效的计算
            pred_masks_flat = pred_masks.flatten(1).float()
            tgt_masks_flat = tgt_masks.flatten(1).float()
            
            # 预先计算sigmoid，避免重复计算
            pred_masks_sigmoid = pred_masks_flat.sigmoid()
            
            # 优化：批量计算BCE成本
            # [num_queries, num_targets, H*W]
            bce_matrix = torch.zeros((pred_masks_flat.shape[0], tgt_masks_flat.shape[0]), 
                                     device=pred_masks_flat.device)
            
            # 处理尺寸问题：分批计算
            chunk_size = min(32, pred_masks_flat.shape[0])
            for i in range(0, pred_masks_flat.shape[0], chunk_size):
                end_i = min(i + chunk_size, pred_masks_flat.shape[0])
                mask_chunk = pred_masks_flat[i:end_i]
                
                # 分批计算BCE
                for j in range(0, tgt_masks_flat.shape[0], chunk_size):
                    end_j = min(j + chunk_size, tgt_masks_flat.shape[0])
                    target_chunk = tgt_masks_flat[j:end_j]
                    
                    # 对每对(查询,目标)计算BCE损失
                    for q_idx in range(end_i - i):
                        for t_idx in range(end_j - j):
                            bce = F.binary_cross_entropy_with_logits(
                                mask_chunk[q_idx:q_idx+1], 
                                target_chunk[t_idx:t_idx+1], 
                                reduction="none"
                            ).mean()
                            bce_matrix[i + q_idx, j + t_idx] = bce
            
            # 计算Dice成本
            dice_matrix = torch.zeros((pred_masks_flat.shape[0], tgt_masks_flat.shape[0]), 
                                      device=pred_masks_flat.device)
            
            for i in range(0, pred_masks_sigmoid.shape[0], chunk_size):
                end_i = min(i + chunk_size, pred_masks_sigmoid.shape[0])
                sigmoid_chunk = pred_masks_sigmoid[i:end_i]
                
                for j in range(0, tgt_masks_flat.shape[0], chunk_size):
                    end_j = min(j + chunk_size, tgt_masks_flat.shape[0])
                    target_chunk = tgt_masks_flat[j:end_j]
                    
                    for q_idx in range(end_i - i):
                        for t_idx in range(end_j - j):
                            numerator = 2 * (sigmoid_chunk[q_idx] * target_chunk[t_idx]).sum()
                            denominator = sigmoid_chunk[q_idx].sum() + target_chunk[t_idx].sum() + 1e-6
                            dice_loss = 1 - numerator / denominator
                            dice_matrix[i + q_idx, j + t_idx] = dice_loss
            
            # 计算总成本矩阵
            C = (
                self.cost_class * cost_class + 
                self.cost_mask * bce_matrix + 
                self.cost_dice * dice_matrix
            )
            
            # 使用匈牙利算法计算最优匹配
            C_np = C.cpu().numpy()
            indices_np = linear_sum_assignment(C_np)
            
            indices.append((
                torch.as_tensor(indices_np[0], dtype=torch.int64, device=out_prob.device),
                torch.as_tensor(indices_np[1], dtype=torch.int64, device=out_prob.device)
            ))
            
            # 调试时打印匹配情况
            if self.debug:
                print(f"批次 {b}: 找到 {len(indices_np[0])}/{len(tgt_ids)} 个匹配")
                
                # 打印匹配的类别信息
                if len(indices_np[0]) > 0:
                    matched_pred_classes = out_prob[indices_np[0]].argmax(-1)
                    matched_tgt_classes = tgt_ids[indices_np[1]]
                    
                    correct_class_matches = (matched_pred_classes == matched_tgt_classes).sum().item()
                    print(f"类别匹配正确率: {correct_class_matches}/{len(indices_np[0])} ({correct_class_matches/len(indices_np[0])*100:.1f}%)")
        
        return indices
    
class SetCriterion(nn.Module):
    """
    RSPrompter-query的损失计算器, 包括匹配后的监督损失计算
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
        """掩码损失: BCE + Dice"""
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
    
    def compute_semantic_losses(self, outputs, targets):
        """计算语义分割损失"""
        # 获取预测
        pred_masks = outputs['pred_masks']  # [B, N, 1, H, W]
        pred_logits = outputs['pred_logits']  # [B, N, C]
        
        batch_size = pred_masks.shape[0]
        losses = {}
        total_loss = 0
        
        for b in range(batch_size):
            target_mask = targets[b]['semantic_mask']  # [H, W]
            valid_pixels = targets[b]['valid_pixels']  # [H, W]
            
            # 创建语义预测图
            h, w = target_mask.shape
            semantic_pred = torch.zeros((h, w), dtype=torch.float, device=pred_masks.device)
            
            # 获取类别预测
            scores, labels = F.softmax(pred_logits[b], dim=-1).max(-1)
            
            # 为每个类别合并掩码
            for class_id in range(1, self.num_classes):  # 跳过背景类(0)
                class_indices = (labels == class_id - 1)  # 类别索引从0开始
                
                if class_indices.sum() > 0:
                    # 获取该类别的所有掩码
                    if pred_masks.dim() == 5:
                        class_masks = pred_masks[b, class_indices, 0]
                    else:
                        class_masks = pred_masks[b, class_indices]
                    
                    # 将掩码上采样到目标尺寸
                    if class_masks.shape[-2:] != (h, w):
                        class_masks = F.interpolate(
                            class_masks.unsqueeze(1),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    
                    # 合并掩码并应用类别
                    combined_mask = (class_masks.sigmoid() > 0.5).any(dim=0).float()
                    semantic_pred[combined_mask > 0] = class_id
            
            # 计算交叉熵损失
            semantic_pred = semantic_pred[valid_pixels].long()
            target_mask = target_mask[valid_pixels].long()
            
            if len(semantic_pred) > 0:
                loss = F.cross_entropy(
                    F.one_hot(semantic_pred, self.num_classes).float(), 
                    F.one_hot(target_mask, self.num_classes).float()
                )
                losses[f'loss_semantic_{b}'] = loss
                total_loss += loss
        
        # 返回平均损失
        losses['loss_semantic'] = total_loss / batch_size
        return losses

    def compute_instance_losses(self, outputs, targets):
        """计算实例分割总损失"""
        # 执行匹配
        indices = self.matcher(outputs, targets)
        
        # 打印调试信息
        matched_count = sum(len(src) for src, _ in indices)
        total_targets = sum(len(t.get("labels", [])) for t in targets)
        # print(f"匹配结果: 找到 {matched_count}/{total_targets} 个匹配对")
        
        # 如果没有匹配项，打印详细信息
        if matched_count == 0 and total_targets > 0:
            print("警告: 没有找到有效的匹配！")
            print(f"预测类别分布: {outputs['pred_logits'].argmax(-1).flatten().tolist()[:20]}...")
            print(f"真实类别分布: {[t['labels'].tolist() for t in targets if len(t['labels']) > 0]}")
        
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
                weighted_losses[k] = v
        
        return weighted_losses
    
    def forward(self, outputs, targets):
        """计算总损失"""
        # 检查分割类型
        segmentation_type = outputs.get('segmentation_type', 'instance')
        
        if segmentation_type == 'semantic':
            # 语义分割损失计算
            return self.compute_semantic_losses(outputs, targets)
        else:
            # 实例分割损失计算
            return self.compute_instance_losses(outputs, targets)