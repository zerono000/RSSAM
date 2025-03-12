import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss
    (计算DICE损失)
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    # 安全处理
    if inputs.shape[0] == 0:
        return torch.tensor(0.0, device=inputs.device)
    
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """
    Focal loss for dense prediction
    (用于密集预测的Focal损失)
    """
    # 安全处理
    if inputs.shape[0] == 0:
        return torch.tensor(0.0, device=inputs.device)
    
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class HungarianMatcher(nn.Module):
    """
    This class computes assignment between targets and predictions
    (计算目标和预测之间的匹配)
    """
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Compute the assignment between targets and predictions
        (计算目标和预测之间的匹配)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 获取掩码的批次信息
        mask_bs, mask_nq = outputs["pred_masks"].shape[:2]
        
        # 打印形状信息以便调试
        print(f"Debug - pred_logits: {outputs['pred_logits'].shape}")
        print(f"Debug - pred_masks: {outputs['pred_masks'].shape}")
        
        # 计算类别成本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*num_queries, num_classes]
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [total_targets]
        cost_class = -out_prob[:, tgt_ids]  # [bs*num_queries, total_targets]
        
        # 计算掩码成本
        out_mask = outputs["pred_masks"].flatten(0, 1).flatten(1).float()  # [mask_bs*mask_nq, H*W]
        tgt_mask = torch.cat([v["masks"].flatten(1) for v in targets]).float()  # [total_targets, H*W]
        
        # 处理批次大小不匹配问题
        if bs * num_queries != mask_bs * mask_nq:
            print(f"批次大小不匹配: logits={bs*num_queries}, masks={mask_bs*mask_nq}")
            
            # 方案1: 调整 cost_mask 和 cost_dice 以匹配 cost_class
            if mask_bs * mask_nq < bs * num_queries and (bs * num_queries) % (mask_bs * mask_nq) == 0:
                # 计算实际掩码成本
                base_mask_cost = F.binary_cross_entropy_with_logits(
                    out_mask.unsqueeze(1).repeat(1, tgt_mask.shape[0], 1),
                    tgt_mask.unsqueeze(0).repeat(out_mask.shape[0], 1, 1),
                    reduction="none"
                ).mean(2)  # [mask_bs*mask_nq, total_targets]
                
                # 复制成本以匹配 cost_class
                repeat_factor = (bs * num_queries) // (mask_bs * mask_nq)
                cost_mask = base_mask_cost.repeat_interleave(repeat_factor, dim=0)
                
                # 为 cost_dice 计算单独的成本
                # 首先实现正确的成对 dice_loss (不返回平均值)
                dice_costs = []
                for i in range(out_mask.shape[0]):
                    i_costs = []
                    for j in range(tgt_mask.shape[0]):
                        # 单对dice损失
                        pred = out_mask[i:i+1].sigmoid()  # 确保应用sigmoid
                        target = tgt_mask[j:j+1]
                        
                        numerator = 2 * (pred * target).sum()
                        denominator = pred.sum() + target.sum()
                        dice_cost = 1 - (numerator + 1) / (denominator + 1)
                        i_costs.append(dice_cost)
                    dice_costs.append(torch.stack(i_costs))
                
                base_dice_cost = torch.stack(dice_costs)  # [mask_bs*mask_nq, total_targets]
                cost_dice = base_dice_cost.repeat_interleave(repeat_factor, dim=0)
            else:
                # 复杂的批次关系，创建空成本矩阵
                print("警告: 复杂的批次关系，创建空成本矩阵")
                cost_mask = torch.zeros_like(cost_class)
                cost_dice = torch.zeros_like(cost_class)
        else:
            # 批次大小匹配的标准情况
            cost_mask = F.binary_cross_entropy_with_logits(
                out_mask.unsqueeze(1).repeat(1, tgt_mask.shape[0], 1),
                tgt_mask.unsqueeze(0).repeat(out_mask.shape[0], 1, 1),
                reduction="none"
            ).mean(2)  # [bs*num_queries, total_targets]
            
            # 为每对计算 dice 损失
            dice_costs = []
            for i in range(out_mask.shape[0]):
                i_costs = []
                for j in range(tgt_mask.shape[0]):
                    # 单对dice损失
                    pred = out_mask[i:i+1].sigmoid()
                    target = tgt_mask[j:j+1]
                    
                    numerator = 2 * (pred * target).sum()
                    denominator = pred.sum() + target.sum()
                    dice_cost = 1 - (numerator + 1) / (denominator + 1)
                    i_costs.append(dice_cost)
                dice_costs.append(torch.stack(i_costs))
            
            cost_dice = torch.stack(dice_costs)  # [bs*num_queries, total_targets]
        
        # 打印成本形状以验证
        print(f"cost_class: {cost_class.shape}")
        print(f"cost_mask: {cost_mask.shape}")
        print(f"cost_dice: {cost_dice.shape}")
        
        # 确保所有成本形状一致
        assert cost_class.shape == cost_mask.shape == cost_dice.shape, \
            "成本形状不匹配"
        
        # 最终成本计算
        C = (
            self.cost_class * cost_class +
            self.cost_mask * cost_mask +
            self.cost_dice * cost_dice
        )
        
        # 重塑为每个图像的成本矩阵
        C = C.view(bs, num_queries, -1).cpu()
        
        # 执行匹配
        sizes = [len(v["labels"]) for v in targets]
        indices = []
        
        # 为每个图像执行匹配
        for i, (c, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            if size > 0:
                cost_matrix = c[i][:, :size]
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                            torch.as_tensor(col_ind, dtype=torch.int64)))
            else:
                indices.append((torch.tensor([], dtype=torch.int64), 
                            torch.tensor([], dtype=torch.int64)))
        
        return indices
    
class SetCriterion(nn.Module):
    """
    Main loss function
    (主要损失函数)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, losses=('labels', 'masks', 'prompt')):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 为no-object类创建权重
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        (分类损失 - 负对数似然)
        """
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device)
        )
        return {'loss_ce': loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask loss: Focal loss + Dice loss
        (掩码损失: Focal损失 + Dice损失)
        """
        # 打印调试信息
        for idx, (t, ind) in enumerate(zip(targets, indices)):
            print(f"Target {idx}: masks shape={t['masks'].shape}, indices={ind[1]}")
        
        # 安全地获取源和目标索引
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        # 如果没有有效匹配，返回零损失
        if len(src_idx[0]) == 0 or len(tgt_idx[0]) == 0:
            return {
                "loss_mask": torch.tensor(0.0, device=outputs["pred_logits"].device),
                "loss_dice": torch.tensor(0.0, device=outputs["pred_logits"].device)
            }
        
        # 安全地获取预测掩码和目标掩码
        try:
            src_masks = outputs["pred_masks"]
            
            # 检查索引是否在有效范围内
            max_batch = src_masks.shape[0] - 1
            max_query = src_masks.shape[1] - 1
            
            valid_batch_indices = []
            valid_query_indices = []
            valid_target_masks = []
            
            # 为每个图像单独处理
            for b_idx, (t, (src, tgt)) in enumerate(zip(targets, indices)):
                if len(src) == 0 or len(tgt) == 0:
                    continue
                    
                # 验证源索引在范围内
                valid_src = src[src <= max_query]
                if len(valid_src) == 0:
                    continue
                    
                # 验证目标索引在范围内
                valid_tgt = tgt[tgt < t["masks"].shape[0]]
                if len(valid_tgt) == 0:
                    continue
                    
                # 只使用两者都有效的索引
                min_valid = min(len(valid_src), len(valid_tgt))
                valid_src = valid_src[:min_valid]
                valid_tgt = valid_tgt[:min_valid]
                
                # 收集有效索引
                for s, t_idx in zip(valid_src, valid_tgt):
                    valid_batch_indices.append(b_idx)
                    valid_query_indices.append(s.item())
                    valid_target_masks.append(t["masks"][t_idx])
            
            if not valid_target_masks:
                return {
                    "loss_mask": torch.tensor(0.0, device=outputs["pred_logits"].device),
                    "loss_dice": torch.tensor(0.0, device=outputs["pred_logits"].device)
                }
            
            # 转换为张量索引
            batch_idx = torch.tensor(valid_batch_indices, device=src_masks.device)
            query_idx = torch.tensor(valid_query_indices, device=src_masks.device)
            
            # 获取预测掩码
            src_masks = src_masks[batch_idx, query_idx].flatten(1)
            
            # 获取目标掩码
            target_masks = torch.stack(valid_target_masks).to(src_masks.device)
            target_masks = target_masks.flatten(1)
            
            # 确保掩码是浮点类型
            src_masks = src_masks.float()
            target_masks = target_masks.float()
            
            # 检查掩码形状是否匹配，并进行必要的大小调整
            if src_masks.shape[-1] != target_masks.shape[-1]:
                print(f"调整掩码大小 - src_masks: {src_masks.shape}, target_masks: {target_masks.shape}")
                
                # 选择调整方法 - 这里选择调整预测掩码以匹配目标掩码大小
                if target_masks.shape[-1] > src_masks.shape[-1]:
                    # 上采样预测掩码
                    src_masks = F.interpolate(
                        src_masks.unsqueeze(1),  # 添加通道维度
                        size=target_masks.shape[-1],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # 移除通道维度
                else:
                    # 下采样目标掩码
                    target_masks = F.interpolate(
                        target_masks.unsqueeze(1),  # 添加通道维度
                        size=src_masks.shape[-1],
                        mode='nearest'  # 对掩码使用最近邻插值保持二值性
                    ).squeeze(1)  # 移除通道维度
            
            # 计算损失
            loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_boxes=len(valid_target_masks))
            loss_dice = dice_loss(src_masks, target_masks, num_boxes=len(valid_target_masks))
            
            losses = {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice
            }
            
            # 处理粗掩码损失
            if "coarse_masks" in outputs and outputs["coarse_masks"] is not None:
                try:
                    coarse_masks = outputs["coarse_masks"][batch_idx, query_idx].flatten(1).float()
                    
                    # 确保粗掩码和目标掩码形状匹配
                    if coarse_masks.shape[-1] != target_masks.shape[-1]:
                        print(f"调整粗掩码大小 - coarse_masks: {coarse_masks.shape}, target_masks: {target_masks.shape}")
                        
                        # 调整粗掩码以匹配目标掩码
                        coarse_masks = F.interpolate(
                            coarse_masks.unsqueeze(1),  # 添加通道维度
                            size=target_masks.shape[-1],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)  # 移除通道维度
                    
                    loss_coarse_mask = sigmoid_focal_loss(coarse_masks, target_masks, num_boxes=len(valid_target_masks))
                    loss_coarse_dice = dice_loss(coarse_masks, target_masks, num_boxes=len(valid_target_masks))
                    
                    losses["loss_coarse_mask"] = loss_coarse_mask
                    losses["loss_coarse_dice"] = loss_coarse_dice
                except Exception as e:
                    print(f"处理粗掩码时出错(已尝试处理): {e}")
                    # 如果出错，不计算粗掩码损失
                    losses["loss_coarse_mask"] = torch.tensor(0.0, device=src_masks.device)
                    losses["loss_coarse_dice"] = torch.tensor(0.0, device=src_masks.device)
            
            return losses
            
        except Exception as e:
            print(f"计算掩码损失时出错: {e}")
            return {
                "loss_mask": torch.tensor(0.0, device=outputs["pred_logits"].device),
                "loss_dice": torch.tensor(0.0, device=outputs["pred_logits"].device)
            }
    
    def loss_prompt(self, outputs, targets, indices, num_masks):
        """Prompt embedding regularization
        (提示嵌入正则化)
        """
        # 简单的L2正则化
        if "prompt_embeddings" in outputs:
            prompt_embeddings = outputs["prompt_embeddings"]
            loss_prompt = torch.norm(prompt_embeddings, p=2, dim=-1).mean()
            return {"loss_prompt": loss_prompt}
        
        return {"loss_prompt": torch.tensor(0.0, device=outputs["pred_logits"].device)}
    
    # def _get_src_permutation_idx(self, indices):
    #     batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    #     src_idx = torch.cat([src for (src, _) in indices])
    #     return batch_idx, src_idx

    # def _get_tgt_permutation_idx(self, indices):
    #     batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    #     tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    #     return batch_idx, tgt_idx
    def _get_src_permutation_idx(self, indices):
        """获取源排列索引（安全版本）"""
        # 过滤空索引
        valid_indices = [(i, (src, _)) for i, (src, _) in enumerate(indices) if len(src) > 0]
        
        if not valid_indices:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        
        batch_idx = []
        src_idx = []
        
        for i, (src, _) in valid_indices:
            batch_idx.extend([i] * len(src))
            src_idx.extend(src.tolist())
        
        return torch.tensor(batch_idx, dtype=torch.int64), torch.tensor(src_idx, dtype=torch.int64)

    def _get_tgt_permutation_idx(self, indices):
        """获取目标排列索引（安全版本）"""
        # 过滤空索引
        valid_indices = [(i, (_, tgt)) for i, (_, tgt) in enumerate(indices) if len(tgt) > 0]
        
        if not valid_indices:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        
        batch_idx = []
        tgt_idx = []
        
        for i, (_, tgt) in valid_indices:
            batch_idx.extend([i] * len(tgt))
            tgt_idx.extend(tgt.tolist())
        
        return torch.tensor(batch_idx, dtype=torch.int64), torch.tensor(tgt_idx, dtype=torch.int64)
    
    def forward(self, outputs, targets):
        # 匹配预测和目标
        indices = self.matcher(outputs, targets)
        
        # 计算掩码数量用于归一化
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 计算每种损失
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_masks))
        
        # 应用权重
        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                # 移除不在权重字典中的损失
                losses.pop(k)
        
        return losses