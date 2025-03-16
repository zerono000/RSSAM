import os
import yaml
import time
import logging
import datetime
import argparse
import random
import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from skimage import measure
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.rsprompter import RSPrompter
from utils.losses import HungarianMatcher, SetCriterion, EnhancedHungarianMatcher
from dataset import build_dataloader, get_train_transform, get_val_transform, get_test_transform
from dataset import SemanticSegmentationDataset

config_global = None

# 设置随机种子函数，确保多进程中结果可复现
def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置日志
def setup_logger(log_dir, rank=0):
    """
    设置日志器，只在主进程中创建和输出日志
    """
    if rank != 0:
        # 在非主进程中禁用打印
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass
        return None
        
    logger = logging.getLogger("RSPrompter")
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    log_file = os.path.join(log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path):
    """
    从yaml文件加载配置
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 添加实例分割相关配置
    if 'model' in config and 'per_query_mask' not in config['model']:
        config['model']['per_query_mask'] = True  # 默认开启每查询掩码
    if 'model' in config and 'max_queries_per_batch' not in config['model']:
        config['model']['max_queries_per_batch'] = 10  # 默认批处理大小
    
    return EasyDict(config)

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth", rank=0):
    """
    保存模型检查点 - 只在主进程中执行
    """
    if rank != 0:
        return
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(state, best_path)

def build_distributed_dataloader(config, distributed=False, rank=0, world_size=1):
    """
    构建支持分布式训练的数据加载器
    """
    # 创建数据集
    train_dataset = SemanticSegmentationDataset(
        root_dir=config.data.root_dir,
        split='train',
        transform=get_train_transform(input_size=config.data.input_size)
    )
    
    val_dataset = SemanticSegmentationDataset(
        root_dir=config.data.root_dir,
        split='val',
        transform=get_val_transform(input_size=config.data.input_size)
    )
    
    test_dataset = SemanticSegmentationDataset(
        root_dir=config.data.root_dir,
        split='test',
        transform=get_test_transform(input_size=config.data.input_size)
    )
    
    # 为训练集创建分布式采样器
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),  # 当使用分布式采样器时不需要shuffle
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的批次，避免分布式训练中批次大小不一致
    )
    
    # 验证和测试集加载器，不需要分布式采样
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_sampler': train_sampler
    }

def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, logger, accumulation_steps=4, distributed=False, rank=0):
    """
    训练模型一个epoch, 支持二分类或多类别模式
    """
    model.train()
    total_loss = 0
    epoch_loss_dict = {}
    
    # 使用分布式采样器时设置当前epoch
    if distributed and hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    # 进度条设置
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    else:
        pbar = dataloader
    
    optimizer.zero_grad()  # 初始化梯度
    
    # 获取处理模式和类别数
    binary_mode = config_global.dataset.binary_mode
    num_classes = config_global.dataset.num_classes
    
    for i, batch in enumerate(pbar):
        # 获取数据
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 打印掩码统计信息 (仅第一批次)
        # if i == 0 and rank == 0:
        #     unique_values = masks.unique().cpu().numpy()
        #     logger.info(f"掩码值分布: {unique_values}")
        
        # 准备目标实例列表
        targets = []
        
        for idx in range(images.shape[0]):
            # 根据处理模式选择不同的实例提取逻辑
            if binary_mode:
                # ---------- 二分类模式 ----------
                # 直接使用二值掩码，背景为0，前景为1
                binary_mask = masks[idx] > 0
                
                # 连通区域分析 - 找出独立实例
                labeled_mask = measure.label(binary_mask.cpu().numpy())
                unique_labels = np.unique(labeled_mask)
                unique_labels = unique_labels[unique_labels != 0]  # 移除背景(0)
                
                # 收集所有实例掩码
                instance_masks = []
                instance_labels = []
                
                for label in unique_labels:
                    instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                    
                    # 忽略太小的实例
                    if instance_mask.sum() > 10:
                        instance_masks.append(instance_mask)
                        instance_labels.append(0)  # 二分类中前景类别为0
            else:
                # ---------- 多类别模式 ----------
                instance_masks = []
                instance_labels = []
                
                # 遍历每个有效类别 (从1开始，0是背景)
                for class_id in range(1, num_classes):
                    # 提取当前类别的掩码
                    class_mask = (masks[idx] == class_id)
                    
                    # 如果当前类别不存在，跳过
                    if not torch.any(class_mask):
                        continue
                    
                    # 连通区域分析 - 为同一类别的不同实例创建掩码
                    labeled_mask = measure.label(class_mask.cpu().numpy())
                    unique_labels = np.unique(labeled_mask)
                    unique_labels = unique_labels[unique_labels != 0]
                    
                    # 为每个实例创建掩码并分配类别
                    for label in unique_labels:
                        instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                        
                        # 忽略太小的实例
                        if instance_mask.sum() > 10:
                            instance_masks.append(instance_mask)
                            instance_labels.append(class_id - 1)  # 内部表示从0开始
            
            # 创建目标字典
            if len(instance_masks) == 0:
                # 如果没有找到实例，提供空目标
                targets.append({
                    'labels': torch.zeros(0, dtype=torch.int64, device=device),
                    'masks': torch.zeros((0, masks.shape[1], masks.shape[2]), 
                                        dtype=torch.bool, device=device)
                })
            else:
                # 将实例掩码堆叠为单个张量
                masks_tensor = torch.stack(instance_masks).bool()
                labels_tensor = torch.tensor(instance_labels, dtype=torch.int64, device=device)
                
                targets.append({
                    'labels': labels_tensor,
                    'masks': masks_tensor
                })
                
                # 调试信息 (仅首批次首图像)
                if i == 0 and idx == 0 and rank == 0:
                    # mode_str = "二分类" if binary_mode else "多类别"
                    # logger.info(f"[{mode_str}模式] 图像 {idx} 找到 {len(instance_masks)} 个实例")
                    
                    if not binary_mode and len(instance_labels) > 0:
                        class_counts = {}
                        for label in instance_labels:
                            label_value = label if isinstance(label, int) else label.item()
                            class_counts[label_value] = class_counts.get(label_value, 0) + 1
                        logger.info(f"类别分布: {class_counts}")
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 汇总损失
        loss = sum(losses.values())
        
        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0 and logger:
                logger.warning(f"批次 {i}: 检测到无效损失 {loss}，跳过此批次")
            continue
            
        # 更新损失统计
        total_loss += loss.item()
        
        # 更新每个损失组件的统计
        for k, v in losses.items():
            if k not in epoch_loss_dict:
                epoch_loss_dict[k] = 0
            epoch_loss_dict[k] += v.item()
        
        # 缩放损失以匹配梯度累积
        scaled_loss = loss / accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 每累积足够步骤后更新参数
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            # 梯度裁剪以防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # 清理缓存
            torch.cuda.empty_cache()
        
        # 更新进度条 - 只在主进程
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({"loss": loss.item()})
    
    # 在分布式训练中同步损失统计
    if distributed:
        # 收集所有进程的总损失
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor)
        total_loss = loss_tensor.item() / dist.get_world_size()
        
        # 收集所有损失组件
        for k in list(epoch_loss_dict.keys()):
            value_tensor = torch.tensor([epoch_loss_dict[k]], device=device)
            dist.all_reduce(value_tensor)
            epoch_loss_dict[k] = value_tensor.item() / dist.get_world_size()
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
    
    # 记录损失 - 只在主进程
    if rank == 0 and logger:
        logger.info(f"Epoch {epoch} [Train] Loss: {avg_loss:.4f}")
        for k, v in avg_loss_dict.items():
            logger.info(f"Epoch {epoch} [Train] {k}: {v:.4f}")
    
    return avg_loss, avg_loss_dict

def validate(model, criterion, dataloader, device, epoch, logger, distributed=False, rank=0):
    """
    验证模型性能，自适应支持二分类或多类别模式
    """
    model.eval()
    total_loss = 0
    epoch_loss_dict = {}
    
    # 获取处理模式和类别数
    binary_mode = config_global.dataset.binary_mode
    num_classes = config_global.dataset.num_classes
    
    # 初始化评估指标
    if binary_mode:
        # 二分类模式 - 只有一组指标
        tp_total = 0
        fp_total = 0
        fn_total = 0
    else:
        # 多类别模式 - 每个类别一组指标
        tp_per_class = [0] * (num_classes - 1)  # 不包括背景类
        fp_per_class = [0] * (num_classes - 1)
        fn_per_class = [0] * (num_classes - 1)
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            # 将数据移到设备上
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 准备目标字典列表
            targets = []
            for idx in range(images.shape[0]):
                if binary_mode:
                    # ---------- 二分类模式 ----------
                    binary_mask = masks[idx] > 0
                    
                    # 连通区域分析
                    labeled_mask = measure.label(binary_mask.cpu().numpy())
                    unique_labels = np.unique(labeled_mask)
                    unique_labels = unique_labels[unique_labels != 0]
                    
                    instance_masks = []
                    instance_labels = []
                    
                    for label in unique_labels:
                        instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                        if instance_mask.sum() > 10:
                            instance_masks.append(instance_mask)
                            instance_labels.append(0)  # 二分类中前景类别为0
                else:
                    # ---------- 多类别模式 ----------
                    instance_masks = []
                    instance_labels = []
                    
                    for class_id in range(1, num_classes):
                        class_mask = (masks[idx] == class_id)
                        if not torch.any(class_mask):
                            continue
                        
                        labeled_mask = measure.label(class_mask.cpu().numpy())
                        unique_labels = np.unique(labeled_mask)
                        unique_labels = unique_labels[unique_labels != 0]
                        
                        for label in unique_labels:
                            instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                            if instance_mask.sum() > 10:
                                instance_masks.append(instance_mask)
                                instance_labels.append(class_id - 1)
                
                if len(instance_masks) == 0:
                    targets.append({
                        'labels': torch.zeros(0, dtype=torch.int64, device=device),
                        'masks': torch.zeros((0, masks.shape[1], masks.shape[2]), 
                                          dtype=torch.bool, device=device)
                    })
                else:
                    masks_tensor = torch.stack(instance_masks).bool()
                    labels_tensor = torch.tensor(instance_labels, dtype=torch.int64, device=device)
                    
                    targets.append({
                        'labels': labels_tensor,
                        'masks': masks_tensor
                    })
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            losses = criterion(outputs, targets)
            loss = sum(losses.values())
            
            # 更新损失统计
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in epoch_loss_dict:
                    epoch_loss_dict[k] = 0
                epoch_loss_dict[k] += v.item()
            
            # 预测后处理
            pred_masks = outputs['pred_masks'].sigmoid() > 0.5  # [B, N, 1, H, W]
            pred_classes = outputs['pred_logits'].argmax(-1)    # [B, N]
            
            for idx in range(images.shape[0]):
                if binary_mode:
                    # ---------- 二分类模式评估 ----------
                    # 只考虑模型预测为前景的掩码 (类别0)
                    valid_pred_indices = (pred_classes[idx] == 0)
                    
                    # 确保有效的形状处理
                    if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                        pred_instance_masks = pred_masks[idx, valid_pred_indices, 0]
                    else:  # 兼容旧版输出形状
                        pred_instance_masks = pred_masks[idx]
                    
                    # 将预测掩码合并为一个二值掩码
                    if pred_instance_masks.shape[0] > 0:
                        pred_binary = torch.any(pred_instance_masks, dim=0)
                    else:
                        pred_binary = torch.zeros_like(masks[idx], dtype=torch.bool)
                    
                    # 获取真实二值掩码
                    true_binary = masks[idx] > 0
                    
                    # 计算TP, FP, FN
                    tp = torch.logical_and(pred_binary, true_binary).sum().item()
                    fp = torch.logical_and(pred_binary, ~true_binary).sum().item()
                    fn = torch.logical_and(~pred_binary, true_binary).sum().item()
                    
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
                else:
                    # ---------- 多类别模式评估 ----------
                    # 为每个类别评估性能
                    for class_id in range(1, num_classes):
                        # 创建当前类别的真实掩码 (二值)
                        true_class_mask = (masks[idx] == class_id)
                        
                        # 获取当前类别的预测实例索引 (类别ID-1)
                        valid_pred_indices = (pred_classes[idx] == (class_id - 1))
                        
                        # 合并当前类别的所有预测实例为一个二值掩码
                        if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                            class_pred_instances = pred_masks[idx, valid_pred_indices, 0]
                        else:  # 兼容旧版输出形状
                            class_pred_instances = pred_masks[idx]
                        
                        # 将预测实例合并为一个二值掩码
                        if class_pred_instances.shape[0] > 0:
                            pred_class_mask = torch.any(class_pred_instances, dim=0)
                        else:
                            pred_class_mask = torch.zeros_like(masks[idx], dtype=torch.bool)
                        
                        # 计算类别级别的TP, FP, FN
                        tp = torch.logical_and(pred_class_mask, true_class_mask).sum().item()
                        fp = torch.logical_and(pred_class_mask, ~true_class_mask).sum().item()
                        fn = torch.logical_and(~pred_class_mask, true_class_mask).sum().item()
                        
                        # 更新统计 (类别索引从0开始)
                        tp_per_class[class_id - 1] += tp
                        fp_per_class[class_id - 1] += fp
                        fn_per_class[class_id - 1] += fn
    
    # 在分布式训练中同步指标
    if distributed:
        if binary_mode:
            metrics = torch.tensor([tp_total, fp_total, fn_total], device=device)
            dist.all_reduce(metrics)
            tp_total, fp_total, fn_total = metrics.cpu().numpy()
        else:
            for class_id in range(num_classes - 1):
                metrics = torch.tensor([tp_per_class[class_id], fp_per_class[class_id], fn_per_class[class_id]], device=device)
                dist.all_reduce(metrics)
                tp_per_class[class_id], fp_per_class[class_id], fn_per_class[class_id] = metrics.cpu().numpy()
        
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor)
        total_loss = loss_tensor.item() / dist.get_world_size()
        
        for k in list(epoch_loss_dict.keys()):
            value_tensor = torch.tensor([epoch_loss_dict[k]], device=device)
            dist.all_reduce(value_tensor)
            epoch_loss_dict[k] = value_tensor.item() / dist.get_world_size()
    
    # 计算评估指标
    if binary_mode:
        # 二分类模式 - 计算整体IoU和Dice
        iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
        dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-10)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
        
        # 记录结果 - 只在主进程
        if rank == 0 and logger:
            logger.info(f"Epoch {epoch} [Val] Loss: {avg_loss:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
            for k, v in avg_loss_dict.items():
                logger.info(f"Epoch {epoch} [Val] {k}: {v:.4f}")
        
        return avg_loss, iou, dice
    else:
        # 多类别模式 - 计算每个类别的IoU和Dice
        iou_per_class = []
        dice_per_class = []
        for class_id in range(num_classes - 1):
            tp = tp_per_class[class_id]
            fp = fp_per_class[class_id]
            fn = fn_per_class[class_id]
            
            class_iou = tp / (tp + fp + fn + 1e-10)
            class_dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
            
            iou_per_class.append(class_iou)
            dice_per_class.append(class_dice)
        
        # 计算平均指标
        mean_iou = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0
        mean_dice = sum(dice_per_class) / len(dice_per_class) if dice_per_class else 0
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
        
        # 记录损失和指标 - 只在主进程
        if rank == 0 and logger:
            logger.info(f"Epoch {epoch} [Val] Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
            for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
                class_name = config_global.dataset.class_names[i+1] if hasattr(config_global.dataset, 'class_names') else f"类别{i+1}"
                logger.info(f"Epoch {epoch} [Val] {class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
            for k, v in avg_loss_dict.items():
                logger.info(f"Epoch {epoch} [Val] {k}: {v:.4f}")
        
        return avg_loss, mean_iou, mean_dice
    
def test(model, dataloader, device, logger, distributed=False, rank=0):
    """
    测试模型，自适应支持二分类或多类别模式
    """
    model.eval()
    
    # 获取处理模式和类别数
    binary_mode = config_global.dataset.binary_mode
    num_classes = config_global.dataset.num_classes
    
    # 初始化评估指标
    if binary_mode:
        # 二分类模式 - 只有一组指标
        tp_total = 0
        fp_total = 0
        fn_total = 0
    else:
        # 多类别模式 - 每个类别一组指标
        tp_per_class = [0] * (num_classes - 1)  # 不包括背景类
        fp_per_class = [0] * (num_classes - 1)
        fn_per_class = [0] * (num_classes - 1)
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc="Testing")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
            # 将数据移到设备上
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 预测处理
            pred_masks = outputs['pred_masks'].sigmoid() > 0.5  # [B, N, 1, H, W]
            pred_classes = outputs['pred_logits'].argmax(-1)    # [B, N]
            
            for idx in range(images.shape[0]):
                if binary_mode:
                    # ---------- 二分类模式评估 ----------
                    # 只考虑模型预测为前景的掩码 (类别0)
                    valid_pred_indices = (pred_classes[idx] == 0)
                    
                    # 确保有效的形状处理
                    if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                        pred_instance_masks = pred_masks[idx, valid_pred_indices, 0]
                    else:  # 兼容旧版输出形状
                        pred_instance_masks = pred_masks[idx]
                    
                    # 将预测掩码合并为一个二值掩码
                    if pred_instance_masks.shape[0] > 0:
                        pred_binary = torch.any(pred_instance_masks, dim=0)
                    else:
                        pred_binary = torch.zeros_like(masks[idx], dtype=torch.bool)
                    
                    # 获取真实二值掩码
                    true_binary = masks[idx] > 0
                    
                    # 计算TP, FP, FN
                    tp = torch.logical_and(pred_binary, true_binary).sum().item()
                    fp = torch.logical_and(pred_binary, ~true_binary).sum().item()
                    fn = torch.logical_and(~pred_binary, true_binary).sum().item()
                    
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
                else:
                    # ---------- 多类别模式评估 ----------
                    # 为每个类别评估性能
                    for class_id in range(1, num_classes):
                        # 创建当前类别的真实掩码 (二值)
                        true_class_mask = (masks[idx] == class_id)
                        
                        # 获取当前类别的预测实例索引 (类别ID-1)
                        valid_pred_indices = (pred_classes[idx] == (class_id - 1))
                        
                        # 合并当前类别的所有预测实例为一个二值掩码
                        if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                            class_pred_instances = pred_masks[idx, valid_pred_indices, 0]
                        else:  # 兼容旧版输出形状
                            class_pred_instances = pred_masks[idx]
                        
                        # 将预测实例合并为一个二值掩码
                        if class_pred_instances.shape[0] > 0:
                            pred_class_mask = torch.any(class_pred_instances, dim=0)
                        else:
                            pred_class_mask = torch.zeros_like(masks[idx], dtype=torch.bool)
                        
                        # 计算类别级别的TP, FP, FN
                        tp = torch.logical_and(pred_class_mask, true_class_mask).sum().item()
                        fp = torch.logical_and(pred_class_mask, ~true_class_mask).sum().item()
                        fn = torch.logical_and(~pred_class_mask, true_class_mask).sum().item()
                        
                        # 更新统计
                        tp_per_class[class_id - 1] += tp
                        fp_per_class[class_id - 1] += fp
                        fn_per_class[class_id - 1] += fn
    
    # 在分布式环境中同步指标
    if distributed:
        if binary_mode:
            metrics = torch.tensor([tp_total, fp_total, fn_total], device=device)
            dist.all_reduce(metrics)
            tp_total, fp_total, fn_total = metrics.cpu().numpy()
        else:
            for class_id in range(num_classes - 1):
                metrics = torch.tensor([tp_per_class[class_id], fp_per_class[class_id], fn_per_class[class_id]], device=device)
                dist.all_reduce(metrics)
                tp_per_class[class_id], fp_per_class[class_id], fn_per_class[class_id] = metrics.cpu().numpy()
    
    # 计算评估指标
    if binary_mode:
        # 二分类模式 - 计算整体IoU和Dice
        iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
        dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-10)
        
        # 记录结果 - 只在主进程
        if rank == 0 and logger:
            logger.info(f"Test Results - IoU: {iou:.4f}, Dice: {dice:.4f}")
        
        return iou, dice
    else:
        # 多类别模式 - 计算每个类别的IoU和Dice
        iou_per_class = []
        dice_per_class = []
        for class_id in range(num_classes - 1):
            tp = tp_per_class[class_id]
            fp = fp_per_class[class_id]
            fn = fn_per_class[class_id]
            
            class_iou = tp / (tp + fp + fn + 1e-10)
            class_dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
            
            iou_per_class.append(class_iou)
            dice_per_class.append(class_dice)
        
        # 计算平均指标
        mean_iou = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0
        mean_dice = sum(dice_per_class) / len(dice_per_class) if dice_per_class else 0
        
        # 记录指标 - 只在主进程
        if rank == 0 and logger:
            logger.info(f"Test Results - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
            for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
                class_name = config_global.dataset.class_names[i+1] if hasattr(config_global.dataset, 'class_names') else f"类别{i+1}"
                logger.info(f"Test Results - {class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
        
        return mean_iou, mean_dice

def init_distributed_mode(args):
    """
    初始化分布式训练环境
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('未检测到分布式训练环境变量，使用默认设置')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| 初始化进程组: 地址={args.dist_url}, 排名={args.rank}, 全局大小={args.world_size}')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    在分布式训练中设置打印函数，非主进程不打印
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main(args):
    """
    主训练函数，支持分布式和单机训练
    """
    # 初始化分布式环境
    if args.distributed:
        init_distributed_mode(args)
    
    # 设置机器本地排名
    if args.distributed:
        local_rank = args.gpu
        rank = args.rank
    else:
        local_rank = 0
        rank = 0
    
    # 加载配置
    config = load_config(args.config)

    # 设置全局配置变量
    global config_global
    config_global = config
    
    # 设置日志 - 只在主进程设置
    logger = setup_logger(args.log_dir, rank=rank)
    if rank == 0:
        logger.info(f"使用配置文件: {args.config} (Using config file)")
        if args.distributed:
            logger.info(f"分布式训练: 全局进程数={args.world_size}, 当前进程排名={rank}")
    
    # 设置设备
    if args.distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logger.info(f"使用设备: {device} (Using device)")
    
    # 设置随机种子以确保可重现性
    if args.seed is not None:
        set_seed(args.seed + rank)  # 不同进程使用不同的种子
        if rank == 0:
            logger.info(f"随机种子设置为: {args.seed + rank} (Random seed set)")
    
    # 加载数据
    if rank == 0:
        logger.info("加载数据... (Loading data...)")
    
    dataloaders = build_distributed_dataloader(
        config, 
        distributed=args.distributed,
        rank=rank, 
        world_size=args.world_size if args.distributed else 1
    )
    
    train_loader = dataloaders['train']
    train_sampler = dataloaders['train_sampler']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    if rank == 0:
        logger.info(f"训练集大小: {len(train_loader.dataset)} 样本 (Training set size)")
        logger.info(f"验证集大小: {len(val_loader.dataset)} 样本 (Validation set size)")
        logger.info(f"测试集大小: {len(test_loader.dataset)} 样本 (Test set size)")
    
    # 创建模型
    if rank == 0:
        logger.info("初始化模型... (Initializing model...)")
    
    model = RSPrompter(config)
    model.to(device)
    
    # 使用DDP包装模型用于分布式训练
    if args.distributed:
        # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # 在main函数中添加处理分布式配置的代码
        if 'distributed' in config and args.distributed:
            if hasattr(config.distributed, 'find_unused_parameters'):
                find_unused_parameters = config.distributed.find_unused_parameters
            else:
                find_unused_parameters = True
                
            # 使用配置的参数创建DDP模型
            model = DDP(model, 
                        device_ids=[local_rank], 
                        output_device=local_rank, 
                        find_unused_parameters=find_unused_parameters,
                        gradient_as_bucket_view=getattr(config.distributed, 'gradient_as_bucket_view', False))
    
    # 打印模型信息 - 只在主进程
    if rank == 0:
        # 获取模型参数，如果使用DDP需要访问模块
        if args.distributed:
            model_to_count = model.module
        else:
            model_to_count = model
            
        total_params = sum(p.numel() for p in model_to_count.parameters())
        trainable_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
        
        logger.info(f"总参数量: {total_params:,} (Total parameters)")
        logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) (Trainable parameters)")
    
    # # 配置匹配器和损失计算器
    # matcher = HungarianMatcher(
    #     cost_class=config.training.loss_weights.class_loss_coef,
    #     cost_mask=config.training.loss_weights.mask_loss_coef,
    #     cost_dice=config.training.loss_weights.dice_loss_coef
    # )

    # 配置匹配器和损失计算器
    matcher = EnhancedHungarianMatcher(
        cost_class=config.training.loss_weights.class_loss_coef,
        cost_mask=config.training.loss_weights.mask_loss_coef,
        cost_dice=config.training.loss_weights.dice_loss_coef
    )

    weight_dict = {
        'loss_ce': config.training.loss_weights.class_loss_coef,
        'loss_mask': config.training.loss_weights.mask_loss_coef,
        'loss_dice': config.training.loss_weights.dice_loss_coef,
        'loss_prompt': config.training.loss_weights.prompt_loss_coef
    }

    if config.model.use_coarse_mask:
        weight_dict['loss_coarse_mask'] = config.training.loss_weights.mask_loss_coef
        weight_dict['loss_coarse_dice'] = config.training.loss_weights.dice_loss_coef

    criterion = SetCriterion(
        num_classes=config.dataset.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,  # 背景类权重
        losses=('labels', 'masks', 'prompt')
    )
    criterion.to(device)
    
    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    # 设置学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.training.lr_drop,
        gamma=0.1
    )
    
    # 恢复检查点（如果存在）
    start_epoch = 0
    best_iou = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info(f"加载检查点: {args.resume} (Loading checkpoint)")
            
            # 加载到CPU以避免GPU内存问题
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['best_iou']
            
            # 加载模型权重
            if args.distributed:
                # DDP模型需要加载到module
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            if rank == 0:
                logger.info(f"成功加载检查点, 从epoch {start_epoch}继续训练 (Resuming from epoch)")
        else:
            if rank == 0:
                logger.warning(f"未找到检查点: {args.resume} (Checkpoint not found)")
    
    # 从配置获取类别数
    num_classes = config.dataset.num_classes
    if rank == 0:
        logger.info(f"类别数量: {num_classes} (不包括背景)")
        logger.info(f"类别名称: {config.dataset.class_names}")

    # 训练循环
    if rank == 0:
        logger.info("开始训练... (Starting training...)")
        
    for epoch in range(start_epoch, config.training.epochs):
        # 训练一个epoch
        epoch_start_time = time.time()
        train_loss, train_loss_dict = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, logger, 
            accumulation_steps=config.training.gradient_accumulation_steps,
            distributed=args.distributed, rank=rank
        )
        
        # 更新学习率
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            logger.info(f"Epoch {epoch} 学习率: {current_lr:.7f} (Learning rate)")
        
        # 验证
        val_loss, val_iou, val_dice = validate(
            model, criterion, val_loader, device, epoch, logger,
            distributed=args.distributed, rank=rank
        )
        
        # 保存检查点 - 只在主进程
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        
        if args.distributed:
            # 使用DDP时，保存module的状态
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        save_checkpoint(
            {
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            },
            is_best,
            args.checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch}.pth",
            rank=rank
        )
        
        # 清理旧检查点，只保留最新的N个 - 只在主进程
        if rank == 0 and args.save_last_n > 0:
            checkpoints = sorted(Path(args.checkpoint_dir).glob("checkpoint_epoch_*.pth"), 
                               key=lambda x: int(x.stem.split('_')[-1]))
            if len(checkpoints) > args.save_last_n:
                for checkpoint in checkpoints[:-args.save_last_n]:
                    os.remove(checkpoint)
        
        # 记录本epoch的总时间 - 只在主进程
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} 耗时: {epoch_time:.2f}秒 (Epoch time)")
    
    # 加载最佳模型进行测试 - 只在主进程或所有进程同步
    if rank == 0:
        logger.info("加载最佳模型进行测试... (Loading best model for testing...)")
        
    best_model_path = os.path.join(args.checkpoint_dir, "model_best.pth")
    
    # 等待所有进程
    if args.distributed:
        dist.barrier()
        
    if os.path.exists(best_model_path):
        # 加载到CPU以避免GPU内存问题
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # 加载模型权重
        if args.distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
            
        if rank == 0:
            logger.info(f"成功加载最佳模型, IoU: {checkpoint['best_iou']:.4f} (Best model loaded)")
        
        # 等待所有进程
        if args.distributed:
            dist.barrier()
            
        # 测试
        test_iou, test_dice = test(
            model, test_loader, device, logger,
            distributed=args.distributed, rank=rank
        )
        
        if rank == 0:
            logger.info(f"最终测试结果 - IoU: {test_iou:.4f}, Dice: {test_dice:.4f} (Final test results)")
    else:
        if rank == 0:
            logger.warning("找不到最佳模型，跳过测试 (Best model not found, skipping test)")
    
    # 清理分布式环境
    if args.distributed:
        dist.destroy_process_group()
        
    if rank == 0:
        logger.info("训练完成! (Training completed!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSPrompter Training Script")
    parser.add_argument('--config', type=str, default='config/config.yml', help='配置文件路径 (Path to config file)')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备 (Device to use)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (Random seed)')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录 (Log directory)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录 (Checkpoint directory)')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径 (Path to checkpoint for resuming training)')
    parser.add_argument('--save-last-n', type=int, default=3, help='保留最近的N个检查点, 设为0则保留所有 (Keep only the last N checkpoints)')
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练 (Enable distributed training)')
    parser.add_argument('--world-size', type=int, default=1, help='分布式训练的进程数 (Number of processes for distributed training)')
    parser.add_argument('--dist-url', default='env://', help='分布式训练的URL (URL for distributed training)')
    
    args = parser.parse_args()
    
    # 判断是否需要启用分布式训练
    if torch.cuda.device_count() > 1 and args.distributed:
        args.world_size = torch.cuda.device_count()
        print(f"检测到{args.world_size}个GPU, 启用分布式训练")
    else:
        args.distributed = False
        print(f"检测到{torch.cuda.device_count()}个GPU, 使用单卡训练")
    
    main(args)