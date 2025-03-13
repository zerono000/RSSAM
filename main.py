import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
import time
import numpy as np
from pathlib import Path
from easydict import EasyDict
from tqdm import tqdm
import logging
import datetime
import math
from skimage import measure

from models.rsprompter import RSPrompter
from utils.losses import HungarianMatcher, SetCriterion
from dataset import build_whu_dataloader

# 设置日志
def setup_logger(log_dir):
    """
    Set up logger
    (设置日志器)
    """
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
    Load configuration from yaml file
    (从yaml文件加载配置)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 添加实例分割相关配置
    if 'model' in config and 'per_query_mask' not in config['model']:
        config['model']['per_query_mask'] = True  # 默认开启每查询掩码
    if 'model' in config and 'max_queries_per_batch' not in config['model']:
        config['model']['max_queries_per_batch'] = 10  # 默认批处理大小
    
    return EasyDict(config)

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Save model checkpoint
    (保存模型检查点)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(state, best_path)

def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, logger, accumulation_steps=4):
    """
    训练模型一个epoch(使用梯度累积)
    
    Args:
        model: 训练的模型
        criterion: 损失函数
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 使用的设备
        epoch: 当前epoch
        logger: 日志记录器
        accumulation_steps: 梯度累积步骤数
        
    Returns:
        avg_loss: 平均损失
        avg_loss_dict: 各损失组件的平均值
    """
    model.train()
    total_loss = 0
    epoch_loss_dict = {}
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()  # 初始化梯度
    
    for i, batch in enumerate(pbar):
        # 将数据移到设备上
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 准备目标字典列表
        targets = []
        for idx in range(images.shape[0]):
            # 将背景(0)设为忽略类别，建筑物(1)设为第0类
            binary_mask = masks[idx] > 0
            
            # 实例分割：需要将连通区域分离为不同实例
            labeled_mask = measure.label(binary_mask.cpu().numpy())
            unique_labels = np.unique(labeled_mask)
            unique_labels = unique_labels[unique_labels != 0]  # 移除背景
            
            instance_masks = []
            for label in unique_labels:
                instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                instance_masks.append(instance_mask)
            
            # 如果没有实例，添加一个空目标
            if len(instance_masks) == 0:
                targets.append({
                    'labels': torch.zeros(0, dtype=torch.int64, device=device),
                    'masks': torch.zeros((0, binary_mask.shape[0], binary_mask.shape[1]), 
                                    dtype=torch.bool, device=device)
                })
            else:
                targets.append({
                    'labels': torch.zeros(len(instance_masks), dtype=torch.int64, device=device),  # 所有实例都是建筑物(类别0)
                    'masks': torch.stack(instance_masks)
                })
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 汇总损失
        loss = sum(losses.values())
        
        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
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
        
        # 更新进度条
        pbar.set_postfix({"loss": loss.item()})
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
    
    # 记录损失
    logger.info(f"Epoch {epoch} [Train] Loss: {avg_loss:.4f}")
    for k, v in avg_loss_dict.items():
        logger.info(f"Epoch {epoch} [Train] {k}: {v:.4f}")
    
    return avg_loss, avg_loss_dict

def validate(model, criterion, dataloader, device, epoch, logger):
    """
    Validate model
    (验证模型)
    """
    model.eval()
    total_loss = 0
    epoch_loss_dict = {}
    
    # 评估指标
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            # 将数据移到设备上
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 准备目标字典列表 (与训练循环相同)
            targets = []
            for idx in range(images.shape[0]):
                binary_mask = masks[idx] > 0
                
                labeled_mask = measure.label(binary_mask.cpu().numpy())
                unique_labels = np.unique(labeled_mask)
                unique_labels = unique_labels[unique_labels != 0]
                
                instance_masks = []
                for label in unique_labels:
                    instance_mask = torch.from_numpy(labeled_mask == label).to(device)
                    instance_masks.append(instance_mask)
                
                if len(instance_masks) == 0:
                    targets.append({
                        'labels': torch.zeros(0, dtype=torch.int64, device=device),
                        'masks': torch.zeros((0, binary_mask.shape[0], binary_mask.shape[1]), 
                                          dtype=torch.bool, device=device)
                    })
                else:
                    targets.append({
                        'labels': torch.zeros(len(instance_masks), dtype=torch.int64, device=device),
                        'masks': torch.stack(instance_masks)
                    })
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            losses = criterion(outputs, targets)
            
            # 汇总损失
            loss = sum(losses.values())
            
            # 更新损失统计
            total_loss += loss.item()
            
            # 更新每个损失组件的统计
            for k, v in losses.items():
                if k not in epoch_loss_dict:
                    epoch_loss_dict[k] = 0
                epoch_loss_dict[k] += v.item()
            
            # 计算评估指标
            pred_masks = outputs['pred_masks'].sigmoid() > 0.5  # [B, N, 1, H, W]
            pred_classes = outputs['pred_logits'].argmax(-1)    # [B, N]
            
            for idx in range(images.shape[0]):
                # 只考虑模型预测为建筑物的掩码
                valid_pred_indices = (pred_classes[idx] == 0)
                
                # 确保有效的形状处理
                if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                    pred_instance_masks = pred_masks[idx, valid_pred_indices, 0]  # [num_valid, H, W]
                else:  # 兼容旧版输出形状 [B, 1, H, W]
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
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
    
    # 计算IoU和Dice评分
    iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
    dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-10)
    
    # 记录损失和指标
    logger.info(f"Epoch {epoch} [Val] Loss: {avg_loss:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    for k, v in avg_loss_dict.items():
        logger.info(f"Epoch {epoch} [Val] {k}: {v:.4f}")
    
    return avg_loss, iou, dice

def test(model, dataloader, device, logger):
    """
    Test model
    (测试模型)
    """
    model.eval()
    
    # 评估指标
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc="Testing")
    
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
                # 只考虑模型预测为建筑物的掩码
                valid_pred_indices = (pred_classes[idx] == 0)
                
                # 确保有效的形状处理
                if len(pred_masks.shape) == 5:  # [B, N, 1, H, W]
                    pred_instance_masks = pred_masks[idx, valid_pred_indices, 0]  # [num_valid, H, W]
                else:  # 兼容旧版输出形状 [B, 1, H, W]
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
    
    # 计算评估指标
    iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
    dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-10)
    
    # 记录指标
    logger.info(f"Test Results - IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    return iou, dice

def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info(f"使用配置文件: {args.config} (Using config file)")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device} (Using device)")
    
    # 设置随机种子以确保可重现性
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        logger.info(f"随机种子设置为: {args.seed} (Random seed set)")
    
    # 加载数据
    logger.info("加载数据... (Loading data...)")
    dataloaders = build_whu_dataloader(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    logger.info(f"训练集大小: {len(train_loader.dataset)} 样本 (Training set size)")
    logger.info(f"验证集大小: {len(val_loader.dataset)} 样本 (Validation set size)")
    logger.info(f"测试集大小: {len(test_loader.dataset)} 样本 (Test set size)")
    
    # 创建模型
    logger.info("初始化模型... (Initializing model...)")
    model = RSPrompter(config)
    model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,} (Total parameters)")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%) (Trainable parameters)")
    
    # 配置匹配器和损失计算器
    matcher = HungarianMatcher(
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
            logger.info(f"加载检查点: {args.resume} (Loading checkpoint)")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info(f"成功加载检查点, 从epoch {start_epoch}继续训练 (Resuming from epoch)")
        else:
            logger.warning(f"未找到检查点: {args.resume} (Checkpoint not found)")
    
    # 训练循环
    logger.info("开始训练... (Starting training...)")
    for epoch in range(start_epoch, config.training.epochs):
        # 训练一个epoch
        epoch_start_time = time.time()
        train_loss, train_loss_dict = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, logger
        )
        
        # 更新学习率
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} 学习率: {current_lr:.7f} (Learning rate)")
        
        # 验证
        val_loss, val_iou, val_dice = validate(
            model, criterion, val_loader, device, epoch, logger
        )
        
        # 保存检查点
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        
        save_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            },
            is_best,
            args.checkpoint_dir,
            filename=f"checkpoint_epoch_{epoch}.pth"
        )
        
        # 清理旧检查点，只保留最新的N个
        if args.save_last_n > 0:
            checkpoints = sorted(Path(args.checkpoint_dir).glob("checkpoint_epoch_*.pth"), 
                               key=lambda x: int(x.stem.split('_')[-1]))
            if len(checkpoints) > args.save_last_n:
                for checkpoint in checkpoints[:-args.save_last_n]:
                    os.remove(checkpoint)
        
        # 记录本epoch的总时间
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} 耗时: {epoch_time:.2f}秒 (Epoch time)")
    
    # 加载最佳模型进行测试
    logger.info("加载最佳模型进行测试... (Loading best model for testing...)")
    best_model_path = os.path.join(args.checkpoint_dir, "model_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.info(f"成功加载最佳模型, IoU: {checkpoint['best_iou']:.4f} (Best model loaded)")
        
        # 测试
        test_iou, test_dice = test(model, test_loader, device, logger)
        logger.info(f"最终测试结果 - IoU: {test_iou:.4f}, Dice: {test_dice:.4f} (Final test results)")
    else:
        logger.warning("找不到最佳模型，跳过测试 (Best model not found, skipping test)")
    
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
    
    args = parser.parse_args()
    main(args)