import torch
from torch.utils.data import DataLoader, random_split
# from .whu_dataset import SemanticSegmentationDataset
from .whu_dataset import SemanticSegmentationDataset
from .transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, Normalize, ToTensor


def get_train_transform(input_size=1024):
    """
    Get transformation for training
    (获取训练数据变换)
    """
    return Compose([
        Resize(input_size, input_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomRotation(p=0.3),
        RandomCrop(height=input_size, width=input_size),
        Normalize(),
        ToTensor()
    ])


def get_val_transform(input_size=1024):
    """
    Get transformation for validation
    (获取验证数据变换)
    """
    return Compose([
        Resize(input_size, input_size),
        Normalize(),
        ToTensor()
    ])


def get_test_transform(input_size=1024):
    """
    Get transformation for testing
    (获取测试数据变换)
    """
    return Compose([
        Resize(input_size, input_size),
        Normalize(),
        ToTensor()
    ])


def build_whu_dataloader(config):
    """
    Build data loaders for WHU building dataset
    (为WHU建筑物数据集构建数据加载器)
    
    Args:
        config: Configuration with dataset parameters
    
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }