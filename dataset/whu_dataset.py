import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class WHUBuildingDataset(Dataset):
    """
    Dataset class for WHU building segmentation dataset
    (WHU建筑物分割数据集类)
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset containing train, test, val folders
            split (str): Split to use ('train', 'test', or 'val')
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Get image and mask directories
        self.image_dir = os.path.join(root_dir, split, 'Image')
        self.mask_dir = os.path.join(root_dir, split, 'Mask')
        
        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith('.png') or f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image and mask paths
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image (using PIL for better compatibility)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask 
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Binarize mask (if not already binary)
        mask_np = (mask_np > 0).astype(np.uint8)
        
        # Prepare sample dictionary
        sample = {
            'image': image_np,
            'mask': mask_np,
            'filename': img_name
        }
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class SemanticSegmentationDataset(Dataset):
    """
    通用语义分割数据集类
    """
    def __init__(self, root_dir, split='train', transform=None, num_classes=None):
        """
        参数:
            root_dir: 数据集根目录
            split: 数据集划分('train', 'val', 'test')
            transform: 数据增强转换
            num_classes: 类别数量
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # 图像和掩码目录
        self.image_dir = os.path.join(root_dir, split, 'Image')
        self.mask_dir = os.path.join(root_dir, split, 'Mask')
        
        # 获取文件列表
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                          if f.endswith('.png') or f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像和掩码路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 加载掩码 (假设掩码是类别索引格式)
        mask = Image.open(mask_path).convert('L')
        
        # 转换为numpy数组
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # 准备样本字典
        sample = {
            'image': image_np,
            'mask': mask_np,
            'filename': img_name
        }
        
        # 应用数据增强
        if self.transform:
            sample = self.transform(sample)
            
        return sample