import torch
import numpy as np
import random
import cv2
from torchvision import transforms as T


class Resize:
    """
    Resize image and mask to specified size
    (将图像和掩码调整为指定大小)
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class RandomHorizontalFlip:
    """
    Randomly flip image and mask horizontally
    (随机水平翻转图像和掩码)
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class RandomVerticalFlip:
    """
    Randomly flip image and mask vertically
    (随机垂直翻转图像和掩码)
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.p:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class RandomRotation:
    """
    Randomly rotate image and mask by 90 degrees
    (随机旋转图像和掩码90度)
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.p:
            k = random.randint(1, 3)  # 1, 2, or 3 times 90 degrees
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
            
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class RandomCrop:
    """
    Randomly crop image and mask
    (随机裁剪图像和掩码)
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        
        if h > self.height and w > self.width:
            # Randomly choose top-left corner of the crop
            top = random.randint(0, h - self.height)
            left = random.randint(0, w - self.width)
            
            # Crop image and mask
            image = image[top:top+self.height, left:left+self.width]
            mask = mask[top:top+self.height, left:left+self.width]
        
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class Normalize:
    """
    Normalize image with mean and standard deviation
    (使用均值和标准差归一化图像)
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class ToTensor:
    """
    Convert image and mask to PyTorch tensors
    (将图像和掩码转换为PyTorch张量)
    """
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Swap color axis for image because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        sample['image'] = image
        sample['mask'] = mask
        
        return sample


class Compose:
    """
    Compose multiple transforms together
    (将多个变换组合在一起)
    """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample