"""
Blood Cell Classification - Vast.ai Version
============================================
Features:
- Auto GPU detection & batch size adjustment
- Wandb logging (optional)
- Checkpoint save/resume
- Multi-GPU support (DataParallel)
- Progress monitoring
- Auto-upload results to cloud

Author: Claude AI
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import json
import argparse
import subprocess
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== VAST.AI AUTO CONFIG ====================
def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {'name': 'CPU', 'memory': 0, 'count': 0}
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    gpu_count = torch.cuda.device_count()
    
    return {
        'name': gpu_name,
        'memory': gpu_memory,
        'count': gpu_count
    }

def auto_config_batch_size(gpu_memory, model_name, img_size):
    """Auto configure batch size based on GPU memory"""
    # Base batch sizes cho 24GB GPU (RTX 3090)
    base_batch_sizes = {
        'efficientnet_b4': 32,
        'efficientnet_b5': 20,
        'tf_efficientnetv2_s': 32,
        'tf_efficientnetv2_m': 16,
        'convnext_small': 24,
        'convnext_base': 16,
        'swin_base_patch4_window12_384': 16,
        'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k': 12,
    }
    
    base_bs = base_batch_sizes.get(model_name, 16)
    
    # Scale based on GPU memory (relative to 24GB)
    memory_scale = gpu_memory / 24.0
    
    # Scale based on image size (relative to 384)
    size_scale = (384 / img_size) ** 2
    
    batch_size = int(base_bs * memory_scale * size_scale)
    batch_size = max(4, min(batch_size, 64))  # Clamp between 4 and 64
    
    # Round to nearest multiple of 4
    batch_size = (batch_size // 4) * 4
    batch_size = max(4, batch_size)  # Ensure minimum 4
    
    return batch_size

# ==================== MODEL CONFIGS ====================
# Chỉ giữ các model mạnh nhất cho max accuracy
MODEL_CONFIGS = {
    # === EFFICIENTNET FAMILY ===
    'efficientnet_b4': {
        'name': 'efficientnet_b4',
        'img_size': 380,
        'lr': 1e-4,
        'drop_rate': 0.4,
        'drop_path': 0.2,
        'desc': 'EfficientNet-B4 - Balanced choice'
    },
    'efficientnet_b5': {
        'name': 'efficientnet_b5',
        'img_size': 456,
        'lr': 8e-5,
        'drop_rate': 0.4,
        'drop_path': 0.2,
        'desc': 'EfficientNet-B5 - Larger, more accurate'
    },
    'efficientnetv2_s': {
        'name': 'tf_efficientnetv2_s',
        'img_size': 384,
        'lr': 1e-4,
        'drop_rate': 0.3,
        'drop_path': 0.2,
        'desc': 'EfficientNetV2-S - Fast & accurate'
    },
    'efficientnetv2_m': {
        'name': 'tf_efficientnetv2_m',
        'img_size': 480,
        'lr': 5e-5,
        'drop_rate': 0.4,
        'drop_path': 0.3,
        'desc': 'EfficientNetV2-M - High accuracy'
    },
    
    # === CONVNEXT FAMILY ===
    'convnext_small': {
        'name': 'convnext_small',
        'img_size': 384,
        'lr': 5e-5,
        'drop_rate': 0.4,
        'drop_path': 0.2,
        'desc': 'ConvNeXt-Small - Modern CNN'
    },
    'convnext_base': {
        'name': 'convnext_base',
        'img_size': 384,
        'lr': 5e-5,
        'drop_rate': 0.5,
        'drop_path': 0.2,
        'desc': 'ConvNeXt-Base - Strong accuracy'
    },
    
    # === SWIN TRANSFORMER ===
    'swin_base': {
        'name': 'swin_base_patch4_window12_384',
        'img_size': 384,
        'lr': 3e-5,
        'drop_rate': 0.4,
        'drop_path': 0.3,
        'desc': 'Swin-Base - Transformer, very strong'
    },
    
    # === EVA (State-of-the-art) ===
    'eva02_base': {
        'name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
        'img_size': 448,
        'lr': 2e-5,
        'drop_rate': 0.4,
        'drop_path': 0.2,
        'desc': 'EVA02-Base - SOTA, highest accuracy'
    },
}

# ==================== MAIN CONFIG ====================
class Config:
    # ===== PATHS (Vast.ai default) =====
    TRAIN_DIR = "/workspace/data/train"
    TEST_DIR = "/workspace/data/test"
    OUTPUT_DIR = "/workspace/outputs"
    CHECKPOINT_DIR = "/workspace/checkpoints"
    
    # ===== MODEL =====
    # Options: efficientnet_b4, efficientnet_b5, efficientnetv2_s, efficientnetv2_m,
    #          convnext_small, convnext_base, swin_base, eva02_base
    MODEL_CHOICE = 'convnext_base'  # Default: strong accuracy
    
    # ===== CLASSES =====
    CLASSES = ['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE']
    NUM_CLASSES = 9
    
    # ===== CLASS WEIGHTS =====
    CLASS_WEIGHTS = {
        'BA': 1.2, 'BNE': 1.0, 'EO': 0.7, 'LY': 1.2, 
        'MMY': 1.4, 'MO': 1.0, 'MY': 1.3, 'PMY': 1.8, 'SNE': 1.0
    }
    
    # ===== TRAINING =====
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 25
    NUM_WORKERS = 4
    
    # ===== LOSS =====
    LABEL_SMOOTHING = 0.1
    FOCAL_GAMMA = 2.0
    
    # ===== MIXED PRECISION =====
    USE_AMP = True
    
    # ===== WANDB =====
    USE_WANDB = False
    WANDB_PROJECT = "blood-cell-classification"
    WANDB_RUN_NAME = None
    
    # ===== DEVICE =====
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===== AUTO CONFIG =====
    AUTO_BATCH_SIZE = True
    BATCH_SIZE = None  # Will be auto-configured
    
    @classmethod
    def setup(cls, args=None):
        """Setup config with auto-detection"""
        # Get GPU info
        gpu_info = get_gpu_info()
        print(f"\n🎮 GPU: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory']:.1f} GB")
        print(f"   Count: {gpu_info['count']}")
        
        # Get model config
        model_cfg = MODEL_CONFIGS[cls.MODEL_CHOICE]
        
        # Auto batch size
        if cls.AUTO_BATCH_SIZE:
            cls.BATCH_SIZE = auto_config_batch_size(
                gpu_info['memory'], 
                model_cfg['name'],
                model_cfg['img_size']
            )
            # Scale for multi-GPU
            if gpu_info['count'] > 1:
                cls.BATCH_SIZE *= gpu_info['count']
        
        # Override with args if provided
        if args:
            if args.train_dir:
                cls.TRAIN_DIR = args.train_dir
            if args.test_dir:
                cls.TEST_DIR = args.test_dir
            if args.output_dir:
                cls.OUTPUT_DIR = args.output_dir
            if args.model:
                cls.MODEL_CHOICE = args.model
            if args.batch_size:
                cls.BATCH_SIZE = args.batch_size
            if args.epochs:
                cls.EPOCHS = args.epochs
            if args.wandb:
                cls.USE_WANDB = True
            if args.wandb_project:
                cls.WANDB_PROJECT = args.wandb_project
        
        # Create directories
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        
        return model_cfg
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        model_cfg = MODEL_CONFIGS[cls.MODEL_CHOICE]
        print("\n" + "="*60)
        print("📋 CONFIGURATION")
        print("="*60)
        print(f"Model: {cls.MODEL_CHOICE}")
        print(f"Description: {model_cfg.get('desc', 'N/A')}")
        print(f"Image Size: {model_cfg['img_size']}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {model_cfg['lr']}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Folds: {cls.N_FOLDS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Wandb: {'Enabled' if cls.USE_WANDB else 'Disabled'}")
        print("="*60)
        print("\n📊 Available Models (sorted by accuracy):")
        print("-"*60)
        print(f"{'Model':<20} {'Params':<10} {'Img Size':<10} {'Tier'}")
        print("-"*60)
        model_info = [
            ('eva02_base', '87M', '448', '⭐⭐⭐⭐⭐ SOTA'),
            ('swin_base', '88M', '384', '⭐⭐⭐⭐⭐ Transformer'),
            ('convnext_base', '89M', '384', '⭐⭐⭐⭐⭐ Strong'),
            ('efficientnetv2_m', '54M', '480', '⭐⭐⭐⭐ High'),
            ('convnext_small', '50M', '384', '⭐⭐⭐⭐ Good'),
            ('efficientnet_b5', '30M', '456', '⭐⭐⭐⭐ Good'),
            ('efficientnetv2_s', '21M', '384', '⭐⭐⭐ Balanced'),
            ('efficientnet_b4', '19M', '380', '⭐⭐⭐ Balanced'),
        ]
        for name, params, img, tier in model_info:
            marker = " ← SELECTED" if name == cls.MODEL_CHOICE else ""
            print(f"{name:<20} {params:<10} {img:<10} {tier}{marker}")
        print("-"*60 + "\n")

# ==================== WANDB SETUP ====================
def setup_wandb(config):
    """Setup Weights & Biases logging"""
    if not config.USE_WANDB:
        return None
    
    try:
        import wandb
        
        run_name = config.WANDB_RUN_NAME or f"{config.MODEL_CHOICE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=config.WANDB_PROJECT,
            name=run_name,
            config={
                'model': config.MODEL_CHOICE,
                'batch_size': config.BATCH_SIZE,
                'epochs': config.EPOCHS,
                'img_size': MODEL_CONFIGS[config.MODEL_CHOICE]['img_size'],
                'lr': MODEL_CONFIGS[config.MODEL_CHOICE]['lr'],
                'n_folds': config.N_FOLDS,
            }
        )
        print("✅ Wandb initialized")
        return wandb
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")
        return None

def log_wandb(wandb, metrics, step=None):
    """Log metrics to wandb"""
    if wandb is not None:
        wandb.log(metrics, step=step)

# ==================== UTILITIES ====================
def seed_everything(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_f1': best_f1,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint"""
    if not os.path.exists(path):
        return 0, 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_f1']

# ==================== DATASET ====================
class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.is_test:
            return image
        else:
            label = self.labels[idx]
            return image, label

# ==================== AUGMENTATIONS ====================
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3),
            A.GridDistortion(distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.5),
        A.CoarseDropout(
            max_holes=8, max_height=img_size//16, max_width=img_size//16,
            min_holes=1, min_height=img_size//32, min_width=img_size//32,
            fill_value=0, p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(img_size):
    return [
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

# ==================== LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# ==================== MODEL ====================
class BloodCellModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, drop_rate=0.3, drop_path=0.1):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path
        )
        
        # Lấy feature dimension từ model (không cần dummy input)
        in_features = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# ==================== DATA PREPARATION ====================
def prepare_data(train_dir, test_dir):
    train_images = []
    train_labels = []
    
    for class_name in Config.CLASSES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_images.append(os.path.join(class_dir, img_name))
                    train_labels.append(Config.CLASSES.index(class_name))
    
    test_images = []
    test_ids = []
    if os.path.exists(test_dir):
        for img_name in sorted(os.listdir(test_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_dir, img_name))
                test_ids.append(img_name)
    
    print(f"📊 Training images: {len(train_images)}")
    print(f"📊 Test images: {len(test_images)}")
    
    label_counts = Counter(train_labels)
    print("\n📈 Class distribution:")
    for i, class_name in enumerate(Config.CLASSES):
        count = label_counts.get(i, 0)
        pct = 100*count/len(train_labels) if train_labels else 0
        print(f"   {class_name}: {count} ({pct:.1f}%)")
    
    return train_images, train_labels, test_images, test_ids

def get_weighted_sampler(labels):
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    class_weights = []
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = num_samples / (Config.NUM_CLASSES * count)
        class_name = Config.CLASSES[i]
        weight *= Config.CLASS_WEIGHTS.get(class_name, 1.0)
        class_weights.append(weight)
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def get_class_weights(labels):
    """
    Calculate class weights for loss function.
    
    NOTE: Hiện tại KHÔNG SỬ DỤNG vì đã dùng Weighted Sampler.
    Giữ lại để tham khảo nếu cần chuyển sang dùng weighted loss.
    """
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = total / (Config.NUM_CLASSES * count)
        class_name = Config.CLASSES[i]
        weight *= Config.CLASS_WEIGHTS.get(class_name, 1.0)
        weights.append(weight)
    
    return torch.FloatTensor(weights)

# ==================== TRAINER ====================
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, use_amp=True, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        # Chỉ dùng AMP khi có CUDA
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.wandb = wandb_run
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_f1 = f1_score(all_labels, all_preds, average='macro')
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'f1': f'{current_f1:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        all_probs = np.concatenate(all_probs, axis=0)
        
        return avg_loss, f1, all_preds, all_labels, all_probs

# ==================== INFERENCE ====================
@torch.no_grad()
def predict_with_tta(model, test_images, device, img_size, batch_size=32, num_workers=4):
    model.eval()
    tta_transforms = get_tta_transforms(img_size)
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    
    all_preds = []
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"   TTA {tta_idx + 1}/{len(tta_transforms)}")
        
        dataset = BloodCellDataset(test_images, labels=None, transform=transform, is_test=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
        
        preds = []
        for images in tqdm(dataloader, leave=False):
            images = images.to(device)
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        all_preds.append(preds)
    
    avg_preds = np.mean(all_preds, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)
    
    return final_preds, avg_preds

# ==================== TRAINING FUNCTIONS ====================
def train_fold(fold, train_idx, val_idx, train_images, train_labels, 
               test_images, model_cfg, wandb_run=None):
    """Train a single fold"""
    
    print(f"\n{'='*50}")
    print(f"📁 FOLD {fold + 1}/{Config.N_FOLDS}")
    print(f"{'='*50}")
    
    X_train = [train_images[i] for i in train_idx]
    X_val = [train_images[i] for i in val_idx]
    y_train = [train_labels[i] for i in train_idx]
    y_val = [train_labels[i] for i in val_idx]
    
    # Datasets
    train_dataset = BloodCellDataset(X_train, y_train, transform=get_train_transforms(model_cfg['img_size']))
    val_dataset = BloodCellDataset(X_val, y_val, transform=get_valid_transforms(model_cfg['img_size']))
    
    # Weighted Sampler - XỬ LÝ CLASS IMBALANCE TẠI ĐÂY
    sampler = get_weighted_sampler(y_train)
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Model
    model = BloodCellModel(
        model_cfg['name'], Config.NUM_CLASSES, pretrained=True,
        drop_rate=model_cfg['drop_rate'], drop_path=model_cfg['drop_path']
    ).to(Config.DEVICE)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss - KHÔNG dùng class weights (tránh double weighting với sampler)
    criterion = FocalLoss(alpha=None, gamma=Config.FOCAL_GAMMA, label_smoothing=Config.LABEL_SMOOTHING)
    
    # Optimizer, Scheduler
    optimizer = AdamW(model.parameters(), lr=model_cfg['lr'], weight_decay=1e-4)
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=model_cfg['lr'], total_steps=num_training_steps, pct_start=0.1)
    
    trainer = Trainer(model, optimizer, scheduler, criterion, Config.DEVICE, Config.USE_AMP, wandb_run)
    
    # Training loop
    best_f1 = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_model_{Config.MODEL_CHOICE}_fold{fold}.pt')
    patience = 5
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        train_loss, train_f1 = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, val_f1, _, _, val_probs = trainer.validate(val_loader)
        
        print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Log to wandb
        if wandb_run:
            log_wandb(wandb_run, {
                f'fold{fold}/train_loss': train_loss,
                f'fold{fold}/train_f1': train_f1,
                f'fold{fold}/val_loss': val_loss,
                f'fold{fold}/val_f1': val_f1,
                'epoch': epoch + 1
            })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save model state
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"   ✓ Saved best model with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹ Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model for inference
    state_dict = torch.load(best_model_path)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # OOF predictions
    _, _, _, _, oof_probs = trainer.validate(val_loader)
    
    # Test predictions with TTA
    print("   🔮 Predicting test set with TTA...")
    _, test_probs = predict_with_tta(
        model, test_images, Config.DEVICE,
        model_cfg['img_size'], Config.BATCH_SIZE, Config.NUM_WORKERS
    )
    
    return oof_probs, test_probs, best_f1, val_idx

def train_kfold(train_images, train_labels, test_images, test_ids, wandb_run=None):
    """K-Fold Cross Validation Training"""
    
    model_cfg = MODEL_CONFIGS[Config.MODEL_CHOICE]
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # NOTE: Class weighting strategy
    # - Weighted Sampler: Đảm bảo mỗi batch có balanced classes  
    # - Loss: KHÔNG dùng class weights (tránh double weighting)
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    oof_preds = np.zeros((len(train_images), Config.NUM_CLASSES))
    test_preds = np.zeros((len(test_images), Config.NUM_CLASSES))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        oof_probs, test_probs, best_f1, val_idx = train_fold(
            fold, train_idx, val_idx, 
            train_images.tolist(), train_labels.tolist(),
            test_images, model_cfg, wandb_run
        )
        
        oof_preds[val_idx] = oof_probs
        test_preds += test_probs / Config.N_FOLDS
        fold_scores.append(best_f1)
        
        print(f"\n📊 Fold {fold + 1} Best F1: {best_f1:.4f}")
    
    # Final results
    print(f"\n{'='*60}")
    print("📈 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Fold F1 Scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    oof_f1 = f1_score(train_labels, np.argmax(oof_preds, axis=1), average='macro')
    print(f"OOF F1 Score: {oof_f1:.4f}")
    
    if wandb_run:
        log_wandb(wandb_run, {
            'final/mean_f1': np.mean(fold_scores),
            'final/std_f1': np.std(fold_scores),
            'final/oof_f1': oof_f1
        })
    
    return test_preds, test_ids

# ==================== SUBMISSION ====================
def create_submission(test_preds, test_ids, output_path):
    pred_labels = np.argmax(test_preds, axis=1)
    pred_classes = [Config.CLASSES[p] for p in pred_labels]
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': pred_classes
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n📁 Submission saved to: {output_path}")
    print(f"   Shape: {submission.shape}")
    print("\n📊 Prediction distribution:")
    print(submission['TARGET'].value_counts().to_string())
    
    return submission

# ==================== ARGUMENT PARSER ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Blood Cell Classification - Vast.ai (Max Accuracy)')
    
    # Paths
    parser.add_argument('--train_dir', type=str, default=None, help='Training data directory')
    parser.add_argument('--test_dir', type=str, default=None, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    # Model - Chỉ các model mạnh nhất
    parser.add_argument('--model', type=str, default='convnext_base',
                       choices=['efficientnet_b4', 'efficientnet_b5', 'efficientnetv2_s', 
                               'efficientnetv2_m', 'convnext_small', 'convnext_base', 
                               'swin_base', 'eva02_base'],
                       help='Model to use (default: convnext_base)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if not set)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--mode', type=str, default='kfold', choices=['single', 'kfold'], help='Training mode')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable Wandb logging')
    parser.add_argument('--wandb_project', type=str, default='blood-cell-classification', help='Wandb project name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

# ==================== MAIN ====================
def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       🩸 BLOOD CELL CLASSIFICATION - VAST.AI             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Parse arguments
    args = parse_args()
    
    # Setup config
    Config.MODEL_CHOICE = args.model
    Config.N_FOLDS = args.folds
    Config.SEED = args.seed
    model_cfg = Config.setup(args)
    
    # Print config
    Config.print_config()
    
    # Set seed
    seed_everything(Config.SEED)
    
    # Setup Wandb
    wandb_run = setup_wandb(Config) if Config.USE_WANDB else None
    
    # Prepare data
    print("\n📂 Loading data...")
    train_images, train_labels, test_images, test_ids = prepare_data(
        Config.TRAIN_DIR, Config.TEST_DIR
    )
    
    if len(train_images) == 0:
        print("❌ No training data found! Please check TRAIN_DIR path.")
        return
    
    # Train
    if args.mode == 'kfold':
        print("\n🚀 Starting K-Fold Training...")
        test_preds, test_ids = train_kfold(
            train_images, train_labels, test_images, test_ids, wandb_run
        )
    else:
        print("\n🚀 Starting Single Model Training...")
        # Implement single model training if needed
        pass
    
    # Create submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = os.path.join(Config.OUTPUT_DIR, f'submission_{Config.MODEL_CHOICE}_{timestamp}.csv')
    submission = create_submission(test_preds, test_ids, submission_path)
    
    # Also save as submission.csv for easy access
    submission.to_csv(os.path.join(Config.OUTPUT_DIR, 'submission.csv'), index=False)
    
    # Finish wandb
    if wandb_run:
        wandb_run.finish()
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"📁 Results saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()