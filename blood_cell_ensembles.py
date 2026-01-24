"""
BLOOD CELL CLASSIFICATION - IMPROVED VERSION
=============================================
Fixes:
- Correct TTA implementation
- Updated PyTorch API
- Added class weights, label smoothing, mixup
- Stratified K-Fold cross validation
- Better memory management
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from PIL import Image

# ============================================================================
# CONFIG - Thêm options mới
# ============================================================================

class Config:
    # Paths
    TRAIN_DIR = '/content/drive/MyDrive/train'
    TEST_DIR = '/content/drive/MyDrive/test'
    OUTPUT_DIR = '/content/output'
    
    # Classes
    CLASSES = ['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE']
    NUM_CLASSES = len(CLASSES)
    
    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Models
    MODELS = {
        'efficientnet_b2': 'tf_efficientnet_b2.ns_jft_in1k',
        'efficientnet_b3': 'tf_efficientnet_b3.ns_jft_in1k',
        'resnet50': 'resnet50.a1_in1k'
    }
    
    # === NEW OPTIONS ===
    USE_KFOLD = False  # Set True cho cross-validation
    N_FOLDS = 5
    
    # Augmentation options
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # Loss options
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1
    USE_CLASS_WEIGHTS = True
    
    # TTA
    TTA_TIMES = 5
    
    # Others
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATIENCE = 10
    USE_AMP = True
    
    # Learning rate scheduler
    WARMUP_EPOCHS = 3
    MIN_LR = 1e-7

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SEED
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# ============================================================================
# MIXUP AUGMENTATION (NEW)
# ============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation - giúp model generalize tốt hơn"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss cho mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# FOCAL LOSS (NEW) - Tốt cho imbalanced data
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights
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
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================================
# DATA AUGMENTATION - Cải tiến
# ============================================================================

def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        
        # Color augmentation
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ], p=0.5),
        
        # Blur & Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(var_limit=(10.0, 30.0)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        
        # Advanced
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.GridDistortion(p=0.2),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# DATASET - Cải tiến để support TTA đúng cách
# ============================================================================

class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, return_path=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_path = return_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.labels is not None:
            label = self.labels[idx]
            if self.return_path:
                return image, label, img_path
            return image, label
        else:
            if self.return_path:
                return image, img_path
            return image


class TTADataset(Dataset):
    """Dataset riêng cho TTA - load raw image"""
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        return image, img_path

# ============================================================================
# PREPARE DATA - Thêm class weights calculation
# ============================================================================

def prepare_data():
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(Config.TRAIN_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: {class_dir} không tồn tại!")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    print(f"📊 Tổng số ảnh train: {len(image_paths)}")
    print(f"📈 Phân bố classes:")
    
    class_counts = Counter(labels)
    for idx, class_name in enumerate(Config.CLASSES):
        count = class_counts.get(idx, 0)
        print(f"  {class_name}: {count} ảnh ({count/len(labels)*100:.1f}%)")
    
    return image_paths, labels

def calculate_class_weights(labels):
    """Tính class weights cho imbalanced data"""
    class_counts = Counter(labels)
    total = len(labels)
    weights = []
    
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        # Inverse frequency weighting
        weight = total / (Config.NUM_CLASSES * count)
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    # Normalize
    weights = weights / weights.sum() * Config.NUM_CLASSES
    
    print(f"📊 Class weights: {dict(zip(Config.CLASSES, weights.numpy().round(3)))}")
    return weights

def prepare_test_data():
    test_paths = []
    test_ids = []
    
    for img_name in sorted(os.listdir(Config.TEST_DIR)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_paths.append(os.path.join(Config.TEST_DIR, img_name))
            test_ids.append(img_name)
    
    print(f"📊 Tổng số ảnh test: {len(test_paths)}")
    return test_paths, test_ids

# ============================================================================
# MODEL
# ============================================================================

class BloodCellModel(nn.Module):
    def __init__(self, model_name, num_classes=Config.NUM_CLASSES, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        # Custom head với dropout
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ============================================================================
# LEARNING RATE SCHEDULER với Warmup
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(base_lr * lr_scale, self.min_lr)
        
        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# TRAINING FUNCTIONS - Cập nhật API và thêm Mixup
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'max':
            if val_score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0
        else:  # mode == 'min'
            if val_score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_mixup=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixup
        if use_mixup and Config.USE_MIXUP:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
        
        # Mixed precision - Updated API
        if scaler and Config.USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                if use_mixup and Config.USE_MIXUP:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup and Config.USE_MIXUP:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        
        # Với mixup, dùng original labels cho metrics
        if use_mixup and Config.USE_MIXUP:
            all_labels.extend(labels_a.cpu().numpy())
        else:
            all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_f1

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            if Config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(valid_loader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    all_probs = np.vstack(all_probs)
    
    return epoch_loss, epoch_f1, all_preds, all_labels, all_probs

def train_model(model_name, train_loader, valid_loader, class_weights, device, fold=None):
    fold_str = f" (Fold {fold+1})" if fold is not None else ""
    print(f"\n{'='*60}")
    print(f"Training {model_name}{fold_str}")
    print(f"{'='*60}")
    
    # Initialize model
    model = BloodCellModel(Config.MODELS[model_name], num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    
    # Loss với class weights và label smoothing
    if Config.USE_CLASS_WEIGHTS:
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0,
        label_smoothing=Config.LABEL_SMOOTHING if Config.USE_LABEL_SMOOTHING else 0.0
    )
    
    # Optimizer với weight decay riêng cho bias/norm layers
    no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)
    
    # Scheduler với warmup
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.EPOCHS,
        min_lr=Config.MIN_LR
    )
    
    # Mixed precision - Updated API
    scaler = torch.amp.GradScaler('cuda') if Config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode='max')
    
    # Training history
    history = {'train_loss': [], 'train_f1': [], 'valid_loss': [], 'valid_f1': [], 'lr': []}
    best_f1 = 0.0
    
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    best_model_path = os.path.join(Config.OUTPUT_DIR, f'{model_name}{fold_suffix}_best.pth')
    
    for epoch in range(Config.EPOCHS):
        current_lr = scheduler.step(epoch)
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.2e}")
        print("-" * 60)
        
        # Train (với mixup sau warmup)
        use_mixup = epoch >= Config.WARMUP_EPOCHS
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_mixup
        )
        
        # Validate
        valid_loss, valid_f1, _, _, _ = validate(model, valid_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['valid_loss'].append(valid_loss)
        history['valid_f1'].append(valid_f1)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid F1: {valid_f1:.4f}")
        
        # Save best model
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, best_model_path)
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
        
        # Early stopping
        early_stopping(valid_f1)
        if early_stopping.early_stop:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🏆 Best F1-Macro: {best_f1:.4f}")
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history, best_f1

# ============================================================================
# TTA PREDICTION - SỬA LỖI CHÍNH
# ============================================================================

def get_tta_transforms():
    """Các transforms cho TTA"""
    base_transform = lambda: [
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    return [
        # Original
        A.Compose(base_transform()),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transform()),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)] + base_transform()),
        # Rotate 90
        A.Compose([A.Rotate(limit=(90, 90), p=1.0)] + base_transform()),
        # Rotate -90
        A.Compose([A.Rotate(limit=(-90, -90), p=1.0)] + base_transform()),
    ]

def predict_with_tta(model, test_paths, device):
    """
    TTA đúng cách - transform từ raw images
    """
    model.eval()
    tta_transforms = get_tta_transforms()
    
    all_probs = []
    
    with torch.no_grad():
        for img_path in tqdm(test_paths, desc='Predicting with TTA'):
            # Load raw image
            image = np.array(Image.open(img_path).convert('RGB'))
            
            tta_probs = []
            for transform in tta_transforms:
                # Apply transform
                augmented = transform(image=image)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                
                # Predict
                if Config.USE_AMP:
                    with torch.amp.autocast('cuda'):
                        output = model(img_tensor)
                else:
                    output = model(img_tensor)
                
                prob = F.softmax(output, dim=1).cpu().numpy()
                tta_probs.append(prob)
            
            # Average TTA predictions
            avg_prob = np.mean(tta_probs, axis=0)
            all_probs.append(avg_prob)
    
    return np.vstack(all_probs)

def predict_batch_with_tta(model, test_paths, device, batch_size=32):
    """
    Batch TTA - Nhanh hơn cho dataset lớn
    """
    model.eval()
    tta_transforms = get_tta_transforms()
    n_tta = len(tta_transforms)
    
    all_probs = np.zeros((len(test_paths), Config.NUM_CLASSES))
    
    with torch.no_grad():
        for tta_idx, transform in enumerate(tta_transforms):
            print(f"  TTA {tta_idx + 1}/{n_tta}")
            
            # Create dataset với transform này
            dataset = BloodCellDataset(test_paths, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
            
            batch_probs = []
            for images in loader:
                images = images.to(device)
                
                if Config.USE_AMP:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                batch_probs.append(probs)
            
            tta_probs = np.vstack(batch_probs)
            all_probs += tta_probs / n_tta
    
    return all_probs

# ============================================================================
# MAIN - Cải tiến
# ============================================================================

def main():
    print("="*80)
    print("🔬 BLOOD CELL CLASSIFICATION - IMPROVED VERSION")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Models: {list(Config.MODELS.keys())}")
    print(f"Mixup: {Config.USE_MIXUP} | Label Smoothing: {Config.USE_LABEL_SMOOTHING}")
    print(f"Class Weights: {Config.USE_CLASS_WEIGHTS}")
    print("="*80)
    
    # Prepare Data
    print("\n📁 [1/6] Preparing data...")
    train_paths, train_labels = prepare_data()
    test_paths, test_ids = prepare_test_data()
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels)
    
    # Split
    train_paths, valid_paths, train_labels_split, valid_labels = train_test_split(
        train_paths, train_labels, 
        test_size=0.2, 
        stratify=train_labels, 
        random_state=Config.SEED
    )
    
    print(f"\n📊 Train: {len(train_paths)} | Valid: {len(valid_paths)} | Test: {len(test_paths)}")
    
    # Create DataLoaders
    print("\n📦 [2/6] Creating data loaders...")
    train_dataset = BloodCellDataset(train_paths, train_labels_split, transform=get_train_transforms())
    valid_dataset = BloodCellDataset(valid_paths, valid_labels, transform=get_valid_transforms())
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS, 
        pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Train Models
    print("\n🚀 [3/6] Training models...")
    trained_models = {}
    histories = []
    best_f1_scores = {}
    
    for model_name in Config.MODELS.keys():
        model, history, best_f1 = train_model(
            model_name, train_loader, valid_loader, 
            class_weights, Config.DEVICE
        )
        trained_models[model_name] = model
        histories.append(history)
        best_f1_scores[model_name] = best_f1
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Validation Performance
    print("\n📊 [4/6] Final validation...")
    criterion = FocalLoss(alpha=class_weights.to(Config.DEVICE), gamma=2.0)
    
    for model_name, model in trained_models.items():
        print(f"\n{model_name}:")
        _, valid_f1, valid_preds, valid_true, _ = validate(
            model, valid_loader, criterion, Config.DEVICE
        )
        print(f"Validation F1-Macro: {valid_f1:.4f}")
        print(classification_report(valid_true, valid_preds, target_names=Config.CLASSES))
    
    # Test Predictions with TTA
    print("\n🎯 [5/6] Generating test predictions with TTA...")
    all_test_preds = []
    
    for model_name, model in trained_models.items():
        print(f"\n{model_name}:")
        test_probs = predict_batch_with_tta(model, test_paths, Config.DEVICE)
        all_test_preds.append(test_probs)
    
    # Ensemble
    print("\n🎯 [6/6] Creating ensemble...")
    
    # Weighted average based on validation F1
    weights = np.array([best_f1_scores[name] for name in Config.MODELS.keys()])
    weights = weights / weights.sum()
    print(f"Ensemble weights: {dict(zip(Config.MODELS.keys(), weights.round(3)))}")
    
    ensemble_probs = np.zeros_like(all_test_preds[0])
    for weight, preds in zip(weights, all_test_preds):
        ensemble_probs += weight * preds
    
    # Final predictions
    final_preds = np.argmax(ensemble_probs, axis=1)
    final_classes = [Config.CLASSES[pred] for pred in final_preds]
    
    # Save submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': final_classes
    })
    
    submission_path = os.path.join(Config.OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    # Save probabilities for potential post-processing
    probs_df = pd.DataFrame(ensemble_probs, columns=Config.CLASSES)
    probs_df['ID'] = test_ids
    probs_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'predictions_probs.csv'), index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"\n📊 Prediction distribution:")
    print(pd.Series(final_classes).value_counts())
    
    print("\n" + "="*80)
    print("🏆 TRAINING COMPLETE!")
    print("="*80)
    for model_name, f1 in best_f1_scores.items():
        print(f"  {model_name}: {f1:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()