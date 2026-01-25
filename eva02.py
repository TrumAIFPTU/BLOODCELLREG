"""
Blood Cell Classification - EVA02-Base Only
============================================
Model: EVA02-Base (SOTA - State of the Art)
Params: 87M | Image Size: 448 | Accuracy: Highest

Optimized for RTX 3090 (24GB)

Author: Claude AI
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
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

# ==================== CONFIG ====================
class Config:
    # Paths (Vast.ai)
    TRAIN_DIR = "/workspace/data/train"
    TEST_DIR = "/workspace/data/test"
    OUTPUT_DIR = "/workspace/outputs"
    CHECKPOINT_DIR = "/workspace/checkpoints"
    
    # Model - EVA02-Base SOTA
    MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
    IMG_SIZE = 448
    
    # Classes
    CLASSES = ['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE']
    NUM_CLASSES = 9
    
    # Class weights (imbalance handling)
    CLASS_WEIGHTS = {
        'BA': 1.2, 'BNE': 1.0, 'EO': 0.7, 'LY': 1.2, 
        'MMY': 1.4, 'MO': 1.0, 'MY': 1.3, 'PMY': 1.8, 'SNE': 1.0
    }
    
    # Training - Optimized for EVA02
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 20  # EVA02 converges faster
    BATCH_SIZE = 12  # Safe for 24GB VRAM
    NUM_WORKERS = 4
    
    # Optimizer - Lower LR for large pretrained model
    LR = 2e-5
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.05
    
    # Loss
    LABEL_SMOOTHING = 0.1
    FOCAL_GAMMA = 2.0
    
    # Regularization
    DROP_RATE = 0.3
    DROP_PATH = 0.2
    
    # Mixed precision
    USE_AMP = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        return image, self.labels[idx]

# ==================== AUGMENTATIONS ====================
def get_train_transforms():
    """Strong augmentation for training"""
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.2),
            A.GridDistortion(distort_limit=0.2),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.Sharpen(),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        A.CoarseDropout(
            max_holes=8, max_height=Config.IMG_SIZE//16, max_width=Config.IMG_SIZE//16,
            min_holes=1, min_height=Config.IMG_SIZE//32, min_width=Config.IMG_SIZE//32,
            fill_value=0, p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_transforms():
    """Validation transforms"""
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms():
    """Test Time Augmentation - 5 transforms"""
    return [
        # Original
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # HFlip
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # VFlip
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Rotate90
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Transpose(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # HFlip + VFlip
        A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

# ==================== LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==================== MODEL ====================
class EVA02Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load EVA02-Base
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH
        )
        
        # Get feature dim - Dùng đúng IMG_SIZE của model
        in_features = self.backbone.num_features
        
        # Custom head
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        
        self._init_head()
        
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ==================== UTILITIES ====================
def prepare_data():
    """Load data"""
    train_images, train_labels = [], []
    
    for class_name in Config.CLASSES:
        class_dir = os.path.join(Config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_images.append(os.path.join(class_dir, img_name))
                    train_labels.append(Config.CLASSES.index(class_name))
    
    test_images, test_ids = [], []
    if os.path.exists(Config.TEST_DIR):
        for img_name in sorted(os.listdir(Config.TEST_DIR)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(Config.TEST_DIR, img_name))
                test_ids.append(img_name)
    
    return train_images, train_labels, test_images, test_ids

def get_class_weights(labels):
    """Calculate class weights"""
    class_counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = total / (Config.NUM_CLASSES * count)
        weight *= Config.CLASS_WEIGHTS.get(Config.CLASSES[i], 1.0)
        weights.append(weight)
    return torch.FloatTensor(weights)

def get_weighted_sampler(labels):
    """Create weighted sampler"""
    class_counts = Counter(labels)
    num_samples = len(labels)
    class_weights = []
    for i in range(Config.NUM_CLASSES):
        count = class_counts.get(i, 1)
        weight = num_samples / (Config.NUM_CLASSES * count)
        weight *= Config.CLASS_WEIGHTS.get(Config.CLASSES[i], 1.0)
        class_weights.append(weight)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ==================== TRAINING ====================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'f1': f'{current_f1:.4f}'})
    
    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro')

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    
    for images, labels in tqdm(loader, desc='Validating'):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro'), all_probs

@torch.no_grad()
def predict_tta(model, test_images):
    """Predict with TTA"""
    model.eval()
    tta_transforms = get_tta_transforms()
    all_preds = []
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"   TTA {tta_idx + 1}/{len(tta_transforms)}")
        dataset = BloodCellDataset(test_images, transform=transform, is_test=True)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        preds = []
        for images in tqdm(loader, leave=False):
            images = images.to(Config.DEVICE)
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
        
        all_preds.append(np.concatenate(preds, axis=0))
    
    return np.mean(all_preds, axis=0)

# ==================== MAIN TRAINING ====================
def train():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🏆 EVA02-Base - SOTA Blood Cell Classifier      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"📦 Model: {Config.MODEL_NAME}")
    print(f"🖼️  Image Size: {Config.IMG_SIZE}")
    print(f"📊 Batch Size: {Config.BATCH_SIZE}")
    print(f"📈 Epochs: {Config.EPOCHS}")
    print(f"🔄 Folds: {Config.N_FOLDS}")
    
    # Load data
    print("\n📂 Loading data...")
    train_images, train_labels, test_images, test_ids = prepare_data()
    print(f"   Train: {len(train_images)} images")
    print(f"   Test: {len(test_images)} images")
    
    if len(train_images) == 0:
        print("❌ No training data found!")
        return
    
    # Class distribution
    print("\n📊 Class distribution:")
    for i, name in enumerate(Config.CLASSES):
        count = train_labels.count(i)
        print(f"   {name}: {count} ({100*count/len(train_labels):.1f}%)")
    
    # Prepare
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # NOTE: Class weighting strategy
    # - Weighted Sampler: Đảm bảo mỗi batch có balanced classes
    # - Loss: KHÔNG dùng class weights (tránh double weighting)
    # → Chỉ dùng 1 trong 2, không dùng cả 2 cùng lúc
    
    # K-Fold
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    oof_preds = np.zeros((len(train_images), Config.NUM_CLASSES))
    test_preds = np.zeros((len(test_images), Config.NUM_CLASSES))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        print(f"\n{'='*60}")
        print(f"📁 FOLD {fold + 1}/{Config.N_FOLDS}")
        print(f"{'='*60}")
        
        # Data
        X_train = train_images[train_idx].tolist()
        X_val = train_images[val_idx].tolist()
        y_train = train_labels[train_idx].tolist()
        y_val = train_labels[val_idx].tolist()
        
        # Datasets
        train_dataset = BloodCellDataset(X_train, y_train, transform=get_train_transforms())
        val_dataset = BloodCellDataset(X_val, y_val, transform=get_valid_transforms())
        
        # Loaders
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE,
            sampler=get_weighted_sampler(y_train),
            num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE * 2,
            shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
        )
        
        # Model
        model = EVA02Model(pretrained=True).to(Config.DEVICE)
        
        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"   Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # Loss - KHÔNG dùng class weights vì đã có weighted sampler
        # (tránh double weighting làm F1 xấu)
        criterion = FocalLoss(alpha=None, gamma=Config.FOCAL_GAMMA, 
                             label_smoothing=Config.LABEL_SMOOTHING)
        
        # Optimizer, Scheduler
        optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        
        num_steps = len(train_loader) * Config.EPOCHS
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_steps//3, T_mult=1, eta_min=Config.MIN_LR)
        
        # GradScaler chỉ dùng khi có CUDA
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Training
        best_f1 = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f'eva02_fold{fold}.pt')
        patience = 5
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            print(f"\n   Epoch {epoch + 1}/{Config.EPOCHS}")
            
            train_loss, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler
            )
            val_loss, val_f1, val_probs = validate(model, val_loader, criterion)
            
            print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, best_model_path)
                print(f"   ✅ Saved best model with F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   ⏹️ Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        state_dict = torch.load(best_model_path)
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        # OOF predictions
        _, _, oof_probs = validate(model, val_loader, criterion)
        oof_preds[val_idx] = oof_probs
        
        # Test predictions with TTA
        print("   🔮 Predicting test set with TTA...")
        test_probs = predict_tta(model, test_images)
        test_preds += test_probs / Config.N_FOLDS
        
        fold_scores.append(best_f1)
        print(f"\n   📊 Fold {fold + 1} Best F1: {best_f1:.4f}")
        
        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Final results
    print(f"\n{'='*60}")
    print("🏆 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Fold F1 Scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    oof_f1 = f1_score(train_labels, np.argmax(oof_preds, axis=1), average='macro')
    print(f"OOF F1 Score: {oof_f1:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        train_labels, np.argmax(oof_preds, axis=1),
        target_names=Config.CLASSES
    ))
    
    # Save submission
    pred_labels = np.argmax(test_preds, axis=1)
    pred_classes = [Config.CLASSES[p] for p in pred_labels]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission = pd.DataFrame({'ID': test_ids, 'TARGET': pred_classes})
    
    # Save with timestamp
    submission_path = os.path.join(Config.OUTPUT_DIR, f'submission_eva02_{timestamp}.csv')
    submission.to_csv(submission_path, index=False)
    
    # Save as main submission
    submission.to_csv(os.path.join(Config.OUTPUT_DIR, 'submission2.csv'), index=False)
    
    print(f"\n📁 Submission saved to: {submission_path}")
    print(f"   Shape: {submission.shape}")
    print("\n📊 Prediction distribution:")
    print(submission['TARGET'].value_counts().to_string())
    
    print("\n✅ TRAINING COMPLETE!")
    
    return submission

if __name__ == "__main__":
    train()