"""
Blood Cell Classification - Inspired by Rabia Asghar et al. (2023)
===================================================================
Paper: "Classification of Blood Cells Using Deep Learning Models"
Link: https://www.researchgate.net/publication/373117082

Key Ideas Applied:
1. Ensemble multiple pre-trained models (EVA02, ConvNeXt, Swin)
2. Sparse Categorical Cross-Entropy (thay Focal Loss)
3. Longer training (more epochs)
4. Majority Voting for final prediction
5. Full fine-tuning (không freeze layers)

Dataset: 9-class blood cell classification (imbalanced)
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== CONFIG ====================
class Config:
    # Paths
    TRAIN_DIR = "/workspace/data/train"
    TEST_DIR = "/workspace/data/test1"
    OUTPUT_DIR = "/workspace/outputs"
    CHECKPOINT_DIR = "/workspace/checkpoints"
    
    # Classes - SORTED alphabetically (quan trọng!)
    CLASSES = sorted(['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE'])
    NUM_CLASSES = 9
    
    # Ensemble Models - inspired by paper comparing multiple architectures
    ENSEMBLE_MODELS = [
        {
            'name': 'eva02_base',
            'timm_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            'img_size': 448,
            'batch_size': 12,
        },
        {
            'name': 'convnext_base',
            'timm_name': 'convnext_base.fb_in22k_ft_in1k_384',
            'img_size': 384,
            'batch_size': 16,
        },
        {
            'name': 'swin_base',
            'timm_name': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
            'img_size': 384,
            'batch_size': 16,
        },
    ]
    
    # Training - fine-tuning with CLAHE
    EPOCHS = 15
    LEARNING_RATE = 5e-6
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.01
    
    # Cross-validation
    N_FOLDS = 5
    
    # Resume Control
    RESUME_TRAINING = True
    RESUME_EPOCH = 0 # Default starting epoch if resuming
    
    # Other
    SEED = 42
    NUM_WORKERS = 4
    USE_AMP = True
    
    # TTA
    TTA_TRANSFORMS = 4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ưu tiên speed cho training (benchmark=True)
    # Nếu cần reproducibility hoàn toàn: đổi benchmark=False, deterministic=True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==================== DATASET ====================
class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image

# ==================== TRANSFORMS ====================
def get_transforms(img_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(50.0, 400.0), p=0.8),
                A.ImageCompression(quality_lower=50, quality_upper=80, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def get_tta_transforms(img_size):
    """TTA transforms - deterministic rotations"""
    return [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # HFlip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # VFlip
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Rotate 90 degrees (deterministic)
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=(90, 90), p=1.0, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

# ==================== MODEL ====================
class BloodCellModel(nn.Module):
    """
    Simple model inspired by paper:
    - Pre-trained backbone as feature extractor
    - Custom classification head
    """
    def __init__(self, model_name, num_classes=9, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        # Classification head - inspired by paper's simple approach
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # Faster + less VRAM
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(dataloader), f1, all_preds, all_labels

def predict_tta(model, image_paths, img_size, device, use_amp, batch_size=32):
    """TTA prediction"""
    model.eval()
    tta_transforms = get_tta_transforms(img_size)
    
    all_probs = []
    
    for transform in tta_transforms:
        dataset = BloodCellDataset(image_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=Config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        
        probs = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc='Predicting', leave=False):
                images = images.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        
        all_probs.append(np.concatenate(probs, axis=0))
    
    # Average TTA predictions
    return np.mean(all_probs, axis=0)

# ==================== DATA PREPARATION ====================
def prepare_data():
    """Load data with SORTED class order (critical!)"""
    train_images, train_labels = [], []
    
    print(f"Classes order: {Config.CLASSES}")
    
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(Config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            images = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for img_name in images:
                train_images.append(os.path.join(class_dir, img_name))
                train_labels.append(class_idx)
            print(f"  {class_name} (idx={class_idx}): {len(images)} images")
    
    print(f"\nTotal training images: {len(train_images)}")
    return np.array(train_images), np.array(train_labels)

def prepare_test_data():
    """Load test data"""
    test_images = sorted([
        os.path.join(Config.TEST_DIR, f) 
        for f in os.listdir(Config.TEST_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Total test images: {len(test_images)}")
    return test_images

# ==================== MAIN TRAINING ====================
def train_single_model(model_config, train_images, train_labels, device):
    """Train a single model with K-Fold CV"""
    
    model_name = model_config['name']
    timm_name = model_config['timm_name']
    img_size = model_config['img_size']
    batch_size = model_config['batch_size']
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Timm: {timm_name}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Setup
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    use_pin_memory = torch.cuda.is_available()
    
    # K-Fold
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    oof_preds = np.zeros((len(train_images), Config.NUM_CLASSES))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        
        # Data
        X_train, X_val = train_images[train_idx], train_images[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        
        train_transform = get_transforms(img_size, is_train=True)
        val_transform = get_transforms(img_size, is_train=False)
        
        train_dataset = BloodCellDataset(X_train, y_train, train_transform)
        val_dataset = BloodCellDataset(X_val, y_val, val_transform)
        
        # Weighted Sampler for imbalanced data
        class_counts = Counter(y_train)
        weights = [1.0 / class_counts[label] for label in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True  # replacement=True quan trọng cho imbalanced data
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=Config.NUM_WORKERS, pin_memory=use_pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                               num_workers=Config.NUM_WORKERS, pin_memory=use_pin_memory)
        
        # Model
        model = BloodCellModel(timm_name, Config.NUM_CLASSES).to(device)
        
        # Load checkpoint if resuming
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}.pt")
        if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
            print(f"  --> Resuming training from existing checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
        
        # Loss - Sparse Categorical Cross-Entropy (inspired by paper)
        # Không dùng class weights trong loss vì đã có weighted sampler
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer - Adam (inspired by paper)
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        
        # Scheduler - Cosine Annealing (per epoch)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)
        
        # GradScaler
        scaler = GradScaler() if use_amp else None
        
        # Training loop
        best_f1 = 0
        best_epoch = Config.RESUME_EPOCH
        patience = 10
        patience_counter = 0
        
        for epoch in range(Config.RESUME_EPOCH, Config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, 
                                         device, use_amp, scaler)
            val_loss, val_f1, val_preds, val_labels = validate(model, val_loader, criterion, device, use_amp)
            
            # Step scheduler per epoch
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\nFold {fold + 1} Best F1: {best_f1:.4f} (Epoch {best_epoch})")
        fold_scores.append(best_f1)
        
        # Load best model and get OOF predictions
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("  ⚠️ No checkpoint found, using last epoch weights")
        oof_preds[val_idx] = predict_tta(model, X_val, img_size, device, use_amp, batch_size * 3)
        
        # Clear memory
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    
    # OOF Score
    oof_pred_labels = oof_preds.argmax(axis=1)
    oof_f1 = f1_score(train_labels, oof_pred_labels, average='macro')
    
    print(f"\n{model_name} Results:")
    print(f"  Fold F1 Scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  OOF F1: {oof_f1:.4f}")
    
    return oof_preds, fold_scores, oof_f1

def predict_single_model(model_config, test_images, device):
    """Predict with a single model (ensemble of folds)"""
    
    model_name = model_config['name']
    timm_name = model_config['timm_name']
    img_size = model_config['img_size']
    batch_size = model_config['batch_size']
    
    use_amp = Config.USE_AMP and torch.cuda.is_available()
    
    all_preds = []
    
    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}.pt")
        
        model = BloodCellModel(timm_name, Config.NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        
        preds = predict_tta(model, test_images, img_size, device, use_amp, batch_size * 3)
        all_preds.append(preds)
        
        del model
        torch.cuda.empty_cache()
    
    # Average predictions from all folds
    return np.mean(all_preds, axis=0)

def ensemble_predict(all_model_preds, method='average'):
    """
    Ensemble predictions from multiple models
    
    Methods:
    - 'average': Average probabilities (soft voting)
    - 'voting': Majority voting (hard voting)
    """
    if method == 'average':
        # Soft voting - average probabilities
        return np.mean(all_model_preds, axis=0)
    
    elif method == 'voting':
        # Hard voting - majority vote on predictions
        all_labels = [preds.argmax(axis=1) for preds in all_model_preds]
        final_labels = []
        
        for i in range(len(all_labels[0])):
            votes = [labels[i] for labels in all_labels]
            most_common = Counter(votes).most_common(1)[0][0]
            final_labels.append(most_common)
        
        return np.array(final_labels)
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ==================== MAIN ====================
def main():
    print("="*60)
    print("Blood Cell Classification - Ensemble Approach")
    print("Inspired by Rabia Asghar et al. (2023)")
    print("="*60)
    
    # Setup
    seed_everything(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    print("\n" + "="*40)
    print("Loading Data...")
    print("="*40)
    train_images, train_labels = prepare_data()
    test_images = prepare_test_data()
    
    # Train all models
    all_oof_preds = []
    all_test_preds = []
    model_results = []
    
    for model_config in Config.ENSEMBLE_MODELS:
        # Train
        oof_preds, fold_scores, oof_f1 = train_single_model(
            model_config, train_images, train_labels, device
        )
        all_oof_preds.append(oof_preds)
        model_results.append({
            'name': model_config['name'],
            'fold_scores': fold_scores,
            'oof_f1': oof_f1
        })
        
        # Predict on test
        print(f"\nPredicting test set with {model_config['name']}...")
        test_preds = predict_single_model(model_config, test_images, device)
        all_test_preds.append(test_preds)
    
    # Ensemble predictions
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    
    # Method 1: Average (Soft Voting)
    ensemble_oof = ensemble_predict(all_oof_preds, method='average')
    ensemble_oof_labels = ensemble_oof.argmax(axis=1)
    ensemble_oof_f1 = f1_score(train_labels, ensemble_oof_labels, average='macro')
    
    ensemble_test = ensemble_predict(all_test_preds, method='average')
    ensemble_test_labels = ensemble_test.argmax(axis=1)
    
    print(f"\nEnsemble OOF F1 (Soft Voting): {ensemble_oof_f1:.4f}")
    
    # Method 2: Majority Voting (Hard Voting) - Fixed
    all_oof_labels = [p.argmax(axis=1) for p in all_oof_preds]
    voting_oof = []
    for i in range(len(train_labels)):
        votes = [labels[i] for labels in all_oof_labels]
        most_common = Counter(votes).most_common(1)[0][0]
        voting_oof.append(most_common)
    voting_oof = np.array(voting_oof)
    voting_oof_f1 = f1_score(train_labels, voting_oof, average='macro')
    
    print(f"Ensemble OOF F1 (Hard Voting): {voting_oof_f1:.4f}")
    
    # Print individual model results
    print("\nIndividual Model Results:")
    for result in model_results:
        print(f"  {result['name']}: OOF F1 = {result['oof_f1']:.4f}")
    
    # Save submission
    print("\n" + "="*40)
    print("Creating Submission...")
    print("="*40)
    
    submission = pd.DataFrame({
        'ID': [os.path.basename(p) for p in test_images],
        'TARGET': [Config.CLASSES[i] for i in ensemble_test_labels]
    })
    
    submission_path = os.path.join(Config.OUTPUT_DIR, 'submission_ensemble.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")
    
    # Check prediction distribution
    print("\nPrediction Distribution:")
    pred_counts = Counter(ensemble_test_labels)
    for idx, count in sorted(pred_counts.items()):
        print(f"  {Config.CLASSES[idx]}: {count} ({count/len(ensemble_test_labels)*100:.1f}%)")
    
    # Save probabilities for potential ensemble with other models
    np.save(os.path.join(Config.OUTPUT_DIR, 'ensemble_test_probs.npy'), ensemble_test)
    np.save(os.path.join(Config.OUTPUT_DIR, 'ensemble_oof_probs.npy'), ensemble_oof)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Ensemble OOF F1: {max(ensemble_oof_f1, voting_oof_f1):.4f}")

if __name__ == "__main__":
    main()