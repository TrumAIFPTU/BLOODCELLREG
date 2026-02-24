"""
Fast Track - Blood Cell Classification
- Mô hình: EVA02 Base (chỉ 1 mô hình duy nhất)
- Dữ liệu: Train_Combined (có chứa Pseudo-labels)
- Chế độ: 1 Fold (để tiết kiệm tối đa thời gian thuê VGA)
- Resume: False (Học lại từ đầu để quen với Domain đùn ép 10 epochs)
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== CONFIG ====================
class Config:
    TRAIN_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train_combined"
    TEST_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\test1"
    OUTPUT_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\outputs"
    CHECKPOINT_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\checkpoints"
    
    CLASSES = sorted(['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE'])
    NUM_CLASSES = 9
    
    # Chỉ dùng 1 mô hình duy nhất nhưng TĂNG BATCH_SIZE để vét sạch 24GB VRAM
    MODEL_NAME = 'eva02_base'
    TIMM_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
    IMG_SIZE = 448
    BATCH_SIZE = 24  # Nếu OOM thì giảm xuống 16
    
    EPOCHS = 10     # Lọc bỏ nhiễu nhanh gọn
    LEARNING_RATE = 5e-6
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.01
    
    N_FOLDS = 5     # Vẫn chia 5 Folds để Sklearn Stratified hoạt động mượt mà
    RUN_FOLDS = 1   # NHƯNG CHỈ CHẠY DUY NHẤT FOLD ĐẦU TIÊN RỒI DỪNG LẠI!
    
    SEED = 42
    NUM_WORKERS = 4
    USE_AMP = True
    TTA_TRANSFORMS = 4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==================== DATA & TRANFORMS (GIỮ NGUYÊN BỘ AUG GẤU) ====================
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
            return image, self.labels[idx]
        return image

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
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(img_size):
    return [
        A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(90, 90), p=1.0, border_mode=0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    ]

# ==================== ARCHITECTURE ====================
class BloodCellModel(nn.Module):
    def __init__(self, timm_name, num_classes=9):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# ==================== PIPELINE ====================
def train_fast_track():
    print("="*60)
    print("FAST TRACK (Pseudo-Label Test): EVO02 - 1 FOLD - NO RESUME")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Batch Size: {Config.BATCH_SIZE} | Epochs: {Config.EPOCHS}\n")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Load Data
    train_images, train_labels = [], []
    for idx, cls in enumerate(Config.CLASSES):
        cls_dir = os.path.join(Config.TRAIN_DIR, cls)
        if not os.path.exists(cls_dir): continue
        for img in os.listdir(cls_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(os.path.join(cls_dir, img))
                train_labels.append(idx)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(f"Loaded {len(train_images)} mixed training images.")

    # 2. Setup K-Fold (But only run 1)
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        if fold >= Config.RUN_FOLDS: 
            break
            
        print(f"\n--- Running the ONLY FOLD (0) ---")
        
        X_train, y_train = train_images[train_idx], train_labels[train_idx]
        X_val, y_val = train_images[val_idx], train_labels[val_idx]
        
        train_ds = BloodCellDataset(X_train, y_train, get_transforms(Config.IMG_SIZE, True))
        val_ds = BloodCellDataset(X_val, y_val, get_transforms(Config.IMG_SIZE, False))
        
        # Weighted Sampler fixes class imbalance
        counts = Counter(y_train)
        weights = [1.0 / counts[lbl] for lbl in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), True)
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=Config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE*2, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        # 3. Model & Optim
        model = BloodCellModel(Config.TIMM_NAME, Config.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)
        scaler = GradScaler() if Config.USE_AMP else None
        
        best_f1 = 0
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_fast_fold{fold}.pt")
        
        # 4. Training Loop
        for epoch in range(Config.EPOCHS):
            # TRAIN
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=Config.USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            # VAL
            model.eval()
            val_loss = 0
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for imgs, lbls in tqdm(val_loader, desc="[Valid]"):
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    with autocast(enabled=Config.USE_AMP):
                        outputs = model(imgs)
                        loss = criterion(outputs, lbls)
                    val_loss += loss.item()
                    all_preds.extend(outputs.argmax(1).cpu().numpy())
                    all_lbls.extend(lbls.cpu().numpy())
            
            val_f1 = f1_score(all_lbls, all_preds, average='macro')
            scheduler.step()
            print(f" -> T_Loss: {train_loss/len(train_loader):.4f} | V_Loss: {val_loss/len(val_loader):.4f} | V_F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(" -> (Saved Best Model!)")
                
        # 5. Predict on Test Set
        print("\n" + "="*40 + "\n GENERATING FAST SUBMISSION...\n" + "="*40)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        test_images = sorted([os.path.join(Config.TEST_DIR, f) for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg','.png'))])
        tta_transforms = get_tta_transforms(Config.IMG_SIZE)
        
        all_probs = []
        for transform in tta_transforms:
            test_ds = BloodCellDataset(test_images, transform=transform)
            test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE*2, num_workers=Config.NUM_WORKERS, pin_memory=True)
            
            probs = []
            with torch.no_grad():
                for imgs in tqdm(test_loader, desc="TTA Inference"):
                    imgs = imgs.to(device)
                    with autocast(enabled=Config.USE_AMP):
                        outputs = model(imgs)
                    probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            all_probs.append(np.concatenate(probs, axis=0))
            
        final_preds = np.mean(all_probs, axis=0).argmax(axis=1)
        sub = pd.DataFrame({
            'ID': [os.path.basename(p) for p in test_images],
            'TARGET': [Config.CLASSES[i] for i in final_preds]
        })
        sub_path = os.path.join(Config.OUTPUT_DIR, 'fast_submission.csv')
        sub.to_csv(sub_path, index=False)
        print(f"Done! Evaluated F1: {best_f1:.4f}. Submission saved at: {sub_path}")

if __name__ == "__main__":
    seed_everything(Config.SEED)
    train_fast_track()
