"""
Unsupervised Analysis: Train vs Test Distribution
==================================================
Mục đích: Debug tại sao OOF F1 = 0.91 nhưng LB F1 = 0.56

Approaches:
1. Extract features từ pre-trained model
2. Clustering trên Test set
3. t-SNE visualization Train + Test
4. So sánh distribution

Author: Debug script for blood cell classification
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import seaborn as sns

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== CONFIG ====================
class Config:
    TRAIN_DIR = "/workspace/data/train"
    TEST_DIR = "/workspace/data/test1"
    OUTPUT_DIR = "/workspace/outputs"
    
    CLASSES = sorted(['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE'])
    NUM_CLASSES = 9
    
    # Model for feature extraction
    MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
    IMG_SIZE = 448
    BATCH_SIZE = 16
    
    SEED = 42

# ==================== DATASET ====================
class SimpleDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

# ==================== FEATURE EXTRACTION ====================
def extract_features(model, dataloader, device):
    """Extract features from model backbone using forward_features"""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            # Dùng forward_features để đảm bảo lấy backbone output
            # không phụ thuộc vào num_classes config
            features = model.forward_features(images)
            
            # Handle different output shapes (some models return [B, H, W, C] or [B, N, C])
            if features.dim() == 4:  # [B, C, H, W]
                features = features.mean(dim=[2, 3])  # Global average pooling
            elif features.dim() == 3:  # [B, N, C] - ViT style
                features = features[:, 0]  # Take CLS token or mean
            
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)

# ==================== VISUALIZATION ====================
def plot_tsne(train_features, test_features, train_labels, 
              test_clusters=None, save_path=None):
    """
    Plot t-SNE visualization of Train + Test
    PCA fit on train only to avoid information leakage
    """
    print("Computing t-SNE...")
    
    n_train = len(train_features)
    n_test = len(test_features)
    
    # Reduce dimensionality first with PCA (faster t-SNE)
    # FIT ON TRAIN ONLY to avoid leaking test distribution
    if train_features.shape[1] > 50:
        print("  Applying PCA (fit on train only)...")
        pca = PCA(n_components=50, random_state=Config.SEED)
        train_pca = pca.fit_transform(train_features)  # Fit on train
        test_pca = pca.transform(test_features)         # Transform test
        all_features = np.vstack([train_pca, test_pca])
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        all_features = np.vstack([train_features, test_features])
    
    # Adaptive perplexity based on sample size
    min_samples = min(n_train, n_test)
    perplexity = min(30, max(5, min_samples // 10))
    print(f"  Using perplexity={perplexity} (adaptive)")
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=Config.SEED, 
                n_iter=1000, verbose=1)
    embeddings = tsne.fit_transform(all_features)
    
    train_emb = embeddings[:n_train]
    test_emb = embeddings[n_train:]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Train only (colored by true label)
    ax1 = axes[0]
    scatter1 = ax1.scatter(train_emb[:, 0], train_emb[:, 1], 
                          c=train_labels, cmap='tab10', alpha=0.6, s=10)
    ax1.set_title('Train Set (True Labels)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(range(Config.NUM_CLASSES))
    cbar1.set_ticklabels(Config.CLASSES)
    
    # Plot 2: Test only (colored by cluster if available)
    ax2 = axes[1]
    if test_clusters is not None:
        scatter2 = ax2.scatter(test_emb[:, 0], test_emb[:, 1], 
                              c=test_clusters, cmap='tab10', alpha=0.6, s=10)
        ax2.set_title('Test Set (Clusters)', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
    else:
        ax2.scatter(test_emb[:, 0], test_emb[:, 1], c='gray', alpha=0.6, s=10)
        ax2.set_title('Test Set', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # Plot 3: Train + Test overlay
    ax3 = axes[2]
    ax3.scatter(train_emb[:, 0], train_emb[:, 1], c='blue', alpha=0.3, s=10, label='Train')
    ax3.scatter(test_emb[:, 0], test_emb[:, 1], c='red', alpha=0.3, s=10, label='Test')
    ax3.set_title('Train vs Test Overlay', fontsize=14, fontweight='bold')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()
    
    return train_emb, test_emb

def plot_cluster_comparison(train_labels, test_clusters, test_predictions=None, save_path=None):
    """
    So sánh distribution của Train labels, Test clusters, và Test predictions
    """
    fig, axes = plt.subplots(1, 3 if test_predictions is not None else 2, figsize=(15, 5))
    
    # Train label distribution
    train_counts = Counter(train_labels)
    ax1 = axes[0]
    classes = Config.CLASSES
    train_vals = [train_counts.get(i, 0) for i in range(len(classes))]
    bars1 = ax1.bar(classes, train_vals, color='steelblue')
    ax1.set_title('Train Distribution (True Labels)', fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, val in zip(bars1, train_vals):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Test cluster distribution
    test_counts = Counter(test_clusters)
    ax2 = axes[1]
    test_vals = [test_counts.get(i, 0) for i in range(len(classes))]
    bars2 = ax2.bar(classes, test_vals, color='coral')
    ax2.set_title('Test Distribution (Clusters)', fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, val in zip(bars2, test_vals):
        ax2.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # Test predictions distribution (if available)
    if test_predictions is not None:
        pred_counts = Counter(test_predictions)
        ax3 = axes[2]
        pred_vals = [pred_counts.get(i, 0) for i in range(len(classes))]
        bars3 = ax3.bar(classes, pred_vals, color='green')
        ax3.set_title('Test Distribution (Model Predictions)', fontweight='bold')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Count')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        for bar, val in zip(bars3, pred_vals):
            ax3.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()

def analyze_cluster_purity(test_clusters, test_predictions):
    """
    Analyze xem clusters và predictions có match không
    """
    print("\n" + "="*50)
    print("Cluster vs Prediction Analysis")
    print("="*50)
    
    # Contingency table
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_clusters, test_predictions)
    
    # For each cluster, find dominant prediction
    print("\nCluster → Dominant Prediction:")
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            dominant = cm[i].argmax()
            purity = cm[i, dominant] / cm[i].sum()
            print(f"  Cluster {i}: Predicted mostly as {Config.CLASSES[dominant]} ({purity:.1%})")
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(test_clusters, test_predictions)
    nmi = normalized_mutual_info_score(test_clusters, test_predictions)
    
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}")
    
    if ari > 0.8:
        print("→ Clusters và Predictions rất match! Model có vẻ đúng.")
    elif ari > 0.5:
        print("→ Clusters và Predictions match trung bình.")
    else:
        print("→ Clusters và Predictions KHÔNG match! Có vấn đề với model.")
    
    return ari, nmi

# ==================== MAIN ====================
def main():
    print("="*60)
    print("Unsupervised Analysis: Train vs Test Distribution")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # ==================== LOAD DATA ====================
    print("\n" + "="*40)
    print("Loading Data...")
    print("="*40)
    
    # Train data
    train_images, train_labels = [], []
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(Config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_images.append(os.path.join(class_dir, img_name))
                    train_labels.append(class_idx)
    
    train_labels = np.array(train_labels)
    print(f"Train: {len(train_images)} images")
    
    # Test data
    test_images = sorted([
        os.path.join(Config.TEST_DIR, f) 
        for f in os.listdir(Config.TEST_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Test: {len(test_images)} images")
    
    # ==================== EXTRACT FEATURES ====================
    print("\n" + "="*40)
    print("Extracting Features...")
    print("="*40)
    
    # Transform
    transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load pre-trained model (feature extractor only)
    print(f"Loading {Config.MODEL_NAME}...")
    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    
    # DataLoaders
    train_dataset = SimpleDataset(train_images, transform)
    test_dataset = SimpleDataset(test_images, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Extract features
    print("\nExtracting train features...")
    train_features = extract_features(model, train_loader, device)
    print(f"  Train features shape: {train_features.shape}")
    
    print("\nExtracting test features...")
    test_features = extract_features(model, test_loader, device)
    print(f"  Test features shape: {test_features.shape}")
    
    # L2 Normalize features (better for ViT embeddings - cosine distance)
    # StandardScaler làm méo hình học embedding
    print("  L2 normalizing features...")
    train_features_norm = normalize(train_features, axis=1)
    test_features_norm = normalize(test_features, axis=1)
    
    # ==================== CLUSTERING ====================
    print("\n" + "="*40)
    print("Clustering (Fit on Train, Apply to Test)...")
    print("="*40)
    
    # FIT ON TRAIN, APPLY TO TEST (đúng methodology)
    kmeans = KMeans(n_clusters=Config.NUM_CLASSES, random_state=Config.SEED, n_init=10)
    train_clusters = kmeans.fit_predict(train_features_norm)  # Fit on train
    test_clusters = kmeans.predict(test_features_norm)         # Apply to test
    
    # Analyze train clustering quality
    print("\nTrain Cluster vs True Label (sanity check):")
    train_cluster_ari = adjusted_rand_score(train_labels, train_clusters)
    print(f"  ARI (Train Clusters vs True Labels): {train_cluster_ari:.4f}")
    if train_cluster_ari > 0.7:
        print("  → KMeans captures class structure well ✓")
    else:
        print("  → KMeans doesn't align perfectly with classes (expected for some data)")
    
    print("\nTest Cluster Distribution (using Train-fitted KMeans):")
    cluster_counts = Counter(test_clusters)
    for i in range(Config.NUM_CLASSES):
        count = cluster_counts.get(i, 0)
        print(f"  Cluster {i}: {count} ({count/len(test_clusters)*100:.1f}%)")
    
    # ==================== LOAD PREDICTIONS (if available) ====================
    test_predictions = None
    # Try multiple possible submission filenames
    possible_submissions = [
        'submission_ensemble.csv',
        'submission.csv', 
        'submission2.csv'
    ]
    
    submission_path = None
    for fname in possible_submissions:
        path = os.path.join(Config.OUTPUT_DIR, fname)
        if os.path.exists(path):
            submission_path = path
            break
    
    if submission_path:
        print("\n" + "="*40)
        print("Loading Model Predictions...")
        print("="*40)
        
        df = pd.read_csv(submission_path)
        class_to_idx = {c: i for i, c in enumerate(Config.CLASSES)}
        test_predictions = df['TARGET'].map(class_to_idx).values
        
        print("\nPrediction Distribution:")
        pred_counts = Counter(test_predictions)
        for i in range(Config.NUM_CLASSES):
            count = pred_counts.get(i, 0)
            print(f"  {Config.CLASSES[i]}: {count} ({count/len(test_predictions)*100:.1f}%)")
    
    # ==================== ANALYSIS ====================
    print("\n" + "="*40)
    print("Distribution Comparison...")
    print("="*40)
    
    # Train distribution
    train_counts = Counter(train_labels)
    print("\nTrain Distribution (True Labels):")
    for i in range(Config.NUM_CLASSES):
        count = train_counts.get(i, 0)
        print(f"  {Config.CLASSES[i]}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    # Compare train vs test cluster centroids
    print("\n" + "="*40)
    print("Comparing Train Class Centroids vs KMeans Centroids...")
    print("="*40)
    
    # Compute train class centroids (based on TRUE labels)
    train_class_centroids = []
    for i in range(Config.NUM_CLASSES):
        mask = train_labels == i
        if mask.sum() > 0:
            centroid = train_features_norm[mask].mean(axis=0)
        else:
            centroid = np.zeros(train_features_norm.shape[1])
        train_class_centroids.append(centroid)
    train_class_centroids = np.array(train_class_centroids)
    
    # KMeans centroids (from training)
    kmeans_centroids = kmeans.cluster_centers_
    
    # Compute distance matrix: Train Class → KMeans Cluster
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(train_class_centroids, kmeans_centroids, metric='cosine')
    
    print("\nDistance Matrix (Train Class → KMeans Cluster):")
    print("Lower = More Similar (Cosine Distance)")
    print("\n       ", end="")
    for j in range(Config.NUM_CLASSES):
        print(f"  C{j:d}  ", end="")
    print()
    for i in range(Config.NUM_CLASSES):
        print(f"{Config.CLASSES[i]:>5}  ", end="")
        for j in range(Config.NUM_CLASSES):
            print(f"{distance_matrix[i,j]:5.3f}", end="")
        print()
    
    # Find best mapping (Hungarian algorithm)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    print("\nOptimal Mapping (Train Class → KMeans Cluster):")
    for i, j in zip(row_ind, col_ind):
        print(f"  {Config.CLASSES[i]} → Cluster {j} (cosine dist: {distance_matrix[i,j]:.3f})")
    
    # Remap test clusters to class labels using optimal mapping
    cluster_to_class = {j: i for i, j in zip(row_ind, col_ind)}
    test_pseudo_labels = np.array([cluster_to_class.get(c, 0) for c in test_clusters])
    
    print("\nTest Pseudo-Label Distribution (after Hungarian mapping):")
    pseudo_counts = Counter(test_pseudo_labels)
    for i in range(Config.NUM_CLASSES):
        count = pseudo_counts.get(i, 0)
        print(f"  {Config.CLASSES[i]}: {count} ({count/len(test_pseudo_labels)*100:.1f}%)")
    
    # ==================== VISUALIZATION ====================
    print("\n" + "="*40)
    print("Creating Visualizations...")
    print("="*40)
    
    # Sample for t-SNE (too slow for full dataset)
    n_sample = min(3000, len(train_features))
    sample_idx = np.random.choice(len(train_features), n_sample, replace=False)
    train_sample = train_features_norm[sample_idx]
    train_labels_sample = train_labels[sample_idx]
    
    n_test_sample = min(3000, len(test_features))
    test_sample_idx = np.random.choice(len(test_features), n_test_sample, replace=False)
    test_sample = test_features_norm[test_sample_idx]
    test_clusters_sample = test_clusters[test_sample_idx]
    
    # t-SNE plot
    plot_tsne(
        train_sample, test_sample, train_labels_sample, test_clusters_sample,
        save_path=os.path.join(Config.OUTPUT_DIR, 'tsne_train_vs_test.png')
    )
    
    # Distribution comparison plot
    plot_cluster_comparison(
        train_labels, test_clusters, test_predictions,
        save_path=os.path.join(Config.OUTPUT_DIR, 'distribution_comparison.png')
    )
    
    # ==================== CLUSTER VS PREDICTION ANALYSIS ====================
    # Initialize variables for summary
    pseudo_vs_pred_ari = None
    pseudo_vs_pred_match = None
    
    if test_predictions is not None:
        print("\n" + "="*40)
        print("Comparing Clusters vs Model Predictions...")
        print("="*40)
        
        ari, nmi = analyze_cluster_purity(test_clusters, test_predictions)
        
        # Plot cluster vs prediction confusion
        fig, ax = plt.subplots(figsize=(10, 8))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_clusters, test_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=Config.CLASSES, yticklabels=[f'C{i}' for i in range(9)])
        ax.set_xlabel('Model Prediction', fontweight='bold')
        ax.set_ylabel('Cluster', fontweight='bold')
        ax.set_title(f'Cluster vs Prediction\nARI={ari:.3f}, NMI={nmi:.3f}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cluster_vs_prediction.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved: cluster_vs_prediction.png")
        
        # KEY INSIGHT: Compare pseudo-labels vs predictions
        print("\n" + "="*40)
        print("KEY COMPARISON: Pseudo-Labels vs Model Predictions")
        print("="*40)
        
        pseudo_vs_pred_ari = adjusted_rand_score(test_pseudo_labels, test_predictions)
        pseudo_vs_pred_match = (test_pseudo_labels == test_predictions).mean()
        
        print(f"\nPseudo-Labels (from clustering) vs Model Predictions:")
        print(f"  Exact Match Rate: {pseudo_vs_pred_match:.2%}")
        print(f"  Adjusted Rand Index: {pseudo_vs_pred_ari:.4f}")
        
        if pseudo_vs_pred_match > 0.7:
            print("  → Model predictions ALIGN with cluster structure ✓")
        elif pseudo_vs_pred_match > 0.4:
            print("  → Model predictions PARTIALLY align with clusters ⚠️")
        else:
            print("  → Model predictions DO NOT align with clusters ❌")
            print("  → Đây có thể là nguyên nhân OOF cao nhưng LB thấp!")
        
        # Distribution comparison table
        print("\n" + "-"*60)
        print("Distribution Comparison Table:")
        print("-"*60)
        print(f"{'Class':<6} {'Train%':>8} {'Pseudo%':>8} {'Pred%':>8} {'Status':<10}")
        print("-"*60)
        
        train_dist = Counter(train_labels)
        pseudo_dist = Counter(test_pseudo_labels)
        pred_dist = Counter(test_predictions)
        n_train_total = len(train_labels)
        n_test_total = len(test_predictions)
        
        for i in range(Config.NUM_CLASSES):
            train_pct = train_dist.get(i, 0) / n_train_total * 100
            pseudo_pct = pseudo_dist.get(i, 0) / n_test_total * 100
            pred_pct = pred_dist.get(i, 0) / n_test_total * 100
            
            # Check if prediction is way off from both train and pseudo
            if abs(pred_pct - train_pct) > 15 or abs(pred_pct - pseudo_pct) > 15:
                status = "❌ MISMATCH"
            elif abs(pred_pct - train_pct) > 8:
                status = "⚠️ Check"
            else:
                status = "✓ OK"
            
            print(f"{Config.CLASSES[i]:<6} {train_pct:>7.1f}% {pseudo_pct:>7.1f}% {pred_pct:>7.1f}% {status:<10}")
        
        print("-"*60)
    
    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"""
KẾT QUẢ PHÂN TÍCH:
------------------
1. Train Cluster ARI: {train_cluster_ari:.4f}
   {"→ KMeans captures class structure well ✓" if train_cluster_ari > 0.7 else "→ Classes có overlap trong feature space"}

2. Train-Test Domain:
   - Xem tsne_train_vs_test.png để kiểm tra overlap
   
3. Distribution Check:
   - Xem distribution_comparison.png
   
{f"4. Model Prediction Analysis:" if test_predictions is not None else "4. Model Predictions: Không tìm thấy submission file"}
{f"   - Pseudo vs Pred Match: {pseudo_vs_pred_match:.1%}" if pseudo_vs_pred_match is not None else ""}
{f"   - ARI: {pseudo_vs_pred_ari:.4f}" if pseudo_vs_pred_ari is not None else ""}

KHUYẾN NGHỊ:
------------
• Nếu Pseudo vs Pred match thấp (<50%):
  → Model đang map classes SAI
  → Check CLASSES order trong code vs actual folder order
  
• Nếu Train-Test overlap kém (trong t-SNE):
  → Domain shift giữa train và test
  → Cần domain adaptation hoặc augmentation mạnh hơn

• Nếu một class có Pred% khác nhiều với Train% và Pseudo%:
  → Model bị bias với class đó
  → Check class weights / sampling
""")
    
    print("\n" + "="*60)
    print("DONE! Check outputs folder for visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()