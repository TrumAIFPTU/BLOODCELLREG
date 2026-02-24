import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

# Config paths
TEST_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\test1"
OUTPUT_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\outputs"
PSEUDO_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\pseudo_train"
ORIGINAL_TRAIN_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train"
COMBINED_TRAIN_DIR = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train_combined"

CLASSES = sorted(['BA', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PMY', 'SNE'])

# Confidence threshold for pseudo-labeling
THRESHOLD = 0.95

def create_pseudo_labels():
    print(f"--- Pseudo-Labeling Script ---")
    print(f"Threshold set to: {THRESHOLD * 100}%\n")
    
    # 1. Load probabilities and test image list
    probs_path = os.path.join(OUTPUT_DIR, 'ensemble_test_probs.npy')
    if not os.path.exists(probs_path):
        print("Error: ensemble_test_probs.npy not found! Please run all_models.py first.")
        return
        
    probs = np.load(probs_path)
    
    # We need the exact order of test images as they were processed
    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    assert len(test_images) == len(probs), "Mismatch between number of images and predictions!"
    
    # 2. Find high confidence predictions
    max_probs = np.max(probs, axis=1)
    pred_classes = np.argmax(probs, axis=1)
    
    confident_indices = np.where(max_probs > THRESHOLD)[0]
    print(f"Found {len(confident_indices)} out of {len(test_images)} images with > {THRESHOLD} confidence.")
    
    # 3. Create fresh pseudo_train directory structure
    if os.path.exists(PSEUDO_DIR):
        shutil.rmtree(PSEUDO_DIR)
    
    class_counts = {cls: 0 for cls in CLASSES}
    
    print("\nCopying high confidence images to pseudo_train directory...")
    for idx in tqdm(confident_indices):
        img_name = test_images[idx]
        class_idx = pred_classes[idx]
        class_name = CLASSES[class_idx]
        
        # Create class directory if not exists
        class_dir = os.path.join(PSEUDO_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file
        src_path = os.path.join(TEST_DIR, img_name)
        dst_path = os.path.join(class_dir, f"pseudo_{img_name}")
        shutil.copy2(src_path, dst_path)
        
        class_counts[class_name] += 1
        
    print("\nPseudo-labels generated per class:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")

    # 4. Merge robustly into a single new COMBINED_TRAIN_DIR
    print(f"\nMerging original train and pseudo_train into: {COMBINED_TRAIN_DIR}")
    if os.path.exists(COMBINED_TRAIN_DIR):
        shutil.rmtree(COMBINED_TRAIN_DIR)
        
    os.makedirs(COMBINED_TRAIN_DIR, exist_ok=True)
    
    total_combined = 0
    for cls in CLASSES:
        combined_cls_dir = os.path.join(COMBINED_TRAIN_DIR, cls)
        os.makedirs(combined_cls_dir, exist_ok=True)
        
        # Copy from original
        orig_cls_dir = os.path.join(ORIGINAL_TRAIN_DIR, cls)
        if os.path.exists(orig_cls_dir):
            for img in os.listdir(orig_cls_dir):
                shutil.copy2(os.path.join(orig_cls_dir, img), os.path.join(combined_cls_dir, img))
                total_combined += 1
                
        # Copy from pseudo
        pseudo_cls_dir = os.path.join(PSEUDO_DIR, cls)
        if os.path.exists(pseudo_cls_dir):
            for img in os.listdir(pseudo_cls_dir):
                shutil.copy2(os.path.join(pseudo_cls_dir, img), os.path.join(combined_cls_dir, img))
                total_combined += 1
                
    print(f"\nSuccessfully created combined training set with {total_combined} images.")
    print("\nNext step: Modify 'Config.TRAIN_DIR' in 'all_models.py' to point to this new combined directory:")
    print(f"  TRAIN_DIR = r'{COMBINED_TRAIN_DIR}'")

if __name__ == "__main__":
    create_pseudo_labels()
