import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

train_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train"
test_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\test1"

def advanced_comparison(sample_size=300):
    train_paths = []
    for cls in os.listdir(train_dir):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for img in os.listdir(cls_dir):
            train_paths.append(os.path.join(cls_dir, img))
    
    test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg','.png'))]
    
    np.random.shuffle(train_paths)
    np.random.shuffle(test_paths)
    
    train_samples = train_paths[:sample_size]
    test_samples = test_paths[:sample_size]
    
    results = {
        "TRAIN": {"resolutions": {}, "brightness_distribution": [0]*10, "edge_density": []},
        "TEST": {"resolutions": {}, "brightness_distribution": [0]*10, "edge_density": []}
    }
    
    def process_images(paths, key):
        resolutions = {}
        hist_bins = np.zeros(10)
        edge_densities = []
        
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            
            # 1. Thu thập Độ phân giải
            h, w = img.shape[:2]
            res_key = f"{w}x{h}"
            resolutions[res_key] = resolutions.get(res_key, 0) + 1
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Histogram Độ Sáng (Chia làm 10 mức độ trải dài 0-255)
            # Xem thử Test set có bị dư sáng (Overexposed) hay thiếu sáng (Underexposed) hơn Train không
            hist, _ = np.histogram(gray, bins=10, range=(0, 256))
            hist_bins += hist
            
            # 3. Mật độ cấu trúc cạnh (Edge Density qua Canny)
            # Tế bào trong tập Test có bị rách màng hay kết cấu nhân thùy bị vỡ nham nhở không?
            edges = cv2.Canny(gray, 100, 200)
            edge_densities.append(np.sum(edges > 0) / (h * w))
            
        results[key]["resolutions"] = resolutions
        results[key]["brightness_distribution"] = (hist_bins / np.sum(hist_bins)).tolist() # Chuẩn hóa về tỷ lệ %
        results[key]["edge_density"] = np.mean(edge_densities)

    process_images(train_samples, "TRAIN")
    process_images(test_samples, "TEST")
    
    return results

if __name__ == "__main__":
    results = advanced_comparison()
    with open(r"c:\Users\admin\Documents\Project\bloodcellclass\final_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Final Advanced Comparison Complete.")
