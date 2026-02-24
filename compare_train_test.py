import os
import cv2
import numpy as np
import json

train_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\train"
test_dir = r"c:\Users\admin\Documents\Project\bloodcellclass\data\test1"

def analyze_dataset_properties(directory, sample_size=200):
    color_variances = []
    blur_metrics = []
    saturation_means = []
    
    # Collect flat list of images
    if directory == train_dir:
        image_paths = []
        for cls in os.listdir(directory):
            cls_dir = os.path.join(directory, cls)
            if not os.path.isdir(cls_dir): continue
            images = os.listdir(cls_dir)
            for img_name in images:
                image_paths.append(os.path.join(cls_dir, img_name))
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    else:
        images = os.listdir(directory)
        np.random.shuffle(images)
        image_paths = [os.path.join(directory, img) for img in images[:sample_size]]
        
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        means, stds = cv2.meanStdDev(img)
        color_variances.append(stds.flatten().tolist())
        
        saturation_means.append(np.mean(hsv[:,:,1]))
        blur_metrics.append(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "Color_Var_B_G_R": [round(x, 2) for x in np.mean(color_variances, axis=0).tolist()],
        "Mean_Saturation": round(np.mean(saturation_means), 2),
        "Mean_Blur_Var": round(np.mean(blur_metrics), 1)
    }

if __name__ == "__main__":
    results = {
        "TRAIN_SET_OVERALL": analyze_dataset_properties(train_dir),
        "TEST_SET_OVERALL": analyze_dataset_properties(test_dir)
    }
    
    with open(r"c:\Users\admin\Documents\Project\bloodcellclass\train_test_compare.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Comparison complete.")
