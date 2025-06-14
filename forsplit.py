import os
import shutil
import random

def split_data(src_dir, dest_dir, split_ratio=0.8):
    categories = os.listdir(src_dir)
    for category in categories:
        category_path = os.path.join(src_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        images = os.listdir(category_path)
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        for phase, phase_imgs in zip(['train', 'test'], [train_imgs, test_imgs]):
            target_dir = os.path.join(dest_dir, phase, category)
            os.makedirs(target_dir, exist_ok=True)
            
            for img in phase_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(target_dir, img)
                shutil.copy2(src, dst)

        print(f"[✓] {category} — Train: {len(train_imgs)}, Test: {len(test_imgs)}")

# Usage
split_data('data', 'data_split', split_ratio=0.8)
