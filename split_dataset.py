import os
import shutil
import random

dataset_path = 'Banglalekha'
train_path = os.path.join(dataset_path, 'Train')
test_path = os.path.join(dataset_path, 'Test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

classes = [str(i) for i in range(10)]

for cls in classes:
    cls_dir = os.path.join(dataset_path, cls)
    if not os.path.isdir(cls_dir):
        continue
        
    train_cls_dir = os.path.join(train_path, cls)
    test_cls_dir = os.path.join(test_path, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)
    
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    random.seed(42)
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    for img in train_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))
        
    for img in test_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(test_cls_dir, img))
        
    # Verify and remove old empty dirs
    if not os.listdir(cls_dir):
        os.rmdir(cls_dir)
        
print("Dataset successfully split into 80% Train and 20% Test.")
