import os
import shutil
import random

def split_dataset(
    raw_dir='./Data',
    out_dir='./Dataset',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    random.seed(seed)
    class_names = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    for class_name in class_names:
        class_dir = os.path.join(raw_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Make output directories
        for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(out_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)
        print(f"Class '{class_name}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

if __name__ == '__main__':
    split_dataset(
        raw_dir='Comp9444_Proj045/Data',
        out_dir='Dataset',     # where you want train/val/test folders
        train_ratio=0.8,         # 80% train
        val_ratio=0.1,           # 10% val
        test_ratio=0.1           # 10% test
    )

