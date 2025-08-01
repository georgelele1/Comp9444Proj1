import pandas as pd
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

src_root = r'D:\Users\natha\Desktop\Coding\Uni\comp9444\group project\grouped_ds'
dst_root = r'D:\Users\natha\Desktop\Coding\Uni\comp9444\group project\dataset'

df = pd.read_csv("metadata.csv")

# can manipulate data below if needed - change the class distribution etc

# split dataset into train, val, test sets (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

splits = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

for split_name, split_df in splits.items():
    for class_name in np.unique(split_df['class']):
        class_data = split_df[split_df['class'] == class_name]

        class_folder = os.path.join(dst_root, split_name, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for file_name in class_data['file name'].values:
            src_file = os.path.join(src_root, file_name)
            dst_file = os.path.join(class_folder, file_name)
            shutil.copy2(src_file, dst_file)
