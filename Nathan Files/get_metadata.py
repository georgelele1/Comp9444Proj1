import pandas as pd
import os
import shutil

src_root = r'D:\Users\natha\Desktop\Coding\Uni\comp9444\group project\original_ds'
dst_root = r'D:\Users\natha\Desktop\Coding\Uni\comp9444\group project\grouped_ds'

os.makedirs(dst_root, exist_ok=True)

df = pd.DataFrame()

for class_name in os.listdir(src_root):
    class_path = os.path.join(src_root, class_name)
    for root, dirs, files in os.walk(class_path):
        if len(dirs) == 0:
            magnification = int(list(root.split('\\'))[-1])
            instance = list(root.split('\\'))[-2]

            for i in range(len(files)):
                file_name = f'{class_name}_{magnification}_{instance}_{i}.jpg'
                data = pd.DataFrame([{
                    'class': class_name,
                    'magnification': magnification,
                    'file name': file_name
                }])
                df = pd.concat([df, data]).reset_index(drop=True)

                src_file = os.path.join(root, files[i])
                dst_file = os.path.join(dst_root, file_name)
                shutil.copy2(src_file, dst_file)

df.to_csv('metadata.csv')
