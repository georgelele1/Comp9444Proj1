import os
import shutil

# Set source and destination paths


import os
import shutil

src_root = 'C:/Users/Georgelele/Downloads/EBH-HE-IDS/ColHis-IDS/Polyp'  # your input data
dst_root = 'Comp9444_Proj045/Data/Polyp'  # flattened output

os.makedirs(dst_root, exist_ok=True)

# Go through each class folder
for class_name in os.listdir(src_root):
    class_path = os.path.join(src_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # Walk through subdirectories recursively
    for root, dirs, files in os.walk(class_path):
        for file in files:
            src_file = os.path.join(root, file)
            if os.path.isfile(src_file):
                # Build new filename: class_filename
                new_filename = f"{class_name}_{file}"
                dst_file = os.path.join(dst_root, new_filename)

                # Handle duplicate filenames
                i = 1
                name, ext = os.path.splitext(file)
                while os.path.exists(dst_file):
                    new_filename = f"{class_name}_{name}_{i}{ext}"
                    dst_file = os.path.join(dst_root, new_filename)
                    i += 1

                shutil.copy2(src_file, dst_file)

