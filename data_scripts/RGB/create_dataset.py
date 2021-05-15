import os
import random
import shutil

# Data paths
org_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Segmentation"
dest_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset"
labels_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Ground_Truth"
mask_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks"

# Create dataset TOTAL 388. Training: 270, Validation: 59, Testing: 59

training = random.sample(os.listdir(org_path), 270)
test_val_list = []

for file in os.listdir(org_path):
    if file in training:
        shutil.copyfile(os.path.join(org_path, file), os.path.join(dest_path, "Train", file))
    else:
        test_val_list.append(file)

val = random.sample(test_val_list, 59)

for file in test_val_list:
    if file in val:
        shutil.copyfile(os.path.join(org_path, file), os.path.join(dest_path, "Val", file))
    else:
        shutil.copyfile(os.path.join(org_path, file), os.path.join(dest_path, "Test", file))

# Create labels

sections = ["Train", "Val", "Test"]

for section in sections:
    for file in os.listdir(os.path.join(dest_path, section)):
        name = file.split(".")[0] + ".png___fuse.png"
        shutil.copyfile(os.path.join(mask_path, name),
                        os.path.join(labels_path, section, name))


