import os
import random
import shutil

# Frames paths
frames_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Data_Frames\HUSD\Val"
bal_frames_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Dataset\HUSD\Val"

# Copy random files from original folder

for dirs in os.listdir(frames_path):
    subset = random.sample(os.listdir(os.path.join(frames_path, dirs)), 850)
    for file in subset:
        shutil.copyfile(os.path.join(frames_path, dirs, file), os.path.join(bal_frames_path, dirs, file))