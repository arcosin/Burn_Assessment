import cv2
import os
import matplotlib.pyplot as plt

# Paths
org_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Data_Frames_Balanced\HUSD"
dest_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Dataset\HUSD"

# Crop coordinates HUSD: 155:920, 460:1465
# Crop coordinates TDI: 155:920, 275:1650

for dirs in os.listdir(org_path):
    for classes in os.listdir(os.path.join(org_path, dirs)):
        for frame in os.listdir(os.path.join(org_path, dirs, classes)):
            frame_path = os.path.join(org_path, dirs, classes, frame)
            org_frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
            crop_frame = org_frame[155:920, 460:1465]
            cv2.imwrite(os.path.join(dest_path, dirs, classes, frame), crop_frame)


