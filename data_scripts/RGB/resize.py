import cv2
import os

# Paths
path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\PNG_format"
dest_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Segmentation"

# Acc variable

for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    new_img = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join(dest_path, file), new_img)

