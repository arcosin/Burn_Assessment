import cv2
import os

# Paths
path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Online"
dest_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\PNG_format"

# Acc variable
cont = 0

for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    new_name = str(cont) + ".png"
    cv2.imwrite(os.path.join(dest_path, new_name), img)
    cont += 1



