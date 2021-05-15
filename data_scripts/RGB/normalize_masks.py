import os
import cv2

org_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Ground_Truth"
dest_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Normalized"

for folder in os.listdir(org_path):
    for file in os.listdir(os.path.join(org_path, folder)):
        img = cv2.imread(os.path.join(org_path, folder, file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(256):
            for j in range(256):
                print(gray_img[i][j])
                if gray_img[i][j] == 190:
                    gray_img[i][j] = 255
                elif gray_img[i][j] == 242:
                    gray_img[i][j] = 127
                else:
                    gray_img[i][j] = 0
        cv2.imwrite(os.path.join(dest_path, folder, file.split(".")[0] + ".png"), gray_img)


