import os
import cv2
import json

# Ultrasound videos reading path
r_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Burn_Data_Videos_Trim\HUSD"

# Elastography videos reading path
# path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Burn_Data_Videos_Trim\TDI"

# Ultrasound frames writing path
w_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Burn_Data_Frames_Classes\HUSD"


def extract_frames_labels(video, count, write_path, dict):

    """ Extracts and saves every frame of the input video in the corresponding category (Full,
    partial, superficial). It returns the counter variable so the next time the function is
    called, the counter will start where it left. It returns the frame-labels dictionary to
    continue adding pairs when the function is called again"""

    cap = cv2.VideoCapture(video)
    ret = True
    try:
        while ret:
            ret, frame = cap.read()
            count += 1
            cv2.imwrite(os.path.join(write_path, "frame{}.jpg".format(count)), frame)
            dict["frame{}.jpg".format(count)] = video.split("\\")[9]
    except:
        print("Video is over")
    return count, dict


# Initialize counter for frame extraction
count_var = 0

# Initialize dictionary with frame-label pairs
labels_dict = {}

# Navigate directories recursively to obtain the frames for each video
for dirs in os.listdir(r_path):
    for subdir in os.listdir(os.path.join(r_path, dirs)):
        for filename in os.listdir(os.path.join(r_path, dirs, subdir)):
            file_path = os.path.join(r_path, dirs, subdir, filename)
            count_var, labels_dict = extract_frames_labels(file_path, count_var, os.path.join(w_path, dirs),
                                                           labels_dict)
            print(count_var)

# Create json file with frame-label pairs
json_datafile = open("labels.json", "w")
json.dump(labels_dict, json_datafile, indent=2)
json_datafile.close()

