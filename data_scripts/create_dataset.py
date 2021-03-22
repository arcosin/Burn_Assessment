import os
import random
import shutil

# Frames paths
org_frames_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Burn_Dataset\HUSD"
dest_frames_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Burn_Dataset_Split_2\HUSD"

# Copy random files from original folder

for dirs in os.listdir(org_frames_path):
    training = random.sample(os.listdir(os.path.join(org_frames_path, dirs)), 6000)
    for file in os.listdir(os.path.join(org_frames_path, dirs)):
        if file in training:
            shutil.copyfile(os.path.join(org_frames_path, dirs, file), os.path.join(dest_frames_path, "Train",
                                                                                    dirs, file))
        else:
            shutil.copyfile(os.path.join(org_frames_path, dirs, file), os.path.join(dest_frames_path, "Test",
                                                                                    dirs, file))



''' For Train, Test, Val

for dirs in os.listdir(org_frames_path):
    training = random.sample(os.listdir(os.path.join(org_frames_path, dirs)), 3900)
    test_list = []
    for file in os.listdir(os.path.join(org_frames_path, dirs)):
        if file in training:
            shutil.copyfile(os.path.join(org_frames_path, dirs, file), os.path.join(dest_frames_path, "Train",
                                                                                    dirs, file))
        else:
            test_list.append(file)
    testing = random.sample(test_list, 1300)
    for file in test_list:
        if file in testing:
            shutil.copyfile(os.path.join(org_frames_path, dirs, file), os.path.join(dest_frames_path, "Test",
                                                                                    dirs, file))
        else:
            shutil.copyfile(os.path.join(org_frames_path, dirs, file), os.path.join(dest_frames_path, "Val",
                                                                                    dirs, file))

'''