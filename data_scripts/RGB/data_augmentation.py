# from https://github.com/aleju/imgaug#example_images

import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import cv2

images_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset\Train"
masks_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Greyscale\Train"
w_images_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset_Augmented\Train"
w_masks_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Greyscale_Augmented\Train"

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
images = []
masks = []

for file in os.listdir(images_path):
    name = file.split(".")[0]
    image = Image.open(os.path.join(images_path, name + ".png"))
    image = np.array(image)
    images.append(image)
    mask = Image.open(os.path.join(masks_path, name + ".png"))
    mask = np.array(mask)
    mask = np.expand_dims(mask, 2)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)

seq = iaa.Sequential(
    [
        iaa.SomeOf((3, 5),
                   [
                       iaa.Fliplr(0.5),
                       iaa.Flipud(0.5),
                       iaa.OneOf([
                                    iaa.Add((-40, 40)),
                                    iaa.Multiply((0.5, 1.5))
                                 ]),
                       sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))),
                       sometimes(iaa.SaltAndPepper(0.1)),
                       iaa.OneOf([
                                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                    iaa.AverageBlur(k=(2, 5)),
                                    iaa.MedianBlur(k=(3, 5))
                                 ]),
                       iaa.MultiplySaturation((0.5, 1.5)),
                       sometimes(iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())),
                       iaa.GammaContrast((0.5, 2.0)),
                       sometimes(iaa.LogContrast(gain=(0.6, 1.4))),
                       iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
                       iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                       iaa.Rot90(1),
                       sometimes(iaa.CropAndPad(percent=(-0.25, 0.25))),
                       sometimes(iaa.Dropout(p=(0, 0.2)))

                   ], random_order=True)
    ], random_order=True)

images_aug, masks_aug = seq(images=images, segmentation_maps=masks)
print("Finished")

images_aug = list(images_aug)
masks_aug = list(masks_aug)

for i in range(270):
    im = cv2.cvtColor(images_aug[i], cv2.COLOR_RGB2BGR)
    label = np.squeeze(masks_aug[i], 2)
    cv2.imwrite(os.path.join(w_images_path, str(658+i) + ".png"), im)
    cv2.imwrite(os.path.join(w_masks_path, str(658 + i) + ".png"), label)
    # cv2.imshow("images", im)
    # cv2.imshow("mask", label)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
