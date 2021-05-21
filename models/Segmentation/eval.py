import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from customized_unet import UNet
from torchsummary import summary
import matplotlib as plt
import cv2


class BurnDataset(Dataset):

    """
    Class to create our customized dataset
    """

    def __init__(self, inputs_dir, masks_dir):
        self.inputs_dir = inputs_dir
        self.masks_dir = masks_dir
        self.data = os.listdir(self.inputs_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index].split(".")[0]
        input_file = os.path.join(self.inputs_dir, file_name + ".png")
        mask_file = os.path.join(self.masks_dir, file_name + ".png")
        image = Image.open(input_file)  # RGB
        mask = Image.open(mask_file)
        image = np.array(image)
        image = image.transpose(2, 0, 1) / 255
        mask = np.array(mask)
        target = torch.zeros(256, 256)
        target[mask == 0] = 0
        target[mask == 127] = 1
        target[mask == 255] = 2

        im, ground_t = torch.from_numpy(image).type(torch.FloatTensor), target
        return im, ground_t


def reverse_transform(img):
    img_array = np.array(img)
    img_array = img_array.transpose((1, 2, 0))
    return img_array


if __name__ == "__main__":

    # Define variables
    batch_size = 4
    num_classes = 3
    num_channels = 3
    device = torch.device("cuda:0")
    acc = 0.0

    # Location of dataset
    data_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset\Test"
    masks_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Greyscale\Test"
    weights_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Models\Segmentation\Results_Train_12\UNet_std.pth"

    # Create testing dataset
    test_burn_dataset = BurnDataset(data_dir, masks_dir)

    # Create testing dataloader
    test_dataloader = DataLoader(test_burn_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = UNet(num_channels, num_classes)
    model.load_state_dict(torch.load(weights_dir))
    model = model.to(device)
    # summary(model, input_size=(3, 256, 256))
    model.eval()

    # Testing model

    for images, targets in test_dataloader:
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.long)
        predictions = model(images)
        _, preds = torch.max(predictions, 1)
        acc += torch.sum(preds == targets.data) / (256**2)
        preds = preds.data.cpu().numpy()

        rgb_img = [reverse_transform(img) for img in images.cpu()]
        cv2.imshow("image", cv2.cvtColor(rgb_img[2], cv2.COLOR_RGB2BGR))
        cv2.imshow("mask", ((np.array(targets.data.cpu())*127).astype(np.uint8())[2]))
        cv2.imshow("pred", ((np.array(preds)*127).astype(np.uint8)[2]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    print('Testing Accuracy: {:.4f}'.format(acc/len(test_dataloader.dataset)))

