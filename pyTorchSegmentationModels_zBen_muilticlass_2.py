# Taken from: https://smp.readthedocs.io/en/latest/index.html
# and: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
# 24/05/2021
# Modified by JW to load zBeaconHill data set

# Additional Hyper-Parameters / Settings taken from:
# https://github.com/drivendataorg/open-cities-ai-challenge/tree/master/1st%20Place
# https://github.com/drivendataorg/open-cities-ai-challenge - 1st Place Example
# 26/05/2021

import albumentations as albu
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

########################################################################################################################
# Settings
########################################################################################################################

# ORG: 'se_resnext50_32x4d', 'efficientnet-b1', 'efficientnet-b7','timm-efficientnet-b1', 'se_resnext101_32x4d'
# 'timm-efficientnet-b7'
ENCODER = 'efficientnet-b1'
# b7,6,5 too big for 1024^2 input,
# b4 works with 1024x1024,
# b5 works with 768*768

NUMBER_OF_FOLDS = 1
DISPLAY_TRAINING_DATA = False
DISPLAY_PREDICTIONS = False
ARCHITECTURE = 'Unet'  # OPTIONS ARE: 'Unet' ORG:'FPN'
ENCODER_WEIGHTS = 'imagenet'  # 'noisy-student',   'imagenet'
ACTIVATION = 'sigmoid'  # ORG:'sigmoid',
START_WITH_TRAINED = False  # If True, loading in model name and continue training
DEVICE = 'cuda'
EPOCHS_MAX = 40
str_date = datetime.today().strftime('%Y%m%d')
MODEL_NAME_ROOT = ENCODER + '_' \
                 + ARCHITECTURE \
                 + '_' \
                 + str_date \
                 + '_Med768x768_LL_MSK_fold_'
TRAIN_ = 'Train'  # 'Train', 'Test', 'Run'
TRAIN_BATCH_SIZE = 2  # 4 and 8 don't work with unet and efficientnet-b1 at 768*768

input_width = 768  # 1024  # 320 or 420
input_height = 768  # 1024  # 320 or  380

# original class setup - nothing special about it, kept for ease of re-trying earlier setup.
CLASSES = ['white', 'green', 'yellow', 'medium_gray', 'black']  # Lithology setup
# CLASSES = ['dont_care', 'green', 'red_orange', 'yellow', 'orange', 'light_gray', 'medium_grey', 'dark_gray', 'black'] # Arch Elements Setup
# CLASSES = ['dont_care', 'green']  # Grass Setup
# CLASSES = ['dont_care', 'important_edge']  # Grass Setup

# Input directory root - Don't include final fold number
ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data 32k All BH_RS_KN/920x_1X0y_0p03mpx/autogen_dataset_75_25_fold_'

#'C:/Users/r01jw19/Documents/Data 32k Beacon Hill/960x_120y_0p03mpx_rockClean_masks/autogen_dataset_75_25_fold_'


# old folders for ease of use:
# ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data 32k Hayburn/960x_1X0y_0p03mpp/autogen_dataset_75_25_fold_'
# ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data Kettleness 32k/960x_110y_120z_0p03mpp/autogen_dataset_75_25_fold_'
# ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data Beacon Hill 32k/960x_120y_120z_0p03mpp/autogen_dataset_75_25_fold_'
# ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data 32k Yorkshire/960x_Xy_Xz_0p03mpp/autogen_dataset_75_25_fold_'
# ROOT_DATA_DIR = 'C:/Users/r01jw19/Documents/Data 32k aLL Yorkshire/960x_Xy_0p03mpp/autogen_dataset_75_25_fold_'
########################################################################################################################
# _________________ Nov26 Results Reports _____________________________________________________________________________
#

# _________________ Nov22 Results Reports _____________________________________________________________________________
# See note book, forgot to update this section... for version with masks
# Epochs: 15  Duration: 1:07:45.982777  T_IOU: 0.74  V_IOU: 0.78  TeIOU: 0.81 efficientnet-b1 imagenet Unet - No mask

# _________________ Old Results Reports _______________________________________________________________________________
# 2022/05/09 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, ~14k crops before QC 32 Yorkshore - White Bgrd
# Epochs: 40  Duration: 9:29:25.159688  T_IOU: 0.82  V_IOU: 0.68  TeIOU: 0.69 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 9:36:47.449051  T_IOU: 0.80  V_IOU: 0.77  TeIOU: 0.68 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 9:37:35.995551  T_IOU: 0.81  V_IOU: 0.70  TeIOU: 0.69 efficientnet-b1 imagenet Unet

# 2022/05/02 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, ~2500 crops inputs Hayburn - White BgroundNo
# Epochs: 40  Duration: 2:24:46.960219  T_IOU: 0.85  V_IOU: 0.77  TeIOU: 0.71 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 2:23:59.647990  T_IOU: 0.85  V_IOU: 0.75  TeIOU: 0.71 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 2:23:54.351803  T_IOU: 0.86  V_IOU: 0.67  TeIOU: 0.72 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 2:24:36.663666  T_IOU: 0.86  V_IOU: 0.68  TeIOU: 0.71 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 2:24:02.239263  T_IOU: 0.86  V_IOU: 0.69  TeIOU: 0.69 efficientnet-b1 imagenet Unet

# 2022/04/27 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, ~2500 crops inputs Kettleness - White Bground
# Epochs: 40  Duration: 0:56:53.067538  T_IOU: 0.91  V_IOU: 0.86  TeIOU: 0.73 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:58:25.515316  T_IOU: 0.91  V_IOU: 0.91  TeIOU: 0.62 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:59:24.569751  T_IOU: 0.93  V_IOU: 0.77  TeIOU: 0.71 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:59:54.811796  T_IOU: 0.91  V_IOU: 0.92  TeIOU: 0.77 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:57:02.130433  T_IOU: 0.90  V_IOU: 0.88  TeIOU: 0.64 efficientnet-b1 imagenet Unet

# 2022/04/26 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, ~2500 crops inputs Kettleness
# Epochs: 40  Duration: 0:59:04.024596  T_IOU: 0.90  V_IOU: 0.84  TeIOU: 0.77 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:57:19.975275  T_IOU: 0.90  V_IOU: 0.91  TeIOU: 0.79 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:57:28.050806  T_IOU: 0.92  V_IOU: 0.79  TeIOU: 0.68 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:57:29.509458  T_IOU: 0.82  V_IOU: 0.67  TeIOU: 0.66 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:57:27.622808  T_IOU: 0.89  V_IOU: 0.87  TeIOU: 0.73 efficientnet-b1 imagenet Unet
# Comment: Problem input images did not have white backgrounds. So a lot of miss classification of COAL class.
# Redoing test.

# 2022/04/06 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, ~8000 crops inputs Beacon Hill and Ravenscar
# Stopped after ~12hours and 2 folds.
# Epochs: 40  Duration: 6:21:31.671167  T_IOU: 0.81  V_IOU: 0.52  TeIOU: 0.74 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 6:21:38.676349  T_IOU: 0.77  V_IOU: 0.70  TeIOU: 0.77 efficientnet-b1 imagenet Unet
# Next fold stopped at ~30 epochs.


# 2022/04/05 LL, No rockClean, 768*768 crops, 32k 0.03mpp, 40ep, 5 folds, fixed the mosaic'ing -
# Epochs: 40  Duration: 3:06:58.881214  T_IOU: 0.82  V_IOU: 0.79  TeIOU: 0.81 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:06:51.413970  T_IOU: 0.84  V_IOU: 0.65  TeIOU: 0.82 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:06:57.722249  T_IOU: 0.84  V_IOU: 0.63  TeIOU: 0.83 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:07:03.793787  T_IOU: 0.84  V_IOU: 0.66  TeIOU: 0.83 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:06:59.501350  T_IOU: 0.82  V_IOU: 0.81  TeIOU: 0.82 efficientnet-b1 imagenet Unet

# 2022/04/04 LL(Lithology only), No rockClean etc, 768*768 crops, 32k 0.03mpp, 40ep
# 5 folds
# Comments: Once mosaic'd some of the classes are not stacking or in single outputs. Check training data.
# Epochs: 40  Duration: 3:03:46.578403  T_IOU: 0.87  V_IOU: 0.56  TeIOU: 0.84 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:04:14.195041  T_IOU: 0.80  V_IOU: 0.70  TeIOU: 0.88 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:04:24.525127  T_IOU: 0.81  V_IOU: 0.70  TeIOU: 0.88 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:04:31.842551  T_IOU: 0.81  V_IOU: 0.72  TeIOU: 0.87 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 3:04:40.315180  T_IOU: 0.81  V_IOU: 0.83  TeIOU: 0.87 efficientnet-b1 imagenet Unet

# 2021/11/10 Liner: High Pass RGB as input, Edge Detected Lines of Interp as Output. 768*768 crops.
# 5 folds
# All white output - ie did not learn to get lines...

# 2021/11/09 Lith with new 'runners' work flow. Check we can reproduce earlier work. 768*768.
# 5 folds
# Epochs: 40  Duration: 0:36:32.615003  T_IOU: 0.89  V_IOU: 0.81  TeIOU: 0.75 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:36:31.956555  T_IOU: 0.89  V_IOU: 0.83  TeIOU: 0.76 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:36:36.088741  T_IOU: 0.88  V_IOU: 0.83  TeIOU: 0.73 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:36:35.766872  T_IOU: 0.85  V_IOU: 0.76  TeIOU: 0.65 efficientnet-b1 imagenet Unet
# Epochs: 40  Duration: 0:36:36.919144  T_IOU: 0.89  V_IOU: 0.85  TeIOU: 0.75 efficientnet-b1 imagenet Unet
# ERROR Used the wrong number of classes in training, but looks quite good Qua 4.
# 5 Folds, B6, correct CLASSES in training
# Epochs: 40  Duration: 1:28:04.257002  T_IOU: 0.92  V_IOU: 0.76  TeIOU: 0.73 efficientnet-b6 imagenet Unet
# Epochs: 40  Duration: 1:28:00.681809  T_IOU: 0.92  V_IOU: 0.81  TeIOU: 0.74 efficientnet-b6 imagenet Unet
# Epochs: 40  Duration: 1:28:15.734402  T_IOU: 0.92  V_IOU: 0.83  TeIOU: 0.76 efficientnet-b6 imagenet Unet
# Epochs: 40  Duration: 1:28:16.624430  T_IOU: 0.92  V_IOU: 0.83  TeIOU: 0.71 efficientnet-b6 imagenet Unet
# Epochs: 40  Duration: 1:28:15.999536  T_IOU: 0.92  V_IOU: 0.85  TeIOU: 0.75 efficientnet-b6 imagenet Unet

# 2021/11/08 Arch with Masked Predictions as input, Arch Interp (No TED, No QC) output, 768*768
# 2 Folds -
# Epochs: 30  Duration: 0:15:47.689784  T_IOU: 0.72  V_IOU: 0.70  TeIOU: 0.74 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:51.485335  T_IOU: 0.77  V_IOU: 0.77  TeIOU: 0.74 efficientnet-b1 imagenet Unet
# Note when mosaic'ed up these are REALLY BAD Qua:1. Something weird going on in processing.
# 2 FOlds - Fix the problem with not cropping outer edges of image. 768*768.
# Epochs: 30  Duration: 0:15:47.587606  T_IOU: 0.78  V_IOU: 0.74  TeIOU: 0.75 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:47.061757  T_IOU: 0.76  V_IOU: 0.77  TeIOU: 0.75 efficientnet-b1 imagenet Unet

# 2021/11/06 Lith with NO TEDBaT 5FCV
# 5 Folds -
# Epochs: 30  Duration: 0:13:39.496523  T_IOU: 0.73  V_IOU: 0.68  TeIOU: 0.72 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:13:36.188416  T_IOU: 0.78  V_IOU: 0.76  TeIOU: 0.75 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:13:42.419364  T_IOU: 0.76  V_IOU: 0.78  TeIOU: 0.75 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:13:39.461844  T_IOU: 0.77  V_IOU: 0.73  TeIOU: 0.75 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:13:37.609881  T_IOU: 0.73  V_IOU: 0.67  TeIOU: 0.70 efficientnet-b1 imagenet FPN
# 5 Folds -
# Epochs: 30  Duration: 0:15:12.361995  T_IOU: 0.71  V_IOU: 0.68  TeIOU: 0.71 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:13.546825  T_IOU: 0.77  V_IOU: 0.77  TeIOU: 0.76 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:00.590367  T_IOU: 0.78  V_IOU: 0.80  TeIOU: 0.75 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:06.642139  T_IOU: 0.72  V_IOU: 0.73  TeIOU: 0.72 efficientnet-b1 imagenet Unet
# Epochs: 30  Duration: 0:15:25.488711  T_IOU: 0.79  V_IOU: 0.70  TeIOU: 0.74 efficientnet-b1 imagenet Unet

# 2021/11/05 Lith with Tedbat, Normal? 5FCV
# Epochs: 30  Duration: 0:36:16.099953  T_IOU: 0.86  V_IOU: 0.75  TeIOU: 0.68 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:36:13.461726  T_IOU: 0.87  V_IOU: 0.76  TeIOU: 0.69 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:36:12.499400  T_IOU: 0.83  V_IOU: 0.70  TeIOU: 0.62 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:36:15.068218  T_IOU: 0.85  V_IOU: 0.79  TeIOU: 0.69 efficientnet-b1 imagenet FPN
# Epochs: 30  Duration: 0:36:17.453104  T_IOU: 0.86  V_IOU: 0.76  TeIOU: 0.68 efficientnet-b1 imagenet FPN

# 2021/11/05 Arch with TEDBaT, High Pass with Dip, 3FCV
# Three Fold Test:
# Epochs: 10  Duration: 0:26:33.249429  T_IOU: 0.70  V_IOU: 0.56  TeIOU: 0.48 efficientnet-b1 imagenet FPN
# Epochs: 10  Duration: 0:26:38.594792  T_IOU: 0.70  V_IOU: 0.34  TeIOU: 0.42 efficientnet-b1 imagenet FPN
# Epochs: 10  Duration: 0:26:35.597889  T_IOU: 0.74  V_IOU: 0.44  TeIOU: 0.65 efficientnet-b1 imagenet FPN

# 2021/11/05 Arch Only, High Pass Input No TEDBaT, 6FCV
# Fold 0 - Need to get this better before moving on
# Epochs: 5  Duration: 0:14:31.649327  T_IOU: 0.35  V_IOU: 0.28  TeIOU: 0.16 efficientnet-b1 imagenet FPN, LR=0.00005
# Epochs: 5  Duration: 0:14:29.737385  T_IOU: 0.49  V_IOU: 0.41  TeIOU: 0.22 efficientnet-b1 imagenet FPN, LR=0.0001 HIGHER


# 2021/11/04 Arch Only High Pass input - No TEDBaT,
# Epochs: 10  Duration: 0:29:43.174063  T_IOU: 0.76  V_IOU: 0.20  TeIOU: 0.21 efficientnet-b1 imagenet FPN
# Epochs: 35  Duration: 1:32:36.093845  T_IOU: 0.89  V_IOU: 0.24  TeIOU: 0.31 efficientnet-b1 imagenet FPN, LR=0.00005,
#                                                                   overfitting badly, but training output is REALLY GOOD
# Epochs: 35  Duration: 1:32:46.318909  T_IOU: 0.74  V_IOU: 0.23  TeIOU: 0.35 efficientnet-b1 imagenet FPN, Less AUG.
# Epochs: 70  Duration: 2:12:06.339116  T_IOU: 0.49  V_IOU: 0.14  TeIOU: 0.10 efficientnet-b1 imagenet FPN Less Aug.
#                                                                             random Crop 480, POOR Qaul, can't mosaic
# Epochs: 10  Duration: 0:21:35.846808  T_IOU: 0.39  V_IOU: 0.17  TeIOU: 0.25 efficientnet-b0 imagenet FPN,
#                                                                             Med Aug. 768*768. LR=0.00005

########################################################################################################################

# Accessed 20211019
# From: https://github.com/yearing1017/PyTorch_Note/blob/master/PolyLr.py
# Don't know if this is the origin of this class...
# modified to not be a proper callback
class PolyLR:
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, base_lrs, max_iter, power, last_epoch=-1):
        self.base_lrs = base_lrs
        self.max_iter = max_iter
        self.power = power
        self.last_epoch = -1

    def get_lr(self, last_epoch):
        return self.base_lrs * (1 - last_epoch / self.max_iter) ** self.power


# helper function for data visualization
def visualize(**images):
    """Plot images in two rows."""
    n = len(images)
    if n > 1:
        col = (n + 1) // 2
        row = (n + 1) // col
        fig, axs = plt.subplots(row, col, figsize=(16, 5))
        axs = axs.ravel()
        for i, (name, image_) in enumerate(images.items()):
            # axs[i].title(' '.join(name.split('_')).title())
            axs[i].imshow(image_, vmin=0, vmax=1)
        plt.show()


class Dataset(BaseDataset):
    """zBeacon Hill Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        # Remove the random desktop.ini file that has appeared in the folder and can't be seen in
        # windows explorer to delete it...
        # print('Old 0 Index:', self.ids[0])
        if self.ids[0] == 'desktop.ini':
            self.ids.remove('desktop.ini')
            # print('Desktop.ini removed')
            # print('New 0 Index:', self.ids[0])

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.ids = os.listdir(masks_dir)
        # Remove the random desktop.ini file that has appeared in the folder and can't be seen in
        # windows explorer to delete it...
        # print('Old 0 Index:', self.ids[0])
        if self.ids[0] == 'desktop.ini':
            self.ids.remove('desktop.ini')
            # print('Desktop.ini removed')
            # print('New 0 Index:', self.ids[0])

        self.masks_fps = [os.path.join(masks_dir, image_id.replace('RGBMSK', 'IntMsk')) for image_id in self.ids]

        # convert str names to class values on masks, IE 'sandstone' become 1
        self.classes = classes
        self.class_values = [self.classes.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# Reduced almost everything by half
# removed colour changes
def get_training_augmentation():
    train_transform = [

        # albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=input_height, width=input_width, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.1),
        albu.IAAPerspective(p=0.1),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.45,  # 0.9
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.45,  # 0.9
        ),
        # p=0.9 original,
        # 0.45 used for early test on 1kx1k images = overfitting validation.
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                # albu.HueSaturationValue(p=1),
            ],
            p=0.45,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""

    test_transform = [
        albu.Resize(input_height, input_width),  # JW added this to get validation data to the right size...
        albu.CenterCrop(height=input_height, width=input_width, p=1.0)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn_):
    """Construct preprocessing transform

    Args:
        preprocessing_fn_ (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn_),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

if __name__ == '__main__':
    # Get the Model name and Data Directories ready for doing all the folds of data
    DATA_DIRS = []
    MODEL_NAMES = []
    for i in range(NUMBER_OF_FOLDS):
        DATA_DIRS.append(ROOT_DATA_DIR + str(i))
        MODEL_NAMES.append(MODEL_NAME_ROOT + str(i) + '_TeZpZZ.pth')

    # Introduction Message
    print("Number of Fold: " + str(NUMBER_OF_FOLDS))
    print("\nDataset From:")
    for d in DATA_DIRS:
        print("\t", d)
    print("Starting Training...")

    # Loop through all the folds and report
    for DATA_DIR, MODEL_NAME in zip(DATA_DIRS, MODEL_NAMES):
        print('\n...Next fold...\n')
        print('DATA_DIR:', DATA_DIR)
        print('MODEL_NAME:', MODEL_NAME, flush=True)

        # load the repo with data if it does not exist
        if not os.path.exists(DATA_DIR):
            print('Data Directory does not exist! Exiting...')
            exit()
        else:
            x_train_dir = os.path.join(DATA_DIR, 'train')
            y_train_dir = os.path.join(DATA_DIR, 'trainannot')

            x_valid_dir = os.path.join(DATA_DIR, 'val')
            y_valid_dir = os.path.join(DATA_DIR, 'valannot')

            # Test Normally
            x_test_dir = os.path.join(DATA_DIR, 'test')
            y_test_dir = os.path.join(DATA_DIR, 'testannot')

        now = datetime.now()
        now_string = str(now.year) + str(now.month) + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

        multiprocessing.freeze_support()

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        loss = smp.utils.losses.DiceLoss()  # Org: DiceLoss for binary segmentation
        # loss = smp.utils.losses.JaccardLoss()  # for multiclass segmentation

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        if TRAIN_ == 'Train':
            # 1 - Get data and look at it
            dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)

            # TODO automate the channel depending on the number of classes...
            if DISPLAY_TRAINING_DATA:
                # get a 2 sample image and display masks(annotations / labels)
                for j in range(2):
                    image, mask = dataset[j]
                    kw_images = {'image': image}
                    for i, cls in enumerate(CLASSES):
                        kw_images[str(cls)] = mask[:, :, i]
                    visualize(**kw_images)

                # 2 Augment data and Visualize resulted augmented images and masks
                augmented_dataset = Dataset(
                    x_train_dir,
                    y_train_dir,
                    augmentation=get_training_augmentation(),
                    classes=CLASSES,
                )

                # same image with 8 different random transforms
                for j in range(8):
                    image, mask = augmented_dataset[1]
                    kw_images = {'image': image}
                    for i, cls in enumerate(CLASSES):
                        kw_images[str(cls)] = mask[:, :, i]
                    visualize(**kw_images)

            ### 3 - Create and train the model ###

            # create segmentation model with pretrained encoder
            if not START_WITH_TRAINED:
                if ARCHITECTURE == 'FPN':
                    model = smp.FPN(
                        encoder_name=ENCODER,
                        encoder_weights=ENCODER_WEIGHTS,
                        classes=len(CLASSES),
                        activation=ACTIVATION,
                    )
                elif ARCHITECTURE == 'Unet':
                    model = smp.Unet(
                        encoder_name=ENCODER,
                        encoder_weights=ENCODER_WEIGHTS,
                        classes=len(CLASSES),
                        activation=ACTIVATION,
                    )
            else:
                # load saved model under existing MODEL_NAME
                # TODO figure out how to get ENCODER etc to align with loaded model... ignore for now
                model = torch.load(MODEL_NAME)

            train_dataset = Dataset(
                x_train_dir,
                y_train_dir,
                augmentation=get_training_augmentation(),
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            valid_dataset = Dataset(
                x_valid_dir,
                y_valid_dir,
                augmentation=get_validation_augmentation(),
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            # using 1, seems slower on 8
            train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

            # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
            # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=0.0001),  # lr=0.0001
            ])

            poly_lr_scheduler = PolyLR(base_lrs=0.0001, max_iter=EPOCHS_MAX, power=0.9)

            # create epoch runners
            # it is a simple loop of iterating over dataloader`s samples
            train_epoch = smp.utils.train.TrainEpoch(
                model,
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                device=DEVICE,
                verbose=False,
            )

            valid_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=metrics,
                device=DEVICE,
                verbose=False,
            )

            start_time = datetime.now()

            # train model for EPOCHS_MAX epochs
            max_score = 0
            max_score_train = 0
            if TRAIN_ == 'Train':
                for i in range(0, EPOCHS_MAX):

                    print('\nEpoch: {}'.format(i))
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)

                    # do something (save model, change lr, etc.)
                    if max_score < valid_logs['iou_score']:
                        max_score = valid_logs['iou_score']
                        torch.save(model, MODEL_NAME)
                        print('Model saved!')

                    if max_score_train < train_logs['iou_score']:
                        max_score_train = train_logs['iou_score']

                    optimizer.param_groups[0]['lr'] = poly_lr_scheduler.get_lr(i, )
                    print('Decoder learning rate:' + str(optimizer.param_groups[0]['lr']))

            end_time = datetime.now()

        ### 4 - Test best saved model ###

        # load best saved checkpoint
        # Remember we always start from the BEST saved model
        # Not the last one saved...
        best_model = torch.load(MODEL_NAME)

        # create test dataset
        test_dataset = Dataset(
            x_test_dir,
            y_test_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        test_dataloader = DataLoader(test_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )

        max_score_test = 0
        test_logs = test_epoch.run(test_dataloader)

        ### Get Predictions ###

        # test dataset without transformations for image visualization
        test_dataset_vis = Dataset(
            x_test_dir,
            y_test_dir,
            classes=CLASSES,
        )

        ### Code to generate the results report ###
        if TRAIN_ == 'Train':
            for i in test_logs:
                if max_score_test < test_logs['iou_score']:
                    max_score_test = test_logs['iou_score']

            print('\n', '_' * 17, 'Training And Best Model Test Report', '_' * 17)
            print('Epochs:', EPOCHS_MAX,
                  ' Duration:', (end_time - start_time),
                  ' T_IOU: {:.2f}'.format(max_score_train),
                  ' V_IOU: {:.2f}'.format(max_score),
                  ' TeIOU: {:.2f}'.format(max_score_test),
                  ENCODER,
                  ENCODER_WEIGHTS,
                  ARCHITECTURE)

        ### If required display random examples from the test dataset ###
        if DISPLAY_PREDICTIONS:
            for i in range(7):
                n = np.random.choice(len(test_dataset))

                image_vis = test_dataset_vis[n][0].astype('uint8')
                image, gt_mask = test_dataset[n]

                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                pr_mask = best_model.predict(x_tensor[:, :, :, :])
                pr_mask_cpu = pr_mask.cpu()

                kw_images = {'image_0': image_vis}
                for i, cls in enumerate(CLASSES):
                    kw_images['ground_truth_' + str(cls)] = gt_mask[i, :, :]

                kw_images['image_1'] = image_vis
                for j, cls in enumerate(CLASSES):
                    kw_images['predicted_' + str(cls)] = pr_mask_cpu[0, j, :, :]

                visualize(**kw_images)

    # _________________Old Datsset Results Report __________________________________________________________________________

    # 2021/11/04 Arch Only High Pass input
    # Epochs: 10  Duration: 0:29:53.643242  T_IOU: 0.73  V_IOU: 0.23  TeIOU: 0.21 efficientnet-b1 imagenet FPN, badly overfitting
    #
    # 2021/11/04 Arch Only High Pass input - No TEDBaT
    # Epochs: 10  Duration: 0:29:54.303100  T_IOU: 0.67  V_IOU: 0.20  TeIOU: 0.22 efficientnet-b1 imagenet FPN
    #
    # 2021/11/04 Arch Input Red-Previous Prediction, Green - Dip, Blue - Hp RGB image. 768*768.
    # Epochs: 10  Duration: 0:29:52.733127  T_IOU: 0.75  V_IOU: 0.24  TeIOU: 0.32 efficientnet-b1 imagenet FPN
    #
    # 2021/11/04 Inputs RG from Dip and B from High Passed RGB image. 768*768
    # Forgot to copy end statement E10 T_IOU:0.8 V_IOU:0.7 TeIOU: 0.27, predictions look poor Q:1 or 2.
    # 2021/11/04 Similar to above but Hp on Green Channel and Dip /2 on R+B. 768*768. Given Imagenet PreTrain and Human vis.
    # Epochs: 10  Duration: 0:28:08.620517  T_IOU: 0.80  V_IOU: 0.72  TeIOU: 0.15 efficientnet-b1 imagenet FPN
    #
    # 2021/11/03 Inputs now RGB image, Red=Prediction of Litho, Green=Processed Dip, Blue=High Passed RGB at 768*768
    # Epochs: 10  Duration: 0:24:49.332921  T_IOU: 0.78  V_IOU: 0.74  TeIOU: 0.49 efficientnet-b1 imagenet FPN
    # Epochs: 10  Duration: 0:24:48.561214  T_IOU: 0.80  V_IOU: 0.75  TeIOU: 0.46 efficientnet-b1 imagenet FPN, Low Augmentation - Especialy for Colour Changes
    # Manual stop @19 Epoch: 4 train: iou_score - 0.8141 valid: iou_score - 0.7233], I think the augmentation was low, so byE19 lots of overfitting
    # Epochs: 40  Duration: 3:52:07.763967  T_IOU: 0.91  V_IOU: 0.71  TeIOU: 0.37 efficientnet-b5 imagenet FPN, normal Aug.
    # Epochs: 3  Duration: 0:17:29.104740  T_IOU: 0.74  V_IOU: 0.71  TeIOU: 0.36 efficientnet-b5 imagenet FPN, Short test to see if overfitting reduced...
    #
    # 2021/11/02 Using the original Beacon Hill Images cropped to 768x768 (Lime is currently broken)
    # Manual Stop @11 train: iou_score - 0.8643, valid: iou_score - 0.9363, camvid colour encoding is wrong
    # Epochs: 10  Duration: 0:24:41.492633  T_IOU: 0.73  V_IOU: 0.86  TeIOU: 0.73 efficientnet-b1 imagenet FPN, 1/2 fix col
    # Epochs: 10  Duration: 0:24:36.098629  T_IOU: 0.85  V_IOU: 0.91  TeIOU: 0.76 efficientnet-b1 imagenet FPN, Fully fixed
    # Epochs: 10  Duration: 0:24:32.057142  T_IOU: 0.79  V_IOU: 0.92  TeIOU: 0.77 efficientnet-b1 imagenet FPN, No TEDBAT, Missing Light Gray Output
    # Epochs: 10  Duration: 0:25:01.520636  T_IOU: 0.80  V_IOU: 0.79  TeIOU: 0.44 efficientnet-b1 imagenet FPN, No NonGeoMsk
    # Epochs: 10  Duration: 0:11:08.060871  T_IOU: 0.80  V_IOU: 0.66  TeIOU: 0.73 efficientnet-b1 imagenet FPN, NonGeoMsk + Blank and Green QC
    #
    # 2021/10/18 20 Panels at 4kx4k, Cropped to 1k*1k. 50-25-25. Lithology only, with Grass only NN mask.
    # Manual Stop @e10, train: iou_score - 0.8787, valid: iou_score - 0.5611, BeEp: 9 (V_iou 0.57) System overfitting badly
    # Epochs: 10  Duration: 1:52:36.684799  T_IOU: 0.87  V_IOU: 0.62  TeIOU: 0.70 efficientnet-b4 imagenet FPN, More Augmentation
    #
    # 2021/10/18 20 Panels at 4kx4k, Cropped to 1k*1k. 50-25-25. Lithology only.
    # Manual stop @E10, train: iou_score - 0.8895, valid: iou_score - 0.5492, BeEp: 2, System overfitting badly
    #
    # 2021 10 12 LarX 1024*1024, QC50%, 80-10-10. Lithology Only.
    # Epochs: 10  Duration: 0:33:38.459330  T_IOU: 0.81  V_IOU: 0.81  TeIOU: 0.73 efficientnet-b1 imagenet FPN
    # Epochs: 80  Duration: 4:18:29.620974  T_IOU: 0.96  V_IOU: 0.86  TeIOU: 0.78 efficientnet-b1 imagenet FPN,med/high .aug, looking better
    # Epochs: 10  Duration: 0:57:27.395778  T_IOU: 0.89  V_IOU: 0.64  TeIOU: 0.69 efficientnet-b4 imagenet FPN, has potential - train more?
    #
    # 2021 10 13 LarX 1024*1024, QC50%, BIG CHANGE IN RATIOS: 50-25-25. Lithology Only.
    # Epochs: 40  Duration: 2:39:50.715961  T_IOU: 0.97  V_IOU: 0.77  TeIOU: 0.76 efficientnet-b4 imagenet FPN
    #
    # 2021 10 08 LarX dataset, try with input images 1024 by 1024, TVT: 90/05/05
    #
    # 2021 10 07 Lar dataset //4 (~16k crops) with QC at >25% blank == remove from training. TVT: 90/05/05
    # Epochs: 80  Duration: 5:03:12.607893  T_IOU: 0.73  V_IOU: 0.55  TeIOU: 0.69 efficientnet-b7 imagenet FPN
    #
    # 2021 10 07 Lar dataset //4 (~16k crops) with QC at >25% blank == remove from training. TVT: 60/20/20
    # Epochs: 10  Duration: 0:24:40.094173  T_IOU: 0.66  V_IOU: 0.45  TeIOU: 0.62 efficientnet-b7 imagenet FPN
    # Epochs: Stopped after 11... the dataset with spilts is likely not balanced well
    #
    # 2021 10 07 Same Lar //2 dataset, but with green regions, made black and 50% black QC applied (92% removed from dset)
    # ~ 1100 input images
    # Epochs: 10  Duration: 0:06:47.215109  T_IOU: 0.40  V_IOU: 0.31  TeIOU: 0.35 efficientnet-b1 imagenet FPN
    # Epochs: 10  Duration: 0:13:13.849446  T_IOU: 0.49  V_IOU: 0.37  TeIOU: 0.47 efficientnet-b7 imagenet FPN
    # Epochs: 45  Duration: 0:59:17.449101  T_IOU: 0.55  V_IOU: 0.40  TeIOU: 0.48 efficientnet-b7 imagenet FPN
    #
    # 2021 10 06 Lar 4k dataset, int quantised and both masked, tr 60%, va 20%, te 20% splits, LR decreasing
    # Epochs: 10  Duration: 0:31:07.090338  T_IOU: 0.76  V_IOU: 0.91  TeIOU: 0.76 efficientnet-b1 imagenet FPN
    # Epochs: 45  Duration: 2:18:01.681126  T_IOU: 0.78  V_IOU: 0.91  TeIOU: 0.77 efficientnet-b1 imagenet FPN
    #
    # 2021 12 06 Lar Dataset, Phase 2 of Quantizing, tr 60%, va 20%, te 20% splits, LR decreasing
    # Epochs: 45  Duration: 8:23:59.973413  T_IOU: 0.72  V_IOU: 0.95  TeIOU: 0.86 efficientnet-b1 imagenet FPN
    # Same as above but trained with lesser LR at start.
    # Epochs: 10  Duration: 1:48:21.756416  T_IOU: 0.64  V_IOU: 0.94  TeIOU: 0.85 efficientnet-b1 imagenet FPN BeEp9
    #
    # 2021 10 04 Lar Dataset, with manually quantized interp images. Changed the Split to 40% tr, 20% va, 40% test
    # Epoch: 42 crashed, TeIoU: 0.71
    # 2021 10 05 Added in a vertical flip and REMOVED COLOUR augmentations
    # Epochs: 10  Duration: 0:13:47.869136  T_IOU: 0.95  V_IOU: 0.86  TeIOU: 0.90 efficientnet-b1 imagenet FPN,
    # Epochs: 10  Duration: 0:29:35.043213  T_IOU: 0.92  V_IOU: 0.78  TeIOU: 0.84 efficientnet-b7 imagenet FPN, BeEp9
    # With learning rate stepped decay
    # Epochs: 45  Duration: 1:01:42.186802  T_IOU: 0.97  V_IOU: 0.87  TeIOU: 0.91 efficientnet-b1 imagenet FPN, BeEp35
    #
    # 2021 09 024 Lar Dataset, with manually quantized interp images. Phase 1.
    # Epochs: 45  Duration: 0:25:32.557093  T_IOU: 0.78  V_IOU: 0.91  TeIOU: 0.90 efficientnet-b1 imagenet FPN
    # Epochs: 45  Duration: 0:25:43.209229  T_IOU: 0.80  V_IOU: 0.96  TeIOU: 0.96 efficientnet-b1 imagenet FPN, Jaccard loss
    # Same dataset as above but with more crops
    # Epochs: 45  Duration: 1:20:46.331522  T_IOU: 0.77  V_IOU: 0.80  TeIOU: 0.77 efficientnet-b1 imagenet FPN
    #
    # Lar Dataset from new found high res panel exports skills
    # Dataset has new vegetation / grass / green class and we have run both rockQC and rockConvert_Splitter
    # Epochs: 40  Duration: 0:23:36.853735  T_IOU: 0.71  V_IOU: 0.67  TeIOU: 0.71, All aug, 'efficientnet-b1', unet, BeEp39
    # Epochs: 40  Duration: 0:10:34.383148  T_IOU: 0.76  V_IOU: 0.71  TeIOU: 0.73, All aug, 'efficientnet-b1', unet, BeEp30
    # Epochs: 50  Duration: 0:13:06.727081  T_IOU: 0.81  V_IOU: 0.72  TeIOU: 0.72 efficientnet-b1 imagenet FPN, BeEp22
    # Epochs: 35  Duration: 0:18:00.354670  T_IOU: 0.83  V_IOU: 0.73  TeIOU: 0.74 efficientnet-b7 imagenet FPN, BeEp30
    # Epochs: 35  Duration: 0:17:47.109701  T_IOU: 0.80  V_IOU: 0.67  TeIOU: 0.72 timm-efficientnet-b7 noisy-student FPN,Be?
    #
    # Same input Lar Dataset with less stride and thus more crops per input image - no rockQC, No:Grass Manual Mask
    # Epochs: 35  Duration: 0:36:44.599474  T_IOU: 0.86  V_IOU: 0.78  TeIOU: 0.78 efficientnet-b1 imagenet FPN
    #
    # Same input, Lar Dataset with less stride and thus more crops per input image - no rockQC, Yes:Grass Manual Mask
    # Epochs: 35  Duration: 0:36:31.535502  T_IOU: 0.86  V_IOU: 0.78  TeIOU: 0.81 efficientnet-b1 imagenet FPN, No oranges!
    #
    # Lar Dataset with Grass only, phase 2 of training -> labelling
    # Epochs: 35  Duration: 0:24:06.123233  T_IOU: 0.82  V_IOU: 0.70  TeIOU: 0.77 efficientnet-b1 imagenet FPN
    # Epochs: 35  Duration: 0:24:06.100467  T_IOU: 0.81  V_IOU: 0.71  TeIOU: 0.77 efficientnet-b1 imagenet FPN, dice loss
    #
    # Test of putting quantised image back through the system to train with itself
    # - This is not much of a test but at least it proves that the most basic thing works
    # Epochs: 40  Duration: 0:12:03.044361  T_IOU: 0.91  V_IOU: 0.99  TeIOU: 0.99
    #
    # Test of putting AT version of interp. (IE straight out of Lime) and training again manually quantized image (no inter)
    # Epochs: 40  Duration: 0:10:39.143560  T_IOU: 0.90  V_IOU: 0.94  TeIOU: 0.94, efficientnet-b1, Unet
    # Epochs: 80  Duration: 0:41:07.571425  T_IOU: 0.96  V_IOU: 0.98  TeIOU: 0.97, 'efficientnet-b7','FPN','imagenet', BeEp29
    #
    # Same Dataset but with 320x320 input images, in effect no random cropping but much larger dataset
    # Epochs: 20  Duration: 0:17:05.165354  T_IOU: 0.93  V_IOU: 0.94  TeIOU: 0.94 efficientnet-b1 imagenet FPN
    # Stopped manually TeIOU_score - 0.9498 efficientnet-b7 imagenet FPN BeEp~<20, qualitatively looks great. Jacard loss
    #
    # datasetv4 (test is same as validation)
    # Epochs: 40  Duration: 0:08:03.290663  T_IOU: 0.89  V_IOU: 0.76  TeIOU: 0.76, Only Flip Aug.
    # Epochs: 40  Duration: 0:08:10.900289  T_IOU: 0.87  V_IOU: 0.77  TeIOU: 0.77, Bval:30, More Aug.
    #
    # Sma Dataset with Interpretation (IE NOT Lines version) 359 image-label pairs
    # Epochs: 40  Duration: 0:19:42.375712  T_IOU: 0.94  V_IOU: 0.88  TeIOU: 0.74
    #
    # Sma Dataset with Interpretation extended to 1188 image-label pairs, data set
    # Epochs: 40  Duration: 0:30:26.026700  T_IOU: 0.85  V_IOU: 0.80  TeIOU: 0.80,
    #
    # Sma Dataset with Interpretation extended to 1188 image-label pairs then over image > 50% blank removed, ~310 images
    # Epochs: 40  Duration: 0:11:38.320006  T_IOU: 0.80  V_IOU: 0.60  TeIOU: 0.61, no augmentations, evidence of overfit
    # Epochs: 40  Duration: 0:20:45.115432  T_IOU: 0.66  V_IOU: 0.63  TeIOU: 0.66, All aug., BestEp36
    # Epochs: 40  Duration: 0:20:44.902056  T_IOU: 0.47  V_IOU: 0.50  TeIOU: 0.57, All aug, 'softmax2d',  BestEp21
    # Epochs: 40  Duration: 0:22:02.913780  T_IOU: 0.54  V_IOU: 0.55  TeIOU: 0.58, All aug, 'softmax2d', Unet BestEp16
    # Epochs: 40  Duration: 0:21:25.787448  T_IOU: 0.59  V_IOU: 0.56  TeIOU: 0.53, All aug, 'mob.netv2' fpn, sigmoid, BestEp
    # Epochs: 40  Duration: 0:20:40.589565  T_IOU: 0.60  V_IOU: 0.58  TeIOU: 0.51, All aug, 'resnet152' fpn, sigmoid, Best21
    #       Stopped at Ep52. Best Epoch 15: T_IOU: 0.52, V_IOU: 0.601 TeIOU: 0.5948, All aug, se_resnext101_32x4d fpn BestEp
    # Epochs: 50  Duration: 0:27:03.881031  T_IOU: 0.46  V_IOU: 0.48  TeIOU: 0.46, all aug,  'efficientnet-b1', unet BE48
    # Most of the test dataset were not aligned between label and prediction. Problem was with validation ds cropping input
