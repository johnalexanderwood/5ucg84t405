# Taken from: https://smp.readthedocs.io/en/latest/index.html
# and: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
# 24/05/2021
# Modified by JW to load zBeaconHill data set

# Additional Hyper-Parameters / Settings taken from:
# https://github.com/drivendataorg/open-cities-ai-challenge/tree/master/1st%20Place
# https://github.com/drivendataorg/open-cities-ai-challenge - 1st Place Example
# 26/05/2021
import copy
import pickle

import albumentations as albu
import cv2
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
### Settings ###

# --- Setup for Grass/Veg only ---
# Remember to change the rockDict also!
# LOAD_PREDICTIONS = False
# OUTPUT_SINGLE = False
# DISPLAY_RESULTS_SINGLE = False
# OUTPUT_STACKED = True
# DISPLAY_RESULTS_STACKED = False
# # MODEL_NAME = 'Grass_only_ph2_dice_Te0p77.pth'  # alternative MODEL_NAME = 'Grass_only_ph2_Te0p77.pth'
# MODEL_NAME = 'Grass_only_ph2_Te0p77.pth'
# DATA_DIR = 'C:/Users/r01jw19/Documents/Data Beacon Hill with Interp/LarX/Interp_4kx4k_non_geo/'
# X_CROPS_DIR = 'X_all_cropped_grass_div_4/'
# input_width = 320  # 480
# input_height = 320  # 360
# div_width = 4
# div_height = div_width
# output_width = 4096
# output_height = 4096
# rockDict = {
#     'black_dont_care': {
#         'colour': [0, 0, 0]
#     },
#     "green_vegetation": {
#         'colour': [0, 255, 0]
#     },
# }

# --- Setup for interp ---
LOAD_PREDICTIONS = False  # need to uncomment save predictions
OUTPUT_SINGLE = False
DISPLAY_RESULTS_SINGLE = False
OUTPUT_STACKED = True
DISPLAY_RESULTS_STACKED = False
USE_TEDBaT = False

MODEL_NAME = 'efficientnet-b1_Unet_20221126_Med768x768_LL_MSK_fold_0_Te0p85.pth'
#'efficientnet-b1_Unet_20221122_Med768x768_LL_MSK_fold_0_TeZpZZ.pth'
# 'efficientnet-b1_Unet_20221121_Med768x768_LL_fold_0_Te0p80.pth'

DATA_DIR = 'C:/Users/r01jw19/Documents/Data 32k Hayburn/960x_1X0y_0p03mpp'
#'C:/Users/r01jw19/Documents/Data 32k Beacon Hill/960x_120y_0p03mpx_rockClean_masks/'
# 'C:/Users/r01jw19/Documents/Data 32k Beacon Hill/960x_120y_0p03mpx/'

X_CROPS_DIR = 'ztmp_X_all_for_test/'

input_width = 768  # 1024  # 480
input_height = 768  # 1024  # 360
div_width = 2
div_height = 2
output_width = 32000 #1999  # 4096 #7999
output_height = 3333

rockDict_Lith = {
    'white_dont_care': {
        'colour': [255, 255, 255]
    },
    "green_vegetation": {
        'colour': [0, 255, 0]
    },
    "yellow_channel_sandstone": {
        'colour': [224, 192, 0]
    },
    "medium_grey_mudstone": {
        'colour': [96, 96, 96]
    },
    "black_coal": {
        'colour': [0, 0, 0]
    },
    # "extra_1": {
    #     'colour': [0, 0, 250]
    # },
    # "extra_2": {
    #     'colour': [0, 0, 200]
    # },
    # "extra_3": {
    #     'colour': [0, 0, 150]
    # },
    # "extra_4": {
    #     'colour': [0, 0, 100]
    # },
}

## Dictionary to hold the rock color and threshold information - The input thresholds are not used in this system
# rockDict = {
#     'white_dont_care': {
#         'colour': [255, 255, 255]
#     },
#     "green_vegetation": {
#         'colour': [0, 255, 0]
#     },
#     "red_orange_dogger": {
#         'colour': [224, 64, 32]
#     },
#     "yellow_channel_sandstone": {
#         'colour': [224, 192, 0]
#     },
#     "orange_crevasse_splay": {
#         'colour': [224, 128, 32]
#     },
#     "light_grey_mudstone": {
#         'colour': [160, 160, 160]
#     },
#     "medium_grey_mudstone": {
#         'colour': [96, 96, 96]
#     },
#     "dark_grey_mudstone": {
#         'colour': [60, 60, 60]
#     },
#     "black_coal": {
#         'colour': [0, 0, 0]
#     },
# }

ENCODER = 'efficientnet-b1'
ENCODER_WEIGHTS = 'imagenet'  # 'noisy-student',   'imagenet'
DEVICE = 'cuda'

########################################################################################################################

# Automatically Generate the CLASSES from the rockDict
CLASSES = list(rockDict_Lith.keys())

# load the repo with data if it does not exist
if not os.path.exists(DATA_DIR):
    print('Data Directory does not exist! Exiting...')
    exit()
else:
    x_test_dir = os.path.join(DATA_DIR, X_CROPS_DIR)

    # Output directory for results
    result_output_dir = os.path.join(DATA_DIR, 'y_pred_non_masked_' + MODEL_NAME + '/')

    # Create the output directory
    try:
        os.mkdir(result_output_dir)
    except FileExistsError:
        print('Output directory already exists.')
    except FileNotFoundError:
        print('Error in data path. Is there a typo? Exiting...')
        exit()


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

        self.masks_fps = [os.path.join(masks_dir, image_id.replace('', '')) for image_id in
                          self.ids]  # TODO naming problem here

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


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""

    test_transform = [
        #albu.Resize(input_height, input_width),  # JW added this to get validation data to the right size...
        albu.CenterCrop(height=input_height, width=input_width, p=1.0)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def from_tensor(x, **kwargs):
    return x.permute(1, 2, 0)


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


# TEDBaT algorithm
def TEDBaT(img_src, normalise_input=False, thresh_start=0.2, erode_itr=2, dilate_itr=1, kernel_size=3,
           blur_kernel_size=9, thresh_end=0.5, plot_result=False):
    # Check kernels are odd
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size

    # Make a copy
    _img = img_src.copy()

    # Normalise max to 1
    if normalise_input and (np.max(_img) > 0):
        _img *= (1 / np.max(_img))

    # threshold
    _img[_img < thresh_start] = 0
    _img[_img > 0] = 1

    # erode and dilate to smooth out channels
    kernel = np.ones((kernel_size, kernel_size))
    _img = cv2.erode(_img, kernel, iterations=erode_itr)
    _img = cv2.dilate(_img, kernel, iterations=dilate_itr)

    # Blur image and apply final threshold
    _img = cv2.GaussianBlur(_img, (blur_kernel_size, blur_kernel_size), 0)
    _img[_img < thresh_end] = 0
    _img[_img > 0] = 1

    if plot_result:
        plt.title('Before TEDBat')
        plt.imshow(img_src[:, :, 0])
        plt.colorbar()
        plt.show()
        plt.title('After TEDBat')
        plt.imshow(_img)
        plt.colorbar()
        plt.show()

    return _img


# input: the predictions with one prediction per channel
# output: the predictions with one RGB image with the original image
def stack_predictions(predictions, rockDict_, include='all', Use_TEDBaT=False):
    img_stack = np.zeros((output_height, output_width, 3))

    for j, rock in enumerate(rockDict_):
        if (include == 'all') or (rock in include):
            img_pred = predictions[:, :, j]
            if Use_TEDBaT:
                img_pred = TEDBaT(img_pred,
                                  normalise_input=True,
                                  thresh_start=0.5,  # parameter[0],
                                  erode_itr=1,  # int(parameter[1]),
                                  dilate_itr=2,  # int(parameter[2]),
                                  kernel_size=1,  # int(parameter[3]),
                                  blur_kernel_size=3,  # int(parameter[4]), # 101 for prediction or grass mask
                                  thresh_end=0.5,  # parameter[5],
                                  )
            img_stack[:, :, 0] += (img_pred * (rockDict_Lith[rock]['colour'][0] / 255.0))
            img_stack[:, :, 1] += (img_pred * (rockDict_Lith[rock]['colour'][1] / 255.0))
            img_stack[:, :, 2] += (img_pred * (rockDict_Lith[rock]['colour'][2] / 255.0))

    # Clip the image RGB channels so stacks don't exceed 0 or 1
    # TODO deal with the problem here, the clip is making the value CV_64F or some weird data type
    # should be 8 bit at the end
    img_stack[:, :, 0] = np.clip(img_stack[:, :, 0], 0, 1) * 255
    img_stack[:, :, 1] = np.clip(img_stack[:, :, 1], 0, 1) * 255
    img_stack[:, :, 2] = np.clip(img_stack[:, :, 2], 0, 1) * 255

    return img_stack.astype(np.uint8)  # conver from 64float


# Alternative Uncrop (Classic version with Gaps at bottom and right)
# Crop up the image, top to bottom, left to right
# tested with just fake x and y values and seems to be uncropping in order
# Edges round outside as expected
# def img_uncrop(img_list_, width=7999, height=1999, x_s=240, y_s=180):
#     img_ = img_list_[0]
#
#     x_dim = img_.shape[1]
#     y_dim = img_.shape[0]
#
#     imgs_result = []
#     if len(img_.shape) == 3:
#         i_ = 0
#         while i_ < len(img_list_):
#             img_uncrop_ = np.zeros((height, width, len(CLASSES)))
#             for x in range(0, width - input_width, x_s):
#                 for y in range(0, height - input_height, y_s):
#                     if i_ < len(img_list_):
#                         img_uncrop_[y:y + y_dim, x:x + x_dim, :] = img_list_[i_]  # [:, :, :]
#                     i_ += 1
#             imgs_result.append(copy.deepcopy(img_uncrop_))
#     elif len(img_.shape) == 2:
#         i_ = 0
#         while i_ < len(img_list_):
#             img_uncrop_ = np.zeros((height, width))
#             for x in range(0, width - input_width, x_s):
#                 for y in range(0, height - input_height, y_s):
#                     if i_ < len(img_list_):
#                         img_uncrop_[y:y + y_dim, x:x + x_dim] = img_list_[i_]
#                     i_ += 1
#             imgs_result.append(copy.deepcopy(img_uncrop_))
#     return imgs_result


def img_uncrop_full(img_list_, width=7999, height=1999, x_s=240, y_s=180):
    img_ = img_list_[0]

    x_dim = img_.shape[1]
    y_dim = img_.shape[0]

    imgs_result = []
    if len(img_.shape) == 3:
        i_ = 0
        while i_ < len(img_list_):
            img_uncrop_ = np.zeros((height, width, len(CLASSES)))
            for x in range(0, width - x_s, x_s):
                if x > (width - x_dim):  # If off end move back x_dim
                    x = width - x_dim
                for y in range(0, height - y_s, y_s):
                    if y > (height - y_dim):
                        y = height - y_dim

                    if i_ < len(img_list_):
                        img_uncrop_[y:y + y_dim, x:x + x_dim, :] = img_list_[i_]  # [:, :, :]
                    i_ += 1

                progress_bar(freetext=f'Progress Image {i_}:', current_value=x, max_value=(width - x_s), show_precent=True)

            imgs_result.append(copy.deepcopy(img_uncrop_))
    elif len(img_.shape) == 2:
        i_ = 0
        while i_ < len(img_list_):
            img_uncrop_ = np.zeros((height, width))
            for x in range(0, width - x_s, x_s):
                if x > (width - x_dim):  # If off end move back x_dim
                    x = width - x_dim
                for y in range(0, height - y_s, y_s):
                    if y > (height - y_dim):
                        y = height - y_dim

                    if i_ < len(img_list_):
                        img_uncrop_[y:y + y_dim, x:x + x_dim] = img_list_[i_]
                    i_ += 1
                progress_bar(freetext=f'Progress Image {i_}:', current_value=x, max_value=(width - x_s), show_precent=True)

            imgs_result.append(copy.deepcopy(img_uncrop_))
    return imgs_result


def progress_bar(freetext='Progress', current_value=0, max_value=100, show_precent=True):
    filled = int((current_value / max_value) * 30)
    empty = int(((max_value - current_value) / max_value) * 30)
    scrollbar = '|' + '\u2588' * filled + '-' * empty + '|'

    if current_value == max_value:
        end = True
    else:
        end = False

    if not end:
        if show_precent:
            percentage = '{:.2f}%'.format((current_value / max_value) * 100)
            print('\r', freetext, ':', scrollbar, percentage, end='', flush=True)
        else:
            print('\r', freetext, ':', scrollbar, end='', flush=True)
    else:
        if show_precent:
            percentage = '{:.2f}%'.format((current_value / max_value) * 100)
            print('\rProgress:', scrollbar, percentage, '\n', flush=True)
        else:
            print('\r', freetext, ':', scrollbar, '\n', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    print('Loading Model:', MODEL_NAME)
    best_model = torch.load(MODEL_NAME)

    if best_model:
        print('Done')
    else:
        print('Error loading model. Exiting...')
        exit()

    # create test dataset
    test_dataset = Dataset(
        x_test_dir,
        x_test_dir,  # y_test_dir,
        augmentation=get_validation_augmentation(),  # don't augment if not using model.predict
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # Predict on dataset
    img_list = []
    count = 0
    print('Predicting...')

    # Either load predictions or predict using existing trained .pth model.
    if LOAD_PREDICTIONS:
        try:
            img_list = pickle.load(open('temp_predictions.pickle', 'rb'))
        except FileNotFoundError:
            print('No saved predictions loaded.')
        if len(img_list) > 0:
            print('Predictions loaded from file.')
    else:  # run the prediction based on the supplied .pth model
        for image, gt_mask in test_dataset:
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

            progress_bar(freetext='Processing', current_value=count, max_value=len(test_dataset), show_precent=True)
            count += 1

            pr_mask = best_model.predict(x_tensor[:, :, :, :])
            pr_mask_cpu = pr_mask.cpu()
            img_list.append(from_tensor(pr_mask_cpu[0, :, :, :]))

        # uncomment to save predictions for quick tests (IE to change display without running through model)
        # predictions get too big to save everytime by default.
        # pickle.dump(img_list, open('temp_predictions.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        progress_bar(freetext='Processing', current_value=count, max_value=len(test_dataset), show_precent=True)
        print('Done')

    # use a rockUncropper to join the images together
    print('Uncropping...')
    imgs_result = img_uncrop_full(img_list,
                                  width=output_width,
                                  height=output_height,
                                  x_s=input_width // div_width,
                                  y_s=input_height // div_height)  # Returns an image with len(CLASSES) channels
    print('\nDone')

    # Stack the prediction into one colour image and display the results
    for j, img_ in enumerate(imgs_result):
        if OUTPUT_STACKED or DISPLAY_RESULTS_STACKED:
            stacked_image = stack_predictions(img_, rockDict_Lith, include=CLASSES, Use_TEDBaT=USE_TEDBaT)
        if DISPLAY_RESULTS_STACKED:
            plt.imshow(stacked_image)
            plt.show()
        if OUTPUT_STACKED:
            print('Writing Stacked Images To File:', '{:05d}'.format(j) + '.png', ', For Channels:', CLASSES)
            stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(result_output_dir + '_stacked_' + '{:05d}'.format(j) + '.png', stacked_image)

    # Output each class as a separate image
    for j, img_ in enumerate(imgs_result):
        for i, cls in enumerate(CLASSES):
            if OUTPUT_SINGLE:
                print('Writing Single Channel Images To File:', cls)
                cv2.imwrite(result_output_dir + '{:05d}'.format(j) + '_' + cls + '_' + '{:05d}'.format(i) +
                            '.png', (img_[:, :, i] * 255))
            if DISPLAY_RESULTS_SINGLE:
                plt.imshow(img_[:, :, i] * 255, vmin=0, vmax=1)
                plt.show()
