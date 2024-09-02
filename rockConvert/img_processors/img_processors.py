import cv2
import numpy as np
from scipy import stats


# TODO move all the function over to a module and perhaps make it so
# arguments can be passed through

########################################################################################################################
# General Purpose Functions Used Again and Again
########################################################################################################################

def TEDBaT(img_src, normalise_input=False, thresh_start=0.2, erode_itr=2, dilate_itr=2, kernel_size=3,
           blur_kernel_size=9, thresh_end=0.6):
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

    return _img * 255


def split_interpretation_to_camvid_Lith_Org_Dataset_Four_Outcrops(img_, use_tedbat=True, for_visualisation=False):
    """ Updated to attempt Lith on First 4 Outcrops 22-11-2021:
        Ignore:
        Light Green     220,227,133
        Pink            229,  9,127
        Purple           73, 49,133
    """
    img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    # White - Off image - don't care
    index = 0
    mask = cv2.inRange(img_rgb, np.array([254, 254, 254]), np.array([255, 255, 255]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, 0)

    # Green - Grass (Vegetation)
    index = 1
    mask = cv2.inRange(img_rgb, np.array([0, 254, 0]), np.array([1, 255, 1]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Red Orange
    mask = cv2.inRange(img_rgb, np.array([220, 60, 28]), np.array([234, 88, 42]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Yellow
    mask = cv2.inRange(img_rgb, np.array([220, 188, 0]), np.array([255, 240, 64]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Dark Yellow
    mask = cv2.inRange(img_rgb, np.array([162, 157, 63]), np.array([166, 159, 67]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # Orange
    mask = cv2.inRange(img_rgb, np.array([220, 124, 28]), np.array([243, 147, 36]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Light Gray - 160,160,160 Note changed from Arch interp.
    mask = cv2.inRange(img_rgb, np.array([150, 150, 150]), np.array([185, 187, 181]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # # Medium Gray - 96, 96, 96
    mask = cv2.inRange(img_rgb, np.array([80, 80, 80]), np.array([100, 100, 100]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Dark Gray
    mask = cv2.inRange(img_rgb, np.array([50, 50, 50]), np.array([70, 70, 70]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Black - Coal
    mask = cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([39, 39, 39]))
    index = 4
    if use_tedbat == True: mask = TEDBaT(mask, erode_itr=1, dilate_itr=4)
    result = np.where(mask == 255, index, result)

    if for_visualisation and index < 25:
        result = result * 10

    return result


def split_interpretation_to_camvid_Lith_Org_Dataset(img_, use_tedbat=True, for_visualisation=False):
    img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    # White - Off image - don't care
    index = 0
    mask = cv2.inRange(img_rgb, np.array([254, 254, 254]), np.array([255, 255, 255]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, 0)

    # Green - Grass (Vegetation)
    index = 1
    mask = cv2.inRange(img_rgb, np.array([0, 254, 0]), np.array([1, 255, 1]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Red Orange - 224, 64, 32 - Correct
    mask = cv2.inRange(img_rgb, np.array([220, 60, 28]), np.array([230, 70, 34]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Yellow - 224, 192, 0 changed for original data set
    mask = cv2.inRange(img_rgb, np.array([220, 190, 0]), np.array([255, 240, 64]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # Orange - 224, 128, 32
    mask = cv2.inRange(img_rgb, np.array([220, 124, 28]), np.array([230, 134, 36]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Light Gray - 160,160,160 Note changed from Arch interp.
    mask = cv2.inRange(img_rgb, np.array([150, 150, 150]), np.array([170, 170, 170]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # # Medium Gray - 96, 96, 96
    mask = cv2.inRange(img_rgb, np.array([90, 90, 90]), np.array([100, 100, 100]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Dark Gray
    mask = cv2.inRange(img_rgb, np.array([50, 50, 50]), np.array([70, 70, 70]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Black - Coal
    mask = cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([39, 39, 39]))
    index = 4
    if use_tedbat == True: mask = TEDBaT(mask, erode_itr=1, dilate_itr=4)
    result = np.where(mask == 255, index, result)

    if for_visualisation and index < 25:
        result = result * 10

    return result


def split_interpretation_to_camvid_Lith_Lime32k_Dataset(img_, use_tedbat=True, for_visualisation=False):
    img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    # White - Off image - don't care
    index = 0
    mask = cv2.inRange(img_rgb, np.array([254, 254, 254]), np.array([255, 255, 255]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, 0)

    # Green - Grass (Vegetation)
    index = 1
    mask = cv2.inRange(img_rgb, np.array([0, 254, 0]), np.array([1, 255, 1]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Red Orange - Not Used
    mask = cv2.inRange(img_rgb, np.array([220, 60, 28]), np.array([230, 70, 34]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Yellow - 224, 192, 0 changed for original data set
    mask = cv2.inRange(img_rgb, np.array([220, 190, 0]), np.array([255, 240, 64]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # Orange - Not Used
    mask = cv2.inRange(img_rgb, np.array([220, 124, 28]), np.array([230, 134, 36]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Light Gray - 160,160,160 Note changed from Arch interp.
    mask = cv2.inRange(img_rgb, np.array([150, 150, 150]), np.array([170, 170, 170]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # # # Medium Gray - 137, 137, 137
    mask = cv2.inRange(img_rgb, np.array([120, 120, 120]), np.array([145, 145, 145]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Dark Gray
    mask = cv2.inRange(img_rgb, np.array([50, 50, 50]), np.array([70, 70, 70]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Black - Coal
    mask = cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([48, 48, 48]))
    index = 4
    if use_tedbat == True: mask = TEDBaT(mask, erode_itr=1, dilate_itr=4)
    result = np.where(mask == 255, index, result)

    if for_visualisation and index < 25:
        result = result * 10

    return result


def split_interpretation_to_camvid_Lith_Blender4k_Dataset(img_, use_tedbat=True, for_visualisation=False):
    img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    # White - Off image, not used
    index = 0
    mask = cv2.inRange(img_rgb, np.array([254, 254, 254]), np.array([255, 255, 255]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, 0)

    # Blue - The Lowest formation
    index = 1
    mask = cv2.inRange(img_rgb, np.array([0, 0, 220]), np.array([0, 0, 255]))
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Pink - Fine layers
    mask = cv2.inRange(img_rgb, np.array([220, 0, 220]), np.array([255, 0, 255]))
    index = 2
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # Yellow - Sandstone, larger blocks
    mask = cv2.inRange(img_rgb, np.array([220, 220, 0]), np.array([255, 240, 0]))
    index = 3
    if use_tedbat == True: mask = TEDBaT(mask)
    result = np.where(mask == 255, index, result)

    # # Black - Don't care or vegetation
    mask = cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([16, 16, 16]))
    index = 4
    if use_tedbat == True: mask = TEDBaT(mask, erode_itr=1, dilate_itr=4)
    result = np.where(mask == 255, index, result)

    if for_visualisation and index < 25:
        result = result * 10
    elif index > 25:
        raise Exception("Error in Converting to CamVid format: Index output of range.")

    return result

########################################################################################################################
# Define the OpenCV and/or Numpy based functions here.
########################################################################################################################
def mask_invert(img_):
    # Force the green channel onto the other channels
    img_[:, :, 0] = img_[:, :, 1]
    img_[:, :, 2] = img_[:, :, 1]

    # Invert colours
    img_ = cv2.bitwise_not(img_)

    return img_


def colour_invert(img_):
    # Invert colours
    img_[:, :, 0] = 255 - img_[:, :, 0]
    img_[:, :, 1] = 255 - img_[:, :, 1]
    img_[:, :, 2] = 255 - img_[:, :, 2]

    return img_


def mask_tedbat_resize(img_):
    width = 1999
    height = 999
    img_ = cv2.resize(img_, (width, height))

    img_ = TEDBaT(img_)

    return img_


def mask_tedbat_resize_720x360(img_):
    width = 720
    height = 360
    img_ = cv2.resize(img_, (width, height))

    img_ = TEDBaT(img_)

    return img_


def mask_tedbat_resize_32000x3666(img_):
    width = 32000
    height = 3666
    img_ = cv2.resize(img_, (width, height))

    img_ = TEDBaT(img_)
    return img_


def mask_tedbat_resize_32000x4000(img_):
    width = 32000
    height = 4000
    img_ = cv2.resize(img_, (width, height))

    img_ = TEDBaT(img_)
    return img_


def mask_tedbat_resize_32000x6666(img_):
    width = 32000
    height = 6666
    img_ = cv2.resize(img_, (width, height))

    img_ = TEDBaT(img_)
    return img_


def apply_mask_resize_1999x999(imgs_):
    # RBG or INT on img[0]
    # MSK on img[1]

    # the function that actually does the work
    imgs_[0] = imgs_[0][:, :, 0:3]  # Remove the alpha channel if it exists

    # # Resize the mask to match the img_
    # if imgs_[1].shape[0:2] != imgs_[0].shape[0:2]:
    #     print("Mask Resized to match image.")
    #     imgs_[1] = cv2.resize(imgs_[1], (imgs_[0].shape[1], imgs_[0].shape[0]))

    # Changed code: Always resize both to be 1999 x 999, simplier if slower
    imgs_[0] = cv2.resize(imgs_[0], (1999, 999))
    imgs_[1] = cv2.resize(imgs_[1], (1999, 999))

    # Accessed: 06/10/2021
    # https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
    # copy where we'll assign the new values
    output_img = np.copy(imgs_[0])
    # boolean indexing and assignment based on mask
    output_img[(imgs_[1] == 255).all(-1)] = [0, 255, 0]

    return output_img


def apply_mask_resize_32000x3666(imgs_):
    # RBG or INT on img[0]
    # MSK on img[1]

    # the function that actually does the work
    imgs_[0] = imgs_[0][:, :, 0:3]  # Remove the alpha channel if it exists

    # Changed code: Always resize both to be 32000 x 4000, simpler but slower
    imgs_[0] = cv2.resize(imgs_[0], (32000, 3666))
    imgs_[1] = cv2.resize(imgs_[1], (32000, 3666))

    # Accessed: 06/10/2021
    # https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
    # copy where we'll assign the new values
    output_img = np.copy(imgs_[0])

    # boolean indexing and assignment based on mask
    output_img[(imgs_[1] == 255).all(-1)] = [0, 255, 0]

    return output_img


def apply_mask_resize_32000x4000(imgs_):
    # RBG or INT on img[0]
    # MSK on img[1]

    # the function that actually does the work
    imgs_[0] = imgs_[0][:, :, 0:3]  # Remove the alpha channel if it exists

    # Changed code: Always resize both to be 32000 x 4000, simpler but slower
    imgs_[0] = cv2.resize(imgs_[0], (32000, 4000))
    imgs_[1] = cv2.resize(imgs_[1], (32000, 4000))

    # Accessed: 06/10/2021
    # https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
    # copy where we'll assign the new values
    output_img = np.copy(imgs_[0])

    # boolean indexing and assignment based on mask
    output_img[(imgs_[1] == 255).all(-1)] = [0, 255, 0]

    return output_img


def apply_mask_resize_32000x6666(imgs_):
    # RBG or INT on img[0]
    # MSK on img[1]

    # the function that actually does the work
    imgs_[0] = imgs_[0][:, :, 0:3]  # Remove the alpha channel if it exists

    # Changed code: Always resize both to be 32000 x 4000, simpler but slower
    imgs_[0] = cv2.resize(imgs_[0], (32000, 6666))
    imgs_[1] = cv2.resize(imgs_[1], (32000, 6666))

    # Accessed: 06/10/2021
    # https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
    # copy where we'll assign the new values
    output_img = np.copy(imgs_[0])

    # boolean indexing and assignment based on mask
    output_img[(imgs_[1] == 255).all(-1)] = [0, 255, 0]

    return output_img


def apply_mask_resize_720x360(imgs_):
    # RBG or INT on img[0]
    # MSK on img[1]

    # the function that actually does the work
    imgs_[0] = imgs_[0][:, :, 0:3]  # Remove the alpha channel if it exists

    # # Resize the mask to match the img_
    # if imgs_[1].shape[0:2] != imgs_[0].shape[0:2]:
    #     print("Mask Resized to match image.")
    #     imgs_[1] = cv2.resize(imgs_[1], (imgs_[0].shape[1], imgs_[0].shape[0]))

    # Changed code: Always resize both to be 1999 x 999, simplier if slower
    imgs_[0] = cv2.resize(imgs_[0], (720, 360))
    imgs_[1] = cv2.resize(imgs_[1], (720, 360))

    # Accessed: 06/10/2021
    # https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
    # copy where we'll assign the new values
    output_img = np.copy(imgs_[0])
    # boolean indexing and assignment based on mask
    output_img[(imgs_[1] == 255).all(-1)] = [0, 255, 0]

    return output_img


def resize_1999x999(img_):
    # Remove the alpha channel if it exists
    img_ = img_[:, :, 0:3]

    # Changed code: Always resize both to be 1999 x 999, simplier if slower
    img_ = cv2.resize(img_, (1999, 999))

    return img_


def resize_720x360(img_):
    # Remove the alpha channel if it exists
    img_ = img_[:, :, 0:3]

    # Changed code: Always resize both to be 1999 x 999, simplier if slower
    img_ = cv2.resize(img_, (720, 360))

    return img_


def convert_to_camvid_tedbat(img_):
    img_ = split_interpretation_to_camvid_Lith_Org_Dataset(img_, use_tedbat=True, for_visualisation=False)
    return img_


def convert_to_camvid_tedbat_for_vis(img_):
    img_ = split_interpretation_to_camvid_Lith_Org_Dataset(img_, use_tedbat=True, for_visualisation=True)
    return img_


def convert_Lime32k_to_camvid_tedbat(img_):
    img_ = split_interpretation_to_camvid_Lith_Lime32k_Dataset(img_, use_tedbat=True, for_visualisation=False)
    return img_


def convert_Lime32k_to_camvid_tedbat_for_vis(img_):
    img_ = split_interpretation_to_camvid_Lith_Lime32k_Dataset(img_, use_tedbat=True, for_visualisation=True)
    return img_


def convert_Blender4k_to_camvid_tedbat(img_):
    img_ = split_interpretation_to_camvid_Lith_Blender4k_Dataset(img_, use_tedbat=True, for_visualisation=False)
    return img_


def convert_Blender4k_to_camvid_tedbat_for_vis(img_):
    img_ = split_interpretation_to_camvid_Lith_Blender4k_Dataset(img_, use_tedbat=True, for_visualisation=True)
    return img_


def convert_to_camvid_white_black(img_):
    result = np.where(img_ == 255, 1, 0)
    return result


def crop_up_768x768_div2(img_):
    img_list = []

    width = img_.shape[1]   #1999
    height = img_.shape[0]  #999
    x_dim = 768
    y_dim = 768
    x_s = x_dim // 2
    y_s = y_dim // 2

    if len(img_.shape) == 3:
        for x in range(0, width - x_s, x_s):
            if x > (width - x_dim):  # If off end move back x_dim
                x = width - x_dim
            for y in range(0, height - y_s, y_s):
                if y > (height - y_dim):
                    y = height - y_dim
                crop = img_[y:y + y_dim, x:x + x_dim, :]
                img_list.append(crop)
    elif len(img_.shape) == 2:
        for x in range(0, width - x_s, x_s):
            if x > (width - x_dim):  # If off end move back x_dim
                x = width - x_dim
            for y in range(0, height - y_s, y_s):
                if y > (height - y_dim):
                    y = height - y_dim
                crop = img_[y:y + y_dim, x:x + x_dim]
                img_list.append(crop)

    return img_list


def crop_up_64x64_div2(img_):
    img_list = []

    width = img_.shape[1]
    height = img_.shape[0]
    x_dim = 64
    y_dim = 64
    x_s = x_dim // 2
    y_s = y_dim // 2

    if len(img_.shape) == 3:
        for x in range(0, width - x_s, x_s):
            if x > (width - x_dim):  # If off end move back x_dim
                x = width - x_dim
            for y in range(0, height - y_s, y_s):
                if y > (height - y_dim):
                    y = height - y_dim
                crop = img_[y:y + y_dim, x:x + x_dim, :]
                img_list.append(crop)
    elif len(img_.shape) == 2:
        for x in range(0, width - x_s, x_s):
            if x > (width - x_dim):  # If off end move back x_dim
                x = width - x_dim
            for y in range(0, height - y_s, y_s):
                if y > (height - y_dim):
                    y = height - y_dim
                crop = img_[y:y + y_dim, x:x + x_dim]
                img_list.append(crop)

    return img_list


def stack_simple_average(imgs_):
    result = np.zeros(imgs_[0].shape)
    for img in imgs_:
        result += img
    imgs_[0] = result / len(imgs_)
    return imgs_[0]


def gray_and_hp(img_):
    # Note raw image is BGR
    img_[:, :, 0] = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    sigma = 3
    # high pass code
    # Author: zoltron
    # From: https://stackoverflow.com/questions/50508452/implementing-photoshop-high-pass-filter-hpf-in-opencv
    # Accessed 03/11/2021
    img_[:, :, 0] = img_[:, :, 0] - cv2.GaussianBlur(img_[:, :, 0], (0, 0), sigma) + 127

    # Remove the alpha layer
    if img_.shape[2] > 3:
        img_ = img_[:, :, 0:3]

    return img_[:, :, 0]


def edge_detect_canny(img_):
    img_[:, :, 0] = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    sigma = 3
    low = 50
    high = 150
    img_[:, :, 0] = cv2.GaussianBlur(img_[:, :, 0], (5, 5), sigma)
    edges = cv2.Canny(img_, low, high)
    edges = TEDBaT(edges, erode_itr=0, dilate_itr=3, blur_kernel_size=5, thresh_end=0.5)
    edges = cv2.erode(edges, (3, 3), iterations=4)

    return edges
