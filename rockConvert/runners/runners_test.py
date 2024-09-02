import runners as rs

import cv2
import numpy as np

########################################################################################################################
# Define the OpenCV and/or Numpy based functions here.
########################################################################################################################


def green_only_and_invert(img_):
    img_[:, :, 0] = 0
    img_[:, :, 1] = 255 - img_[:, :, 1]
    img_[:, :, 2] = 0

    # Remove the alpha layer
    if img_.shape[2] > 3:
        img_ = img_[:, :, 0:3]

    return img_


def gray(img_):
    img_[:, :, 0] = img_[:, :, 1]
    img_[:, :, 2] = img_[:, :, 1]
    return img_


def combine_green_gray_small(imgs):
    width = 500
    height = 250

    # Need to do the QC in function to allow rest of code to be generic
    if len(imgs) != 2:
        print("Number of input images does not match requirements. Exiting...")
        exit()

    for i, z in enumerate(imgs):
        imgs[i] = cv2.resize(imgs[i], (width, height))

    imgs[0][:, :, 0] = imgs[0][:, :, 0]  # Get the blank change from green only image
    imgs[0][:, :, 2] = imgs[1][:, :, 2]  # get the red channel from the gray image
                                         # don't change the blue channel

    # Remove the alpha layer
    if imgs[0].shape[2] > 3:
        imgs[0] = imgs[0][:, :, 0:3]

    return imgs[0]

# def qc_test(imgs_):
#     # imgs_ will be a list in the order
#     def high_green_precentage(img_, blank_percentage=0.5, threshold=250):
#         # Where green channel value is less than threshold, make 1, else make 0
#         # img_result = np.where(img_[:, :, 1] < threshold, 0, 1)
#         img_result = cv2.inRange(img_, np.array([0, 240, 0]), np.array([0, 255, 0]))
#         img_result = np.where(img_result == 255, 1, 0)
#
#         blank_count = np.count_nonzero(img_result)
#         if blank_count >= (img_result.shape[0] * img_result.shape[1] * blank_percentage):
#             high_green = True
#         else:
#             high_green = False
#
#         return high_green
#
#     def high_blank_percentage(img_, blank_percentage=0.5, threshold=5):
#         # Convert to gray
#         img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
#
#         # then where value is greater than threshold, make 1, else make 0
#         img_result = np.where(img_gray > threshold, 0, 1)
#
#         blank_count = np.count_nonzero(img_result)
#         if blank_count >= (img_result.shape[0] * img_result.shape[1] * blank_percentage):
#             high_blank = True
#         else:
#             high_blank = False
#
#         return high_blank
#
#     # TODO write code to do this test of the high green and high blank
#
#     # return imgs if they are to moved


########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################


data_directory = 'C:/Users/XXXXXXX/Documents/Data Beacon Hill with Interp/Med768x768_Arch/'

setup_d1p1d1 = dict(
    green_only_and_invert={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'X_all/',
        'output_dir': data_directory + 'tmp_green_only/',
        'function': green_only_and_invert,
        'input_name_tag': '.png',
        'output_name_tag': '.png',
    },
    gray={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'tmp_green_only/',
        'output_dir': data_directory + 'tmp_gray/',
        'function': gray,
        'input_name_tag': '.png',
        'output_name_tag': '.png',
    },
)

setup_dxp1d1 = dict(
    Combine_images_rgb={
        'display': False,
        'output': True,
        'input_dirs': [data_directory + 'tmp_green_only/',
                       data_directory + 'tmp_gray/'],
        'output_dir': data_directory + 'tmp_combined/',
        'function': combine_green_gray_small,
        'input_name_tag': '.png',
        'output_name_tag': '.png',
    }
)

# setup_dxpxdx = dict(
#     qc_move={
#         'display': False,
#         'output': True,
#         'input_dirs': [data_directory + 'ztmp_X_processed_croppped/',
#                        data_directory + 'ztmp_y_processed_croppped/'],
#         'output_dirs': [data_directory + 'ztmp_X_qc_failed/',
#                         data_directory + 'ztmp_y_qc_failed/'],
#         'function': qc_test,
#         'input_name_tag': '.png',
#         'output_name_tag': '.png',
#     },
# )



########################################################################################################################

if __name__ == '__main__':
    rs.img_d1p1d1(setup_d1p1d1)
    rs.img_dxp1d1(setup_dxp1d1)
    # rs.img_dxpxdx(setup_dxpxdx)
