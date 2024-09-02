import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# This will:
# Given a directory with RGB images
# Move 'bad images' and the associated interp / annot to seperate folders
#

########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################

display = False  # True
output = True
with_interp = True

test_blank_percentage = True  # If more than X percent blank pixels remove image from dataset
test_white_percentage = True  # If more than X percent white pixels remove image from dataset
test_green_percentage = True  # If more than X percent green pixels remove image from dataset

BLANK_PERCENTAGE = 0.50
WHITE_PERCENTAGE = 0.50
GREEN_PERCENTAGE = 0.50  # Applies to regions with green channel above 250

# setup paths
data_directory = 'C:/Users/XXXXXX/Documents/Data 32k Cloughton Wyke/960x_110x_0p03mpp/'

# Other directories for ease of use in future
# 'C:/Users/XXXXXX/Documents/Data 32k Hayburn/960x_1X0y_0p03mpp/'
# 'C:/Users/XXXXXX/Documents/Data Ravenscar 32k/960x_200y_120z_0p03mpp/'
# 'C:/Users/XXXXXX/Documents/Data Kettleness 32k/960x_110y_120z_0p03mpp/'

rgb_name_tag = '.tif'
int_name_tag = '.tif'

##############################################################################
input_rgb_directory_path = data_directory + 'ztmp_X_processed_cropped/'
input_int_directory_path = data_directory + 'ztmp_y_processed_cropped//'


output_rgb_directory_path = data_directory + 'ztmp_X_poor_quality/'
output_int_directory_path = data_directory + 'ztmp_y_poor_quality/'


# Get all the rgb files in the directory
img_files = [f for f in os.listdir(input_rgb_directory_path) if
             os.path.isfile(os.path.join(input_rgb_directory_path, f))]

# Find files with name_tag in the name
rgb_filepaths = [i for i in img_files if rgb_name_tag in i]

# Get all the interp / annot files in the directory
int_files = [f for f in os.listdir(input_int_directory_path) if
             os.path.isfile(os.path.join(input_int_directory_path, f))]

# Find files with name_tag in the name
int_filepaths = [i for i in int_files if int_name_tag in i]

# Make the directories to move bad images too
try:
    os.mkdir(output_rgb_directory_path)
except FileExistsError:
    print('RGB Poor Quality directory already exists.')

try:
    os.mkdir(output_int_directory_path)
except FileExistsError:
    print('Int Poor Quality directory already exists.')

def high_green_precentage(img_, blank_percentage=0.5, threshold=250):
    # Where green channel value is less than threshold, make 1, else make 0
    # img_result = np.where(img_[:, :, 1] < threshold, 0, 1)
    img_result = cv2.inRange(img_rgb, np.array([0, 240, 0]), np.array([0, 255, 0]))
    img_result = np.where(img_result == 255, 1, 0)

    blank_count = np.count_nonzero(img_result)
    if blank_count >= (img_result.shape[0] * img_result.shape[1] * blank_percentage):
        high_green = True
    else:
        high_green = False

    return high_green

def high_blank_percentage(img_, blank_percentage=0.5, threshold=5):
    # Convert to gray
    img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    # then where value is greater than threshold, make 1, else make 0
    img_result = np.where(img_gray > threshold, 0, 1)

    blank_count = np.count_nonzero(img_result)
    if blank_count >= (img_result.shape[0] * img_result.shape[1] * blank_percentage):
        high_blank = True
    else:
        high_blank = False

    return high_blank

def high_white_percentage(img_, white_percentage=0.5, threshold=253):
    # Convert to gray
    img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    # then where value is greater than threshold, make 1, else make 0
    img_result = np.where(img_gray < threshold, 0, 1)

    blank_count = np.count_nonzero(img_result)
    if blank_count >= (img_result.shape[0] * img_result.shape[1] * white_percentage):
        high_white = True
    else:
        high_white = False

    return high_white

moved_file_count = 0
# loop through all the files
for rgb_file, int_file in zip(rgb_filepaths, int_filepaths):
    img_rgb = cv2.imread(input_rgb_directory_path + rgb_file, cv2.IMREAD_COLOR)

    if display:
        plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        plt.show()

    # TODO make the results 'or'd' together...
    remove_file = False

    if test_blank_percentage:
        remove_file = remove_file or high_blank_percentage(img_rgb, blank_percentage=BLANK_PERCENTAGE)

    if test_white_percentage:
        remove_file = remove_file or high_white_percentage(img_rgb, white_percentage=WHITE_PERCENTAGE, threshold=253)

    if test_green_percentage:
        remove_file = remove_file or high_green_precentage(img_rgb, blank_percentage=0.5, threshold=250)

    print("Checking File:", rgb_file)

    if output and remove_file:
        # So if the RGB input image has failed the test, remove both RGB and Interp image from Dataset
        shutil.move(input_rgb_directory_path + rgb_file, output_rgb_directory_path + rgb_file)
        shutil.move(input_int_directory_path + int_file, output_int_directory_path + int_file)

        moved_file_count += 1
        print('QC failed, file Moved:', rgb_file, int_file)

print('Moved File Count:', moved_file_count, '/', len(rgb_filepaths),
      '(Percentage Moved: ', '{:.2f}'.format((moved_file_count/len(rgb_filepaths))*100), '%)')
