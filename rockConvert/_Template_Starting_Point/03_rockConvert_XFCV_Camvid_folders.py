# Input is path to:
#   Raw images
#   Labels / Masks / Annotations in
# in the camvid format

# Update to allow for XFCV
# Test set will be same for all folds
# train and validation will be stratified, IE with 2 folds and 75% 25% Train vs Test
# - autogen_dataset_fold_0
#   | train_0
#   | val_1
#   | test_global
# - autogen_dataset_fold_1
#   | train_1
#   | val_0
#   | test_global
# etc etc

#
# To use:   All the rgb images should be in one folder
#           All the interp images should be in another folder
#

import os
import shutil

import numpy as np
import sklearn.model_selection

########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################

# setup paths
data_directory = 'C:/Users/XXXXXX/Documents/Data 32k Hayburn/960x_1X0y_0p03mpp/'

# Other directories for ease of use in future
# 'C:/Users/XXXXXX/Documents/Data Ravenscar 32k/960x_200y_120z_0p03mpp/'
# 'C:/Users/XXXXXX/Documents/Data Kettleness 32k/960x_110y_120z_0p03mpp/'

# filters for file names
name_tag_image = '.tif'
name_tag_annot = '.tif'

# setup splits - should add up to 1.0
train_val_split = 0.75
test_split = 0.25

# XFCV
number_of_splits = 5

###############################################################################################################
input_image_path = data_directory + 'ztmp_X_processed_cropped/'
input_annot_path = data_directory + 'ztmp_y_processed_cropped/'

k_folds = sklearn.model_selection.KFold(n_splits=number_of_splits)

# Generate the numbers for the splits
input_image_files = [f for f in os.listdir(input_image_path) if os.path.isfile(os.path.join(input_image_path, f))]
input_image_filepaths = [i for i in input_image_files if name_tag_image in i]  # filter out files without name_tag

input_annot_files = [f for f in os.listdir(input_annot_path) if os.path.isfile(os.path.join(input_annot_path, f))]
input_annot_filepaths = [i for i in input_annot_files if name_tag_annot in i]  # filter out files without name_tag

# # Generate the splits as numbers - might be off by 3 files at most, not a problem for large datasets
# print('Number of RGB Images:', len(input_image_filepaths))
# print('Number of Annotation, Mask, Interp Images:', len(input_annot_filepaths))
#
if len(input_image_filepaths) == 0:
    print('Error, input directories has zero files with name tag. Exiting...')
    exit()

if not (len(input_image_filepaths) == len(input_annot_filepaths)):
    print('Error, input directories do not have the same number of files. Exiting...')
    exit()

train_val_file_count = int(len(input_image_files) * train_val_split)
test_file_count = int(len(input_image_files) * test_split)

# Only XFCV on the train and validation portion of the data
range_ = np.arange(train_val_file_count)
for fold_number, (train, val) in enumerate(k_folds.split(range_)):
    train_val_split_int = int(train_val_split * 100)
    test_split_int = int(test_split * 100)

    output_path = data_directory + 'autogen_dataset_' \
                  + str(train_val_split_int) + '_' \
                  + str(test_split_int) + '_' \
                  + 'fold_' + str(fold_number) + '/'

    # Generate folders
    try:
        os.mkdir(output_path)
    except FileExistsError:
        print('Autogen dataset directory already exists:', output_path)
    except FileNotFoundError:
        print('Error in data path. Is there a typo? Exiting...')
        exit()

    try:
        os.mkdir(output_path + 'train')
        os.mkdir(output_path + 'trainannot')
        os.mkdir(output_path + 'test')
        os.mkdir(output_path + 'testannot')
        os.mkdir(output_path + 'val')
        os.mkdir(output_path + 'valannot')
    except FileExistsError:
        print('Directory already exists, skipping directory creation.')
    except FileNotFoundError:
        print('Error in data path. Is there a typo? Exiting...')
        exit()

    print('train', len(train), 'val', len(val))

    # Move files - images
    for i in train:
        print('Copy to train folders:', i)
        shutil.copy2(input_image_path + input_image_filepaths[i],
                     output_path + 'train/' + input_image_filepaths[i])
        shutil.copy2(input_annot_path + input_annot_filepaths[i],
                     output_path + 'trainannot/' + input_image_filepaths[i])

    for i in val:
        print('Copy to validation folders:', i)
        shutil.copy2(input_image_path + input_image_filepaths[i],
                     output_path + 'val/' + input_image_filepaths[i])
        shutil.copy2(input_annot_path + input_annot_filepaths[i],
                     output_path + 'valannot/' + input_image_filepaths[i])

    # Testing is seperate from XFCV, all folds get same test data
    for i in range(len(input_image_files) - test_file_count, len(input_image_files)):
        print('Copy to test folders:', i)
        shutil.copy2(input_image_path + input_image_filepaths[i],
                     output_path + 'test/' + input_image_filepaths[i])
        shutil.copy2(input_annot_path + input_annot_filepaths[i],
                     output_path + 'testannot/' + input_image_filepaths[i])

