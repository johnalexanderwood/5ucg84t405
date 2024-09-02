import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_d1p1d1(setup):
    """ Applies the processing dictionary configuration, setup, on a single folder
    outputs the results to a single folder """

    for key, value in setup.items():
        print('Working on:', key)

        # Create the output directory
        try:
            os.mkdir(value['output_dir'])
        except FileExistsError:
            print('Output directory already exists.')
        except FileNotFoundError:
            print('Error in data path. Is there a typo? Exiting...')
            exit()

        # Get all the files in the input directory
        try:
            all_files = [f for f in os.listdir(value['input_dir']) if
                         os.path.isfile(os.path.join(value['input_dir'], f))]
        except FileNotFoundError:
            print('The system cannot find the path specified.')

        # Find files with name_tag in the name
        img_filepaths = [i for i in all_files if value['input_name_tag'] in i]

        print('Found files:', len(img_filepaths))

        file_name_count = 0
        # loop through all the found files
        for i, img_path in enumerate(img_filepaths):
            # UNCHANGED = so we don't make gray camvid format interp into 3 channel gray images
            # TODO This might have to change to be RGB format from BGR
            img_raw = cv2.imread(value['input_dir'] + img_path, cv2.IMREAD_COLOR) # IMREAD_UNCHANGED keeps alpha

            if value['display']:
                plt.imshow(img_raw)
                plt.show()

            # Here apply the function as required - TODO make this selection in the setup dictionary
            img_processed = value['function'](img_raw)

            if value['display']:
                plt.imshow(img_processed)
                plt.show()

            if value['output']:
                # append name vs rename output file name
                # file_name = img_path.replace(value['input_name_tag'], '{:05d}'.format(i) + value['output_name_tag'])
                file_name = '{:06d}'.format(file_name_count) + value['output_name_tag']
                file_name_count += 1
                print('File processed, trying to write to disk:', file_name)

                try:
                    cv2.imwrite(value['output_dir'] + file_name, img_processed)
                except FileNotFoundError:
                    print('Error writing file. Does the output directory exist?')


def img_dxp1d1(setup):
    """ Applies the processing dictionary configuration, setup, on more than one folder
    and outputs the results to a single folder """

    for key, value in setup.items():
        print('Working on:', key)

        # Create the output directory
        try:
            os.mkdir(value['output_dir'])
        except FileExistsError:
            print('Output directory already exists.')
        except FileNotFoundError:
            print('Error in data path. Is there a typo? Exiting...')
            exit()

        # Get all the files in the input directory
        try:
            all_files = []
            for directory in value['input_dirs']:
                all_files.append([f for f in os.listdir(directory) if
                                  os.path.isfile(os.path.join(directory, f))])
        except FileNotFoundError:
            print('The system cannot find one of the paths specified.')

        img_filepaths = []
        for dir_files in all_files:  # Go through each directory one at a time
            # Find files with name_tag in the name
            img_filepaths.append([i for i in dir_files if value['input_name_tag'] in i])
            print('Found files:', len(img_filepaths[-1]))

        # Now check all three lists have the same number of files or this will not work
        for paths in img_filepaths:
            if len(paths) != len(img_filepaths[0]):
                print('Number of images in directories after checking name tag does not match. Exiting...')
                exit()

        file_name_count = 0
        # loop through all the found files at the same time - E.G. Three separate directories at once
        for i, img_path in enumerate(img_filepaths[0]):

            # We need to load in all the images from the separate directories here
            imgs_raw = []  # list of raw images
            for j, img_filepath in enumerate(img_filepaths):  # Should be 3 of
                # i is the index through all the directories
                # j in the index of input directories from the setup dict
                imgs_raw.append(cv2.imread(value['input_dirs'][j] + img_filepath[i], cv2.IMREAD_COLOR))

                if value['display']:
                    plt.imshow(imgs_raw[-1])
                    plt.show()

            # Here apply the function as required the list of images in img
            img_processed = value['function'](imgs_raw)  # img_raw is a list of images now

            if value['display']:
                plt.imshow(img_processed)
                plt.show()

            if value['output']:
                # append name vs rename output file name
                # file_name = img_path.replace(value['input_name_tag'], '{:05d}'.format(i) + value['output_name_tag'])
                file_name = '{:06d}'.format(file_name_count) + value['output_name_tag']
                file_name_count += 1
                print('File processed, trying to write to disk:', file_name)

                try:
                    cv2.imwrite(value['output_dir'] + file_name, img_processed)
                except FileNotFoundError:
                    print('Error writing file. Does the output directory exist?')


def img_d1pxd1(setup):
    """ Applies the processing dictionary configuration, setup, on a single folder
    and outputs the multifile results to a single folder """
    for key, value in setup.items():
        print('Working on:', key)

        # Create the output directory
        try:
            os.mkdir(value['output_dir'])
        except FileExistsError:
            print('Output directory already exists.')
        except FileNotFoundError:
            print('Error in data path. Is there a typo? Exiting...')
            exit()

        # Get all the files in the directory
        try:
            all_files = [f for f in os.listdir(value['input_dir']) if
                         os.path.isfile(os.path.join(value['input_dir'], f))]

        except FileNotFoundError:
            print('The system cannot find the path specified.')

        # Find files with name_tag in the name
        img_filepaths = [i for i in all_files if value['input_name_tag'] in i]

        print('Found files:', len(img_filepaths))

        file_name_count = 0
        # loop through all the found files
        for img_path in img_filepaths:
            # UNCHANGED = so we don't make gray camvid format interp into 3 channel gray images
            img_raw = cv2.imread(value['input_dir'] + img_path, cv2.IMREAD_COLOR)

            if value['display']:
                plt.imshow(img_raw)
                plt.show()

            img_int_list = value['function'](img_raw)

            if value['display']:
                for img in img_int_list:
                    plt.imshow(img)
                    plt.show()

            if value['output']:
                # Output the interp images
                for i, img in enumerate(img_int_list):
                    # append name vs rename output file name
                    # file_name = img_path.replace(value['input_name_tag'], '{:05d}'.format(i) + value['output_name_tag'])
                    file_name = '{:06d}'.format(file_name_count) + value['output_name_tag']
                    file_name_count += 1
                    print('Trying write to disk crop:', file_name)

                    try:
                        cv2.imwrite(value['output_dir'] + file_name, img)
                    except FileNotFoundError:
                        print('Error writing file. Does the output directory exist?')


def img_dxpxdx(setup):
    """ NOT TESTED / DEBUGGED: Applies the processing dictionary configuration, setup, on multiple folders
    and outputs the multi-file results to multiple folders """

    for key, value in setup.items():
        print('Working on:', key)

        # Create the output directory
        for output_directory in value['output_dirs']:
            try:
                os.mkdir(output_directory)
            except FileExistsError:
                print('Output directory already exists.')
            except FileNotFoundError:
                print('Error in data path. Is there a typo? Exiting...')
                exit()

        # Get all the files in the input directory
        try:
            all_files = []
            for directory in value['input_dirs']:
                all_files.append([f for f in os.listdir(directory) if
                                  os.path.isfile(os.path.join(directory, f))])
        except FileNotFoundError:
            print('The system cannot find one of the paths specified.')

        img_filepaths = []
        for dir_files in all_files:  # Go through each directory one at a time
            # Find files with name_tag in the name
            img_filepaths.append([i for i in dir_files if value['input_name_tag'] in i])
            print('Found files:', len(img_filepaths[-1]))

        # Now check all three lists have the same number of files or this will not work
        for paths in img_filepaths:
            if len(paths) != len(img_filepaths[0]):
                print('Number of images in directories after checking name tag does not match. Exiting...')
                exit()

        file_name_count = 0
        # loop through all the found files at the same time - E.G. Three separate directories at once
        for i, img_path in enumerate(img_filepaths[0]):

            # We need to load in all the images from the separate directories here
            imgs_raw = []  # list of raw images
            for j, img_filepath in enumerate(img_filepaths):  # Should be 3 of
                # i is the index through all the directories
                # j in the index of input directories from the setup dict
                imgs_raw.append(cv2.imread(value['input_dirs'][j] + img_filepath[i], cv2.IMREAD_COLOR))

                if value['display']:
                    plt.imshow(imgs_raw[-1])
                    plt.show()

            # Here apply the function as required the list of images in img
            img_int_list = value['function'](imgs_raw)

            if value['display']:
                for img in img_int_list:
                    plt.imshow(img)
                    plt.show()

            if value['output']:
                # TODO Adjust loop so each img_processed goes to it's own directory
                for output_directory, img_processed in zip(value['output_dirs'], [img_int_list]):
                    # name is just a number
                    file_name = '{:06d}'.format(file_name_count) + value['output_name_tag']
                    file_name_count += 1
                    print('File processed, trying to write to disk:', file_name)

                    try:
                        cv2.imwrite(output_directory + file_name, img_processed)
                    except FileNotFoundError:
                        print('Error writing file. Does the output directory exist?')

