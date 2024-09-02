import os

# Bulk rename files in python by appending to original name
# This is needed to merge folders of images only named by number with number clashes

# Basis from here:
# https://www.geeksforgeeks.org/rename-multiple-files-using-python/
# Accessed: 20221126

########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################

# setup paths
data_directory = 'C:/Users/XXXXXX/Documents/Data 32k Kettleness/960x_110y_0p03mpx/'

# first filename character replacement
replacement = '2'

# Numebr of folds to change
# Not implemented yet, just does fold 0

########################################################################################################################

# Function to rename multiple files
def main():
    root_folder = data_directory + "autogen_dataset_75_25_fold_0/"
    sub_folders = ['test', 'testannot', 'train', 'trainannot', 'val', 'valannot']

    for folder in sub_folders:
        path = root_folder + folder
        print(f'### Working On Directory: {path}')
        for count, filename in enumerate(os.listdir(path)):
            src = f"{path}/{filename}"  # foldername/filename, if .py file is outside folder
            new_filename = replacement + filename[1:]
            dst = f"{path}/{new_filename}"

            print(f'Changing Files Names From: {src}  To: {dst}')

            # rename() function will rename all the files
            os.rename(src, dst)

    print('Done')

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()

