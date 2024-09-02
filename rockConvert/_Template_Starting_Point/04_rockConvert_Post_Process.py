import runners.runners as rs
import img_processors.img_processors as ip

########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################
data_directory = 'C:/Users/XXXXXX/Documents/Data Kettleness 32k/960x_110y_120z_0p03mpp/'

# Other directories for ease of use in future
# 'C:/Users/XXXXXX/Documents/Data Ravenscar 32k/960x_200y_120z_0p03mpp/'

setup_phase1_dxp1d1 = dict(
    # stack_simple_average_masked={
    #     'display': False,
    #     'output': True,
    #     'input_dirs': [data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_0_TeZpZZ.pth/',
    #                    data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_1_TeZpZZ.pth/',
    #                    data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_2_TeZpZZ.pth/',
    #                    data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_3_TeZpZZ.pth/',
    #                    data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_4_TeZpZZ.pth/'],
    #     'output_dir': data_directory + 'y_pred_stacked/',
    #     'function': ip.stack_simple_average,
    #     'input_name_tag': '.tif',
    #     'output_name_tag': '.tif',
    # },
    stack_simple_average_not_masked={
        'display': False,
        'output': True,
        'input_dirs': [data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_0_TeZpZZ.pth/',
                       data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_1_TeZpZZ.pth/',
                       data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_2_TeZpZZ.pth/',
                       data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_3_TeZpZZ.pth/',
                       data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_4_TeZpZZ.pth/'],
        'output_dir': data_directory + 'y_pred_stacked_not_masked/',
        'function': ip.stack_simple_average,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    # stack_simple_average_all={
    #     'display': False,
    #     'output': True,
    #     'input_dirs': [
    #         data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_0_TeZpZZ.pth/',
    #         data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_1_TeZpZZ.pth/',
    #         data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_2_TeZpZZ.pth/',
    #         data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_3_TeZpZZ.pth/',
    #         data_directory + 'y_pred_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_4_TeZpZZ.pth/',
    #         data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_0_TeZpZZ.pth/',
    #         data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_1_TeZpZZ.pth/',
    #         data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_2_TeZpZZ.pth/',
    #         data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_3_TeZpZZ.pth/',
    #         data_directory + 'y_pred_non_masked_efficientnet-b6_Unet_20211109_Med768x768_Lith_fold_4_TeZpZZ.pth/'],
    #     'output_dir': data_directory + 'y_pred_stacked_all/',
    #     'function': ip.stack_simple_average,
    #     'input_name_tag': '.tif',
    #     'output_name_tag': '.tif',
    # },
)

########################################################################################################################

if __name__ == '__main__':
    rs.img_dxp1d1(setup_phase1_dxp1d1)
