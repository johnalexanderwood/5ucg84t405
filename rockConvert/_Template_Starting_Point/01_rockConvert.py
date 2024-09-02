import runners.runners as rs
import img_processors.img_processors as ip


########################################################################################################################
# Define input/output folders and order functions  will be applied using the setup dictionary
########################################################################################################################
data_directory = 'C:/Users/XXXXXX/Documents/Data 32k Cloughton Wyke/960x_110x_0p03mpp/'


# Other directories for ease of use in future
# 'C:/Users/XXXXXX/Documents/Data 32k Hayburn/960x_1X0y_0p03mpp/'
# 'C:/Users/XXXXXX/Documents/Data Ravenscar 32k/960x_200y_120z_0p03mpp/'
# 'C:/Users/XXXXXX/Documents/Data Kettleness 32k/960x_110y_120z_0p03mpp/'

# How to use:
#   1) Put the full size input (E.G. RGB) images into directory called X inside data directory. Make backgrounds white.
#   2) Put the full size interperation images into directory called y inside data directory. Make backgrounds white.
#   3) Run:
#       rockCovert                          (~8 minutes on 1.14GB of images),
#       rockConvert_QC                      (~2 minutes)
#       rockConvert_XFCV_camvid_folders     (~2 minutes)
#   4) Train system on pyTorch scripts:     (~4 or 5 minutes per Epoch, normally best model with 15 ep)
#   5) Predict on trained pyTorch Scripts   (~8 minutes to predict on all inputs cropped up, then mosaic predictions)
#   6) (TBC) Post process results with rockConvert_Post_Process...

# 20220404 - see other examples for how to do this.
#   TODO add a quantization step for interp images. Might improve things. Make sure it is an integer...

setup_phase10_d1p1d1 = dict(
    dip_invert_tif={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'raw_export_panels/',
        'output_dir': data_directory + 'X_inv/',
        'function': ip.colour_invert,
        'input_name_tag': '_Dip.tif',
        'output_name_tag': '_DipInv.tif',
    }
)

# setup_phase10_d1p1d1 = dict(
#     mask_invert_png={
#         'display': False,
#         'output': True,
#         'input_dir': data_directory + 'mask_non_geo/',
#         'output_dir': data_directory + 'ztmp_mask_png/',
#         'function': ip.mask_invert,
#         'input_name_tag': '.jpg',
#         'output_name_tag': '.png',
#     },
#     mask_tedbat_resize={
#         'display': False,
#         'output': True,
#         'input_dir': data_directory + 'ztmp_mask_png/',
#         'output_dir': data_directory + 'ztmp_mask_tedbat_resized/',
#         'function': ip.mask_tedbat_resize,
#         'input_name_tag': '.png',
#         'output_name_tag': '.png',
#     }
# )

# setup_phase20_dxp1d1 = dict(
#     mask_X_resize={
#         'display': False,
#         'output': True,
#         'input_dirs': [data_directory + 'X/',
#                        data_directory + 'ztmp_mask_tedbat_resized/'],
#         'output_dir': data_directory + 'ztmp_X_processed/',
#         'function': ip.apply_mask_resize_1999x999,
#         'input_name_tag': '.png',
#         'output_name_tag': '.png',
#     },
#     mask_y_resize={
#         'display': False,
#         'output': True,
#         'input_dirs': [data_directory + 'y/',
#                        data_directory + 'ztmp_mask_tedbat_resized/'],
#         'output_dir': data_directory + 'ztmp_y_masked/',
#         'function': ip.apply_mask_resize_1999x999,
#         'input_name_tag': '.png',
#         'output_name_tag': '.png',
#     },
# )

setup_phase30_d1p1p1 = dict(
    convert_to_camvid={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'y/',
        'output_dir': data_directory + 'ztmp_y_processed/',
        'function': ip.convert_Lime32k_to_camvid_tedbat,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    convert_to_camvid_for_vis={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'y/',
        'output_dir': data_directory + 'ztmp_y_camvid_vis/',
        'function': ip.convert_Lime32k_to_camvid_tedbat_for_vis,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
)

setup_phase40_d1pxd1 = dict(
    crop_x={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'X/',
        'output_dir': data_directory + 'ztmp_X_processed_cropped/',
        'function': ip.crop_up_768x768_div2,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    crop_x_test={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'X/',
        'output_dir': data_directory + 'ztmp_X_all_for_test/',
        'function': ip.crop_up_768x768_div2,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    crop_x_test_no_mask={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'X/',
        'output_dir': data_directory + 'ztmp_X_all_for_test_no_mask/',
        'function': ip.crop_up_768x768_div2,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    crop_y={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'ztmp_y_processed/',
        'output_dir': data_directory + 'ztmp_y_processed_cropped/',
        'function': ip.crop_up_768x768_div2,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    },
    crop_y_vis={
        'display': False,
        'output': True,
        'input_dir': data_directory + 'ztmp_y_camvid_vis/',
        'output_dir': data_directory + 'ztmp_y_processed_cropped_vis/',
        'function': ip.crop_up_768x768_div2,
        'input_name_tag': '.tif',
        'output_name_tag': '.tif',
    }
)

########################################################################################################################

if __name__ == '__main__':
    rs.img_d1p1d1(setup_phase10_d1p1d1)
    rs.img_d1p1d1(setup_phase30_d1p1p1)
    rs.img_d1pxd1(setup_phase40_d1pxd1)

