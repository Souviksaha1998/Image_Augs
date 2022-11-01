import argparse
import time

from image_augs.combined import *
from image_augs.converter_for_txtToYolo import converter
from image_augs import utils_py as utils_py
from image_augs import config_loads as config_loads


'''
**RUN THIS FILE IF YOU CLONE THE REPO FROM GITHUB**

'''

def run_(config_file_path):

    '''
    This function combines all the folder.
    
    
    '''
    base_config = config_loads.read_config(config_file_path)
   

    saved_folder_name =  utils_py.folder_creation(base_config['destination_folder']['destination_folder_name']) 

    output = converter(path =  base_config['source_folder']['source_folder_name'],
                       keep_aspect_ratio= base_config.getboolean('image_config','keep_aspect_ratio'),
                       resize_im=base_config.getint('image_config','resize_image'),image_jpg_converter=True)

    
    start_time = time.perf_counter()

    dicc = main(folder=saved_folder_name,

    raw_images_ok= base_config.getboolean('augmentations','raw_images_ok'),
    train_test_split= base_config.getfloat('data_split','train_test_split'),
    blurs= base_config.getboolean('augmentations','blurs') ,blur_f=base_config.getfloat('percentage','blur_f'),
    noise= base_config.getboolean('augmentations','noise'),noise_f=base_config.getfloat('percentage','noise_f'),
    NB=base_config.getboolean('augmentations','noise_and_blur'),NB_f=base_config.getfloat('percentage','noise_and_blur_f'),
    hue=base_config.getboolean('augmentations','hue'),hue_f=base_config.getfloat('percentage','hue_f'),
    sat=base_config.getboolean('augmentations','sat'),sat_f=base_config.getfloat('percentage','sat_f'),
    bright=base_config.getboolean('augmentations','brightness_darkness'),bright_f=base_config.getfloat('percentage','brightness_darkness_f'),
    contrast=base_config.getboolean('augmentations','contrast') , contrast_f=base_config.getfloat('percentage','contrast_f'),
    rotation=base_config.getboolean('augmentations','rotation'),rotation_f=base_config.getfloat('percentage','rotation_f'),
    zoom=base_config.getboolean('augmentations','zoom'),zoom_f=base_config.getfloat('percentage','zoom_f'),
    affine=base_config.getboolean('augmentations','affine'),affine_f=base_config.getfloat('percentage','affine_f'),
    translation=base_config.getboolean('augmentations','translation'),translation_f= base_config.getfloat('percentage','translation_f'),
    vertical_flip=base_config.getboolean('augmentations','vertical_flip'),vertical_f= base_config.getfloat('percentage','vertical_flip_f'))

    





parser = argparse.ArgumentParser(description='Augmentations')
parser.add_argument('--configFile', type=str, nargs='?',help='give your config file path ie parameters.ini',required=True,metavar='.ini file',)
args = vars(parser.parse_args())




if __name__ == '__main__':

    run_(args['configFile'])


            

