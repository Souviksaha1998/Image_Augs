from asyncio.log import logger
import random
import os
from typing import List


def read_txt(filename:os.path) -> os.path:
    '''
    This function will open txt file and shuffle the data

    '''
    lines = open(filename).readlines()
    random.shuffle(lines)
    return lines


def fraction_data(file:os.path,fraction:float=.9) -> List:

    '''
    This function will create a fraction of data,suppose you have 100 images and you want to add blur only
    on randomly selected 20 images, by passing fraction param = 0.2, it will randomly select 20 images from 
    dataset and apply blur to it, for selecting all images you have to pass fraction = 1.0 

    : param file             : it is a list
    : param fraction         : fraction float value
    : return random_sampling : It will return a list of cv2.imread() images(array)
    
    '''
    try:
        percentage = int(len(file)*fraction)
        random_sampling =random.sample(file,percentage)
        return random_sampling

    except Exception as e:
        logger.error(f'Fraction data creating issue, {e}')