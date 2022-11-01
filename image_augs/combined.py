import uuid
import cv2
from tornado import concurrent
from tqdm import tqdm
import numpy as np
from rich.console import Console
from rich.table import Table



import pickle
import yaml
from typing import List 
import os
import time

from image_augs import filereads_py as filereads_py
from image_augs import rotations as rotations 
from image_augs import utils_py as utils_py
from image_augs import augmentations as aug
from image_augs import logging_util as logging_util


'''
Combining all the functions together
'''

mark = '''# Augmentations..'''


#logging
logger = logging_util.get_logger(os.path.basename(__file__))

# https://rich.readthedocs.io/en/stable/introduction.html
console = Console()

#creating table using rich module
table = Table(title="[b]Data augmentations table[/b]",style='red')

table.add_column("Augmentation name", justify="right", style="cyan", no_wrap=True)
table.add_column("Augmented images / Total images", style="magenta")
table.add_column("Fraction used", justify="right", style="green")


def raw_image(coordinates:List[List],images:np.array,folder:os.path) -> None:

    '''
    This function will save RAW images and its coordinates

    : param coordinates : coordinates is a pickle file containing coordinates 
                          of a image. coordinates contain (image_id , class_id , x,y,w,h) 
                          (load the pickle file first)
    
    : param images      : images a pickle file containing key , value pair.
                          key id of image_id (0,1,2,..) , value is cv2.imread() image(np.array)

    : param folder      : folder name where you want to save your augmentations.

    : return            : None
    
    '''
    try:
        for datas in  coordinates:

                name = uuid.uuid4()
                image = images[datas[0][0]]
                for coor in range(len(datas)):
                    id  = int(datas[coor][1])
                    X  = int(datas[coor][2])
                    Y  = int(datas[coor][3])
                    W  = int(datas[coor][4])
                    H = int(datas[coor][5])
            
            # #     #normalize
                    coor = aug.normalize_yolo(image,X,Y,W,H)

        
                    with open(f'{folder}/labels/{name}.txt','a+') as f:
                        f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

                cv2.imwrite(f'{folder}/images/{name}.jpg',image)

    except Exception as e:
        logger.warning(f"Problem with raw images, {e}")

    

    

def blur_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None: 

    '''
    This function will add blur to a fraction of images, suppose you have 100 images and you want to add blur only
    on randomly selected 20 images, by passing fraction param = 0.2, it will randomly select 20 images from 
    dataset and apply blur to it, for selecting all images you have to pass fraction = 1.0 

    : param coordinates : coordinates is a pickle file containing coordinates 
                          of a image. coordinates contain (image_id , class_id , x,y,w,h) 
                          (load the pickle file first)
    
    : param images      : images a pickle file containing key , value pair.
                          key id of image_id (0,1,2,..) , value is cv2.imread() image(np.array)

    : param fraction    : fraction is float. by giving fraction =0.5 , it will randomly selected 50% data from your
                          dataset and apply blur to it.

    : param folder      : folder name where you want to save your augmentations.

    : return            : None
    
    '''
    
    try:
        console.print('[bold blue]Blur started..[bold blue]')
        #select fraction of data by passing fraction param , coordinates is loaded pickle file
        blur_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        #blur frc is list
        for datas in blur_frc:
            #getting the name
            name = uuid.uuid4()
            #getting the read image
            image = images[datas[0][0]]
            #apply blur to it
            blurs = aug.blur(image)
            #getting the id , x,y,w,h coordinates (raw coor)
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
 
                #normalizing those coordinates
                coor = aug.normalize_yolo(blurs,X,Y,W,H)

                #saving coor inside given folder
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            #saving image inside given folder
            cv2.imwrite(f'{folder}/images/{name}.jpg',blurs)

       

    except Exception as e:
        logger.warning(f"Problem with blur images, {e}")

    console.print('[bold green]Blur completed..[bold green]')


def noise_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Noise started..[bold blue]')
        noise_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        for datas in noise_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            noises = aug.noise(image)

            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
                coor = aug.normalize_yolo(noises,X,Y,W,H)
                
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            cv2.imwrite(f'{folder}/images/{name}.jpg',noises)

    except Exception as e:
        logger.warning(f"Problem with noise images, {e}")
        
    console.print('[bold green]Noise completed..[bold green]')
    



def noise_blur_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Noise and blur started..[bold blue]')
        NB_frc = filereads_py.fraction_data(coordinates,fraction=fraction)

        for datas in NB_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            nb = aug.noise_and_blur(image)

            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
                

                coor = aug.normalize_yolo(nb,X,Y,W,H)
                

                

                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            cv2.imwrite(f'{folder}/images/{name}.jpg',nb)

    except Exception as e:
        logger.warning(f"Problem with noise_and_blur images, {e}")

    console.print('[bold green]Noise and blur completed..[bold green]')
    

def hue_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Hue started..[bold blue]')
        hue_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        for datas in hue_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            hue = aug.hue(image)

            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
                

                coor = aug.normalize_yolo(hue,X,Y,W,H)
                

                

                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            cv2.imwrite(f'{folder}/images/{name}.jpg',hue)

    except Exception as e:
        logger.warning(f"Problem with hue images, {e}")
    
    console.print('[bold green]Hue completed..[bold green]')     
   


def saturation_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Saturation started..[bold blue]')
        saturation_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        for datas in saturation_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            sat = aug.image_saturation(image)

            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
                coor = aug.normalize_yolo(sat,X,Y,W,H)

                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            cv2.imwrite(f'{folder}/images/{name}.jpg',sat)

    except Exception as e:
        logger.warning(f"Problem with saturation images, {e}")

    console.print('[bold green]Saturation completed..[bold green]')

def contrast_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        contrast_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        console.print('[bold blue]Contrast started..[bold blue]')
        for datas in contrast_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            contrast = aug.contrast(image)

            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
                coor = aug.normalize_yolo(contrast,X,Y,W,H)
                
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

            cv2.imwrite(f'{folder}/images/{name}.jpg',contrast)

    except Exception as e:
        logger.warning(f"Problem with contrast images, {e}")

    console.print('[bold green]Contrast completed..[bold green]')


def bright_dark_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:

        brightness_darkness_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        console.print('[bold blue]Bright/dark started..[bold blue]')
        for datas in brightness_darkness_frc:
            name = uuid.uuid4()
            image = images[datas[0][0]]
            bright , dark = aug.brightness_contrast(image)
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
            
            
                coor = aug.normalize_yolo(bright,X,Y,W,H)
                

                with open(f'{folder}/labels/{name}_bright.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

                with open(f'{folder}/labels/{name}_dark.txt','a+') as f:
                    f.write(f'{id} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')
        

            cv2.imwrite(f'{folder}/images/{name}_bright.jpg',bright)
            cv2.imwrite(f'{folder}/images/{name}_dark.jpg',dark)

    except Exception as e:
        logger.warning(f"Problem with brightness_darkness images, {e}")

    console.print('[bold green]Bright/dark completed..[bold green]')



def rotate_90_im(coordinates:List[list],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        rotation_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        console.print('[bold blue]Rotation started..[bold blue]')
        
        for datas in rotation_frc:
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        

            rotated_image , coor , id = rotations.rotate_90deg(image,temp_coor,temp_id)

            #adding (hue/sat/contrast/blur to rotated image , if you want you can turn it off)
            rotated_image = aug.more_aug(rotated_image)
            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

    except Exception as e:
        logger.warning(f"Problem with rotation90 images, {e}")
    
    console.print('[bold green]Rotation completed..[bold green]')
    
       




def random_rotate_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        random_rotate_frc = filereads_py.fraction_data(coordinates,fraction=fraction)

        for datas in random_rotate_frc:
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        

            rotated_image , coor , id = rotations.random_rotation(image,temp_coor,temp_id)
            #adding (hue/sat/contrast/blur to rotated image , if you want you can turn it off)
            rotated_image = aug.more_aug(rotated_image)
            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

    except Exception as e:
        logger.warning(f"Problem with random rotation images, {e}")


    



def zoom_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Zoom started..[bold blue]')
        zoom_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
        for datas in zoom_frc:
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        
            rotated_image , coor , id = aug.clipped_zoom(image,temp_coor,temp_id)
            #adding (hue/sat/contrast/blur to rotated image , if you want you can turn it off)
            rotated_image = aug.more_aug(rotated_image)

            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')
    
    except Exception as e:
        logger.warning(f"Problem with zoom images, {e}")
        
    console.print('[bold green]Zoom completed..[bold green]')
    
   

def affine_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Affine started..[bold blue]')
        affine_frc = filereads_py.fraction_data(coordinates,fraction=fraction)

        
        for datas in tqdm(affine_frc,desc='Affine'):
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        

            rotated_image , coor , id = aug.affine_transform(image,temp_coor,temp_id)
        
            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])

                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

    except Exception as e:
        logger.warning(f"Problem with affine images, {e}")

    console.print('[bold green]Affine completed..[bold green]')
    
   



def translation_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Translation started..[bold blue]')
        translation_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
    
        for datas in translation_frc:
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        

            rotated_image , coor , id = aug.image_translation(image,temp_coor,temp_id)
            #adding (hue/sat/contrast/blur to rotated image , if you want you can turn it off)
            # rotated_image = aug.more_aug(rotated_image)
            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')
    except Exception as e:
        logger.warning(f"Problem with translation images, {e}")

    console.print('[bold green]Translation completed..[bold green]')
 




def vertical_im(coordinates:List[List],images:np.array,fraction:float,folder:os.path) -> None:

    '''
    For documentation and comments please, refer to blur_im function

    '''
    try:
        console.print('[bold blue]Vertical flip started..[bold blue]')
        translation_frc = filereads_py.fraction_data(coordinates,fraction=fraction)
    
        for datas in translation_frc:
            name = uuid.uuid4()
            temp_coor = []
            temp_id = []
            image = images[datas[0][0]]
            for coor in range(len(datas)):
                id  = int(datas[coor][1])
                X  = int(datas[coor][2])
                Y  = int(datas[coor][3])
                W  = int(datas[coor][4])
                H = int(datas[coor][5])
                temp_coor.append([X,Y,W,H])
                temp_id.append(id)

        

            rotated_image , coor , id = aug.vertical_flip(image,temp_coor,temp_id)
            #adding (hue/sat/contrast/blur to rotated image , if you want you can turn it off)
            rotated_image = aug.more_aug(rotated_image)
            cv2.imwrite(f'{folder}/images/{name}.jpg',rotated_image)

            for ids , coors in zip(id,coor):
                coor = aug.normalize_yolo(rotated_image,coors[0],coors[1],coors[2],coors[3])
                with open(f'{folder}/labels/{name}.txt','a+') as f:
                    f.write(f'{ids} {coor[0]} {coor[1]} {coor[2]} {coor[3]}\n')

    except Exception as e:
        logger.warning(f"Problem with vertical images, {e}")

    console.print('[bold green]Vertical flip completed..[bold green]')




def main(folder:os.path,train_test_split:float=0.10,raw_images_ok:bool=False,blurs:bool=False,blur_f:float=.6,noise:bool=False,noise_f:float=.6,NB:bool=False,NB_f:float=.6,hue:bool=False,hue_f:float=.6,sat:bool=False,sat_f:float=.6,bright:bool=False,bright_f:float=.6,
 contrast:bool=False,contrast_f:float=0.5,rotation:bool=False,rotation_f:float=0.8,zoom:bool=False,zoom_f:float=.7,affine:bool=False,affine_f:float=0.7,translation:bool=False,translation_f:float= 0.7,vertical_flip:bool=False,vertical_f:float=0.2):

    '''
    This main function combines all the sub functions. This function will load all the pickle files first,by giving
    a folder name , train test split , boolean and float values you will get your augmented data.

    : param folder                         : folder where you want to save your images, create it by folder creation function

    : param train_test_split               : it will divide your data into train test split, it accepts float values 

    : param raw_images_ok                  : It is bool , whether you want to save raw images or not.

    : param blur , noise ...,translation   : It is bool , which augmentation you want to use in your data.

    : param blur_f,zoom_f,..,translation_f : It is float , fraction of data you want to use for augmentations
    
    '''
    #type checking for inputs
    type_1 = {  'train_test_split':train_test_split,
                'blur_f':blur_f,
                'noise_f':noise_f,
                'NB_f':NB_f,
                'hue_f':hue_f,
                'sat_f':sat_f,
                'bright_f':bright_f,
                'contrast_f':contrast_f,
                'rotation_f':rotation_f,
                'zoom_f':zoom_f,
                'affine_f':affine_f,
                'translation_f':translation_f,
                'vertical_f':vertical_f}

    #need to add dictionary

    type_2 = {  'blurs':blurs,
                'noise':noise,
                'NB':NB,
                'hue':hue,
                'sat':sat,
                'bright':bright,
                'contrast':contrast,
                'rotation':rotation,
                'zoom':zoom,
                'affine':affine,
                'translation':translation,
                'vertical_flip':vertical_flip,
                'raw_images_ok':raw_images_ok}

    
    for types , val in type_1.items():
        if type(val) == float:
            pass
        else:
            raise TypeError(f'Expected type "float" for "{types}" param, got {type(val)} , check hyperparameters..')
    
    for typess , vals in type_2.items():

        if type(vals) == bool:

            pass

        else:
            raise TypeError(f'Expected type "bool" for "{typess}" param, got {type(vals)}, check hyperparameters..')
    

    try:
        #loading all the pickle files
        with open('pickle_files/coordinates_list.pickle','rb')  as ff:
            coor = pickle.load(ff)

        with open('pickle_files/images_dict.pickle','rb')  as f:
            imgs = pickle.load(f)

        with open('pickle_files/ids.pickle','rb')  as f:
            ids = pickle.load(f)

    except Exception as e:

        logger.error(f'Pickle file not found ! ,{e}')
        raise FileNotFoundError(f'Pickle file not Found!, {e}')
        

       
        
  
    start_time = time.perf_counter()
    console.print('[bold yellow]Augmentations :[/bold yellow] [b]:arrow_down:[/b]')

    

    #multiThread  #doc --> https://docs.python.org/3/library/concurrent.futures.html
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor: 

        if raw_images_ok:

            executor.submit(raw_image,coor,imgs,folder)
            table.add_row("Raw image", f'{str(int(len(imgs)))}/{len(imgs)}',"Default 1.0")

        if blurs:

            executor.submit(blur_im,coor,imgs,blur_f,folder)
            table.add_row("BLUR", f'{str(int(len(imgs)*blur_f))}/{len(imgs)}',str(blur_f))

        if noise:
            executor.submit(noise_im,coor,imgs,noise_f,folder)
            table.add_row("NOISE", f'{str(int(len(imgs)*noise_f))}/{len(imgs)}',str(noise_f))
          

        if NB:
           
            executor.submit(noise_blur_im,coor,imgs,NB_f,folder)
            table.add_row("NOISE AND BLUR", f'{str(int(len(imgs)*NB_f))}/{len(imgs)}',str(NB_f))

        if hue:
           
            executor.submit(hue_im,coor,imgs,hue_f,folder)
            table.add_row("HUE", f'{str(int(len(imgs)*hue_f))}/{len(imgs)}',str(hue_f))

        if sat:
      
            executor.submit(saturation_im,coor,imgs,sat_f,folder)
            table.add_row("SATURATION", f'{str(int(len(imgs)*sat_f))}/{len(imgs)}',str(sat_f))

        if bright:
          
            executor.submit(bright_dark_im,coor,imgs,bright_f,folder)
            table.add_row("Bright & dark", f'{str(int(len(imgs)*bright_f)*2)}/{len(imgs)*2}',str(bright_f)) 

        if rotation:
      
            executor.submit(rotate_90_im,coor,imgs,rotation_f,folder)
            
            executor.submit(random_rotate_im,coor,imgs,rotation_f,folder)
            table.add_row("Rotation 90deg", f'{str(int(len(imgs)*rotation_f))}/{len(imgs)}',str(rotation_f))
            table.add_row("Random rotation", f'{str(int(len(imgs)*rotation_f))}/{len(imgs)}',str(rotation_f))

        if zoom:
      
            executor.submit(zoom_im,coor,imgs,zoom_f,folder)
            table.add_row("Zoom", f'{str(int(len(imgs)*zoom_f))}/{len(imgs)}',str(zoom_f))

        if affine:
          
            executor.submit(affine_im,coor,imgs,affine_f,folder)
            table.add_row("Affine image",f'{str(int(len(imgs)*affine_f))}/{len(imgs)}',str(affine_f))
        
        if translation:
         
            executor.submit(translation_im,coor,imgs,translation_f,folder)
            table.add_row("Translation image",f'{str(int(len(imgs)*translation_f))}/{len(imgs)}',str(translation_f))
        
        if contrast:
            
            executor.submit(contrast_im,coor,imgs,contrast_f,folder)
            table.add_row("Contrast image",f'{str(int(len(imgs)*contrast_f))}/{len(imgs)}',str(contrast_f))

        if vertical_flip:
            executor.submit(vertical_im,coor,imgs,vertical_f,folder)
            table.add_row("VerticalFlip image",f'{str(int(len(imgs)*vertical_f))}/{len(imgs)}',str(vertical_f))


    console.print(f'[bold red]Time taken:[/bold red] [b]{time.perf_counter()-start_time:.2f}s[/b] :rocket:')

    with open(f'{folder}/classes.txt','a+') as classes:
        for _,value in ids.items():
            classes.write(f'{value}\n')
            

    #https://zetcode.com/python/yaml/
    names = list(ids.values())
    desc = [{
    'train': f'../train/images',
    'val': f'../valid/images',
    'nc': len(ids.keys()),
    'names': names}]

    with open(f'{folder}/data.yaml', 'w') as f:
        yaml.dump(desc,f)
       


    os.remove('pickle_files/coordinates_list.pickle')
    os.remove('pickle_files/images_dict.pickle')

    console.print(f'[bold green]Images and labels[/bold green] saved in [bold blue]{folder}[/bold blue] folder.')

    #it will split the data (train , test)
    utils_py.train_test_split(folder,split=train_test_split)

    if table.columns:
        console.print(table)


    return 'Completed'


























# if __name__ == "__main__":

    



# # # #     aug_image = dict()
# #     saved_folder_name = 'yolo_test'
#     # raw_images = raw_image_process('yolo_denormalize.txt',folder=saved_folder_name,normalize=False)
    
#     with open('pickle_files/coordinates_list.pickle','rb')  as ff:
#             coor = pickle.load(ff)


#     with open('pickle_files/images_dict.pickle','rb')  as f:
#             imgs = pickle.load(f)

#     with open('pickle_files/ids.pickle','rb')  as f:
#             ids = pickle.load(f)

#     bb = vertical_im(coor,imgs,.8,'yolo_test')

    # cc = noise_im(raw_images,0.9,saved_folder_name)
#     dd = noise_blur_im(ra,0.9,aug_image)
#     ee = hue_im(ra,0.9,aug_image)
#     f = saturation_im(ra,0.9,aug_image)
#     g = bright_dark_im(ra,0.9,aug_image)
#     g = rotate_90_im(ra,0.9,aug_image)
    # g = zoom_im(ra,0.4,aug_image)
    # g = affine_im(ra,0.4,aug_image)
    # g = translation_im(ra,0.9,aug_image)


    # for a,b in aug_image.items():
    #         print(a)