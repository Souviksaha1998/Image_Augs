import glob
import pybboxes as pbx
import albumentations as A
import cv2
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.traceback import install

import pickle
import os
from typing import List , Tuple 

from image_augs import utils_py as utils_py
from image_augs import logging_util as logging_util


#logging
logger = logging_util.get_logger(os.path.basename(__file__))

#rich module
console = Console()
install()


#not in use 
def keeping_border(image:np.array,bboxes:List[List],category_ids:List,image_size:int=416) -> Tuple[np.array,List[List],List]:
    
    '''
    This function will keep border on each side of the image

    : param coordinates : coordinates is a pickle file containing coordinates 
                          of a image. coordinates contain (image_id , class_id , x,y,w,h) 
                          (load the pickle file first)
    
    : param images      : images a pickle file containing key , value pair.
                          key id of image_id (0,1,2,..) , value is cv2.imread() image(np.array)

    : param fraction    : fraction is float. by giving fraction =0.5 , it will randomly selected 50% data from your
                          dataset and apply blur to it.

    : param folder      : folder name where you want to save your augmentations.

    : return            : transfomed image , boxes coor and their ids

    '''
    transform = A.Compose(
    [A.augmentations.geometric.transforms.PadIfNeeded (p=1,min_height=image_size,min_width=image_size,border_mode=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    return transformed['image'] , transformed['bboxes'] , transformed['category_ids']


def jpg_converter(folder:os.path) -> None:
    
    '''
    This is a important function of this code.
    if all images are not in the same format it will convert all images into jpg.

    : param folder : It will accept a folder containing annotation , images and classes.
    
    '''
  
    try:

        for images in os.listdir(folder):
            if images.endswith('.txt'):
                continue
            if images.endswith('.jpg'):
                continue
            else:
                name = images.split('.')[0]
                im = cv2.imread(f'{folder}/{images}')
                cv2.imwrite(f'{folder}/{name}.jpg',im)
                os.remove(f'{folder}/{images}')
       

    except Exception as e:

        logger.warning(f'Jpg converter folder not found or does not contain images! {e}')

        raise FileNotFoundError(f'Please provide correct folder or full path of the  folder, Provided folder: {folder}')
        
    
    
def normalize_yolo(X_tl:int,Y_tl:int,b_width:int,b_height:int,image_width:int,image_height:int) -> Tuple[float,float,float,float]:

    '''
    This function will normalize coordinates values (200,100,...,) --> (0.002,0.444,...,)

    : param image_width , image_height  : actual image height and width
    : param X_tl , Y_tl                 : top left corner
    : param b_width , b_height          : height and width of the bounding boxes

    '''

    X_tl = (X_tl+(b_width/2))/image_width
    Y_tl = (Y_tl+(b_height/2))/image_height
    b_width = b_width/image_width
    b_height = b_height/image_height

    return X_tl , Y_tl , b_width , b_height



def converter(path:os.path,resize_im:int=416,keep_aspect_ratio:bool=False,image_jpg_converter:bool=True) -> None:
    
    '''
    Converter function will convert all images into given size and save those  -> images , coordination and class_ids in pickle files.
   

    : param path              : path is your folder containing images ,annotation and classes.
    : param resize_im         : resize im is an int, it will resize images into same size
    : param keep aspect ratio : It is bool , keeping it TRUE means it will keep one side image size same. (either height and width)
    : param image_jpg_conv    : It is bool, if your images contain different extension like .PNG , .JPG , .jpeg , keeping this True means
                                it will convert all images in .jpg format. 
    
    '''
    
    remember_id = dict()
    temp_store_list = []
    temp_image = dict()
    new = []

    
    
    if type(resize_im) == int:
            pass
    else:
        raise TypeError('Please provide resize im as INT!')
    
    if type(keep_aspect_ratio) and type(image_jpg_converter) == bool:
        pass
    else:
        raise TypeError('Please provide boolean in keep asepect ratio and image_jpg_converter')
    
    try:
        os.listdir(path)
        
    except:
        raise NotADirectoryError(f'{path} ...is not a directory!!')

    
    #need to add exception block
    if image_jpg_converter:
        jpg_converter(path)

    image = 0
    label = 0  
    checks = os.listdir(path)

    if checks.count('classes.txt') == 1:
        console.print('[bold green]Classes.txt found..[/bold green]')

    elif checks.count('classes.txt') > 1:
        raise Warning(f'More than "one" Classes.txt found !!')

    else:
          raise Warning(f'Classes.txt not found !!')


    for dataz in checks:
        if dataz.endswith('.jpg'):
            image += 1
        elif dataz.endswith('classes.txt'):
            continue
        elif dataz.endswith('.txt'):
            label += 1

    if (image==label) == True:

        console.print(f'[bold red]{image}[/bold red] [bold green]Images[/bold green] [b]and[/b] [bold red]{label}[/bold red] [bold green]Labels[/bold green] [b]found ![/b]')
    else:
        raise Warning(f'{image} Images and {label} Labels found!! Make sure you have same number of images and labels.')
   
    try:
        
        for paths in checks:
            if paths.endswith('classes.txt'):
        
                with open(f'{path}/{paths}')  as fff:
                    name = fff.read().splitlines()
                    console.print(f'[bold blue]Labels Name :[/bold blue] [b]{",".join(name)}[/b]')
                    for id, names in enumerate(name):
                        remember_id[id] = names
                        

    except Exception as e:
        
        logger.error('Classes.txt not found!')
        raise FileNotFoundError(f'Please provide full folder path,{e}')
    
    for i , data in enumerate(track(glob.glob(f'{path}/*txt'),description='Creating pickle file')):

        

        try:
       
            image = data.split('.')
            
        
            if keep_aspect_ratio:

                try:
                    if image[0].endswith('classes'):
                        continue

                    read_image = utils_py.image_resize(f'{image[0]}.jpg',image_size=resize_im)

                    temp_image[i] = read_image

                    

                    if read_image is None:
                        continue
        
                    h,w,c = read_image.shape

                    with open(f'{data}','r')  as ff:

                        split= ff.read().splitlines()

                        len_split = len(split)

                        for z , s in enumerate(range(len(split))):
                        
                            res = split[s]
                            
                            id = res.split()[0]# --> class id

                            coor = tuple(map(float,res.split()[1:])) # --> coordinates

                            
                            x_tl , y_tl , width , height = pbx.convert_bbox(coor, from_type="yolo", to_type="coco",image_size=(w,h)) 
                            
                            temp_store_list.insert(0,(i,id,x_tl,y_tl,width,height)) #need to write doc

                        
                        new.append(temp_store_list[0:len_split]) ###

                except Exception as e:
                    logger.warning(f'Exact data is not present in the folder, {e}')
                    raise Warning(f'data is missing in {path} ')


           
                  
                
        except Exception as e:
            logger.warning('Conveter can not convert all data.')
            raise FileNotFoundError(f'Please provide full path of {path}')

    if not os.path.exists('pickle_files'):
        os.makedirs('pickle_files')
        
    with open('pickle_files/ids.pickle','wb') as pick:
        pickle.dump(remember_id,pick) 

    with open('pickle_files/images_dict.pickle','wb') as p:
        pickle.dump(temp_image,p,protocol=pickle.HIGHEST_PROTOCOL) 
    
    with open('pickle_files/coordinates_list.pickle','wb') as pickss:
        pickle.dump(new,pickss)
    
    del remember_id 
    del temp_store_list
    del temp_image
    del new 

    return 'Done'

                
            


     
        
        



# if __name__ == '__main__':
#     output = converter('/home/souviks/Documents/codelogicx/l-pesa/flask/pass_mrz/augmentations_for_yolo_1/card_gens',keep_aspect_ratio=True,image_jpg_converter=True,resize_im=640)
#     print(output)











