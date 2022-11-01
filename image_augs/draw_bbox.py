from asyncio.log import logger
import bbox_visualizer as bbv
import pybboxes as pbx
import cv2
import numpy as np
from rich.console import Console

import pickle
import os

from image_augs import utils_py as utils_py



console = Console()


def bbbox_viewer(image_path:os.path,id_pickle_file:os.path='pickle_files/ids.pickle') -> np.array :

    '''
        This function will draw bounding boxes.

        : param id_pickle_file : This function requires "ids.pickle" file, for accessing the id.
                              id pickle file saved in pickle_files/ids.pickle inside script folder
        
        : param image_path     : provide image path for draw bounding boxes from train / test folder

    '''
    
    try:
        console.print('[bold green]Pickle file found..[/bold green]')
        with open(id_pickle_file,'rb') as pick:
            ids = pickle.load(pick)
    
    except Exception as e:
        raise FileNotFoundError('Pickle file not found ! please provide full path..')
        

   
    try:
        coor = image_path.replace('.jpg','.txt')
        coor = coor.replace('images','labels')
        try:
            read_image = cv2.imread(image_path)
            h , w,c = read_image.shape

        except Exception as e:
            raise FileNotFoundError('Image file not found ! please provide full path..')
        

        with open(coor) as f:
            loc = f.read().splitlines()
            for cors in loc:
                
                id = cors.split()[0]
                coor = tuple(map(float,cors.split()[1:]))
                rect , rect1 , rect2 = utils_py.random_num()
                x_tl , y_tl , width , height = pbx.convert_bbox(coor, from_type="yolo", to_type="voc",image_size=(w,h)) 
                bboxs = (x_tl,y_tl,width,height)
                cv2.rectangle(read_image, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),(rect,rect1,rect2),2)
                bbv.add_label(read_image,ids[int(id)], bbox=bboxs, top=True,text_color=(rect,rect1,rect2),draw_bg=False)

        return read_image

    except Exception as e:
        logger.error(f'Some problem with bbbox_viewer , {e}')
        bbv.add_label(read_image,'issue_found', bbox=bboxs, top=True,text_color=(rect,rect1,rect2),draw_bg=False)
        return read_image


# if __name__ == '__main__':
# #     count = 1
    
#     path = '/home/souviks/Documents/codelogicx/l-pesa/flask/pass_mrz/augmentations_for_yolo_1/yolov5_aug/train/images/d54a9689-8acd-43c1-a910-e5ae40b1a322.jpg'
#     image = bbbox_viewer(path,'/home/souviks/Documents/codelogicx/l-pesa/flask/pass_mrz/augmentations_for_yolo_1/pickle_files/ids.pickle')
#     cv2.imshow(f'_zoom',image)
# #     cv2.imwrite(f'{count}_zoom.jpg',image)
#     cv2.waitKey(0)
#     count += 1


  
   



