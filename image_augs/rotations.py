import albumentations as A
import numpy as np

import random
from typing import List , Tuple




def rotate_90deg(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array , List[List] , List]:

    '''
    This function will add rotation to a fraction of images, suppose you have 100 images and you want to add rotation only
    on randomly selected 20 images, by passing fraction param = 0.2, it will randomly select 20 images from 
    dataset and apply rotation to it, for selecting all images you have to pass fraction = 1.0 

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
    try:
        choices = random.choice([0,1])

        if choices == 0:

            transform = A.Compose(
            [A.augmentations.geometric.rotate.Rotate(p=1.0,limit=(-90,-90))],

            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            return transformed['image'] , transformed['bboxes'] , transformed['category_ids']

        else:

            transform = A.Compose(
            [A.augmentations.geometric.rotate.Rotate(p=1.0,limit=(90,90))],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            return transformed['image'] , transformed['bboxes'] , transformed['category_ids']
    except Exception as e:
        pass




def random_rotation(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array , List[List] , List]:

    '''
    For documentation and comments please, refer to rotate_90deg
    
    '''
    try:
        choices = random.choice([0,1])

        if choices == 0:

            val = random.choice(range(5,25))
            transform = A.Compose(
            [A.augmentations.geometric.rotate.Rotate(p=1.0,limit=(val,val))],

            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            return transformed['image'] , transformed['bboxes'] , transformed['category_ids']

        else:
            val = random.choice(range(-25,-5))
            transform = A.Compose(
            [A.augmentations.geometric.rotate.Rotate(p=1.0,limit=(val,val))],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            return transformed['image'] , transformed['bboxes'] , transformed['category_ids']
    except Exception as e:
        pass