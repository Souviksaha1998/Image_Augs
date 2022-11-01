import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

import os
import random
import shutil


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255)

console = Console()


def image_resize(img_path:os.path,image_size:int=416) -> np.array:
  '''
  This function will resize each images of given image_size.

  : param img_path      : provide image path.
  : param image_size    : image resizing value
  : return resized_image: It will return resized image
  '''
  try:
    output_size = image_size
    im_pth = img_path
    cv2_image = cv2.imread(im_pth)
    old_size = cv2_image.shape[:2] 
    ratio = float(output_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    resized_image = cv2.resize(cv2_image , (new_size[1], new_size[0]))
    return resized_image
  except Exception as e:
    pass




def train_test_split(folder:os.path,split:float=0.20) -> None:
  '''
  This function will divide your data into train_test.

  : param folder : provide folder name
  : param split  : provide split ratio
  
  '''
  try:
    all_images =  os.listdir(f'{folder}/images')
    len_total_images = len(all_images)
    split = int(len_total_images*split)
    sample = random.sample(all_images,split)
        
    if not os.path.exists(f'{folder}/train') or not os.path.exists(f'{folder}/test'):

            os.makedirs(f'{folder}/train/images')
            os.makedirs(f'{folder}/train/labels')
            if float(split) != 0.0:
              os.makedirs(f'{folder}/test/images')
              os.makedirs(f'{folder}/test/labels')
              
    if float(split) != 0.0:
      for images_ in sample:
          shutil.move(f'{folder}/images/{images_}',f'{folder}/test/images')
          text = images_.split('.')[0]
          shutil.move(f'{folder}/labels/{text}.txt',f'{folder}/test/labels')

    all_images =  os.listdir(f'{folder}/images')

    for images_ in all_images:
        shutil.move(f'{folder}/images/{images_}',f'{folder}/train/images')
        text = images_.split('.')[0]
        shutil.move(f'{folder}/labels/{text}.txt',f'{folder}/train/labels')


    shutil.rmtree(f'{folder}/images')
    shutil.rmtree(f'{folder}/labels')

  except Exception as e:
    pass







def folder_creation(folder_name:str) -> str:

    '''
    This function will create a folder for augmentations , where results will be saved
    
    '''
    try:
        saved_folder_name = folder_name 

        if not os.path.exists(f'{saved_folder_name}/images') or not os.path.exists(f'{saved_folder_name}/labels'):
            os.makedirs(f'{saved_folder_name}/images')
            os.makedirs(f'{saved_folder_name}/labels')
            console.print(f'[bold blue]{saved_folder_name}/[bold blue] - created')

        return saved_folder_name
    except Exception as e:
        raise RuntimeWarning(e)


def random_num():
    '''
    This function will provide random value for bounding box color / text
    '''
    rect = random.choice(range(1,256))
    rect1 = random.choice(range(1,256))
    rect2 = random.choice(range(1,256))
    return rect , rect1 , rect2



# def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
#     """Visualizes a single bounding box on the image"""
#     x_min, y_min, w, h = bbox
#     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
#     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
#     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#     cv2.putText(
#         img,
#         text=class_name,
#         org=(x_min, y_min - int(0.3 * text_height)),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.35, 
#         color=TEXT_COLOR, 
#         lineType=cv2.LINE_AA,
#     )
#     return img


# def visualize(image, bboxes, category_ids, category_id_to_name):
#     img = image.copy()
#     for bbox, category_id in zip(bboxes, category_ids):
#         class_name = category_id_to_name[category_id]
#         img = visualize_bbox(img, bbox, class_name)
#     plt.figure(figsize=(12, 12))
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()




    