import cv2
import numpy as np
from skimage.util import random_noise
from PIL import Image , ImageEnhance
import albumentations as A
import numpy as np

import random
from typing import List , Tuple




#zoom
def clipped_zoom(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array,List[List],List]:
    '''
    This function will zoom in/out an image and it coordinations

    : param image        : accepts cv2.imread() image
    : param bboxes       : bounding boxes location
    : param category_ids : ids of each category ( its a dictionary)
    : return             : transformed image , its bounding boxes and their ids
    '''

    zoom = random.choice([0.7,0.8,0.9,1.1,1.2,1.3,1.4])
    shears = random.choice([None,3,-3])
    transform = A.Compose(
    [A.augmentations.geometric.transforms.Affine(shear=shears,p=1.0,scale=zoom)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    return transformed['image'] , transformed['bboxes'] , transformed['category_ids']




#affine
def affine_transform(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array,List[List],List]:

    '''
    This function will apply affine transform on an image and it coordinations

    : param image        : accepts cv2.imread() image
    : param bboxes       : bounding boxes location
    : param category_ids : ids of each category ( its a dictionary)
    : return             : transformed image , its bounding boxes and their ids
    '''

    affines = random.choice(range(-70,70,20))
    transform = A.Compose(
    [A.augmentations.geometric.transforms.ElasticTransform(alpha_affine=affines,p=.8,same_dxdy=True,)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    return transformed['image'] , transformed['bboxes'] , transformed['category_ids']





#blurr
def blur(image:np.array) -> np.array:
    '''
        This function will blur images.
        Blur values are selected by random. (3,3) , (5,5)

        : param image : accepts cv2.imread() image
        : return      : blurred image
    '''
    blurrr = [(3,3),(5,5)]
    blurrr = random.choice(blurrr)
    return cv2.GaussianBlur(image ,blurrr,0)




#noise
def noise(image:np.array) -> np.array:

    '''
        This function will add noise to the images.
        Noise values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : noisy image

    '''
    noise = [0.001,0.002,0.003,0.004]
    noise_1 = random.choice(noise)
    noise_img = random_noise(image, mode='s&p',amount=noise_1)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img




#brightness-darkness
def brightness_contrast(image:np.array)-> Tuple[np.array, np.array]:

    '''
        This function will add brightness and darkness to the images.
        brightness/darkness values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : brightness image , darkness image
    '''
    value = random.choice([10,20,30,40,50])
    value2 = random.choice([30,40,50,60,70,80,90])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    v_bright = cv2.add(v,value)
    v_dark = cv2.add(v,-value2)
    final_hsv_bright = cv2.merge((h, s, v_bright))
    final_hsv_dark = cv2.merge((h, s, v_dark))

    img_bright = cv2.cvtColor(final_hsv_bright, cv2.COLOR_HSV2BGR)
    img_dark = cv2.cvtColor(final_hsv_dark, cv2.COLOR_HSV2BGR)
    return img_bright , img_dark




#noise_and_blur 
def noise_and_blur(image:np.array) -> np.array:

    '''
        This function will add noise and blur to the images.
        

        : param image : accepts cv2.imread() image
        : return      : noisy and blur image

    '''

    blurs = blur(image)
    noises = noise(blurs)
    return noises




#hue
def hue(image:np.array) -> np.array:

    '''
        This function will add hue to the images.
        Hue values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : hue image

    '''

    hue_choice = random.choice(range(10,80))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    hnew = np.mod(h + int(hue_choice), 180).astype(np.uint8)
    hsv_new = cv2.merge([hnew,s,v])
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    return bgr_new




#image_translate
def image_translation(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array,List[List],List]:

    '''
    
    This function will apply translation on an image and it coordinations

    : param image        : accepts cv2.imread() image
    : param bboxes       : bounding boxes location
    : param category_ids : ids of each category ( its a dictionary)
    : return             : transformed image , its bounding boxes and their ids
    
    '''

    rotate = random.choice([0,1,2,3])
    zoom_out = random.choice([0.7,0.8,0.9])
    translate = random.choice([-50,40,-39,20,-10,10,20,30,40,50])
    transform = A.Compose(
    [A.augmentations.geometric.transforms.Affine(translate_px=translate,scale=zoom_out,rotate=rotate)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    return transformed['image'] , transformed['bboxes'] , transformed['category_ids']




#saturation
def image_saturation(image:np.array) -> np.array:

    '''
        This function will add saturation to the images.
        Hue values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : saturated image

    '''

    sat_level = random.choice([1,2,3])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(sat_level)
    saturated_image = np.asarray(img2)
    saturated_image = cv2.cvtColor(saturated_image, cv2.COLOR_RGB2BGR)
    return saturated_image 


#contrast
def contrast(img:np.array) -> np.array:

    '''
        
        This function will add contrast to the images.
        contrast values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : hue image

    '''

    level = random.choice(range(40,50))
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb)
    con = im_pil.point(contrast)
    bgr = np.asarray(con)
    contrast_im = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    return contrast_im


#more aug on rotation
def more_aug(imagess:np.array) -> np.array:
    '''
        This function will randomly apply on rotated images (random rotation , +/- 90deg), to give it more complexity.
    '''
    more_augs = random.choice(['blur','noise','saturation','normal','contrast'])


    if more_augs == 'blur':
        return blur(imagess)
    if more_augs == 'noise':
        return noise(imagess)
    if more_augs == 'saturation':
        return image_saturation(imagess)
    if more_augs == 'contrast':
        return contrast(imagess)
    if more_augs == 'normal':
        return imagess





#normalize height and width for yolo
def normalize_yolo(image:np.array ,X_tl:int,Y_tl:int,b_width:int,b_height:int) -> Tuple[float,float,float,float]:

    '''
    This function will normalize coordinates values (200,100,...,) --> (0.002,0.444,...,)

    : param image              : it accepts cv2.imread()
    : param X_tl , Y_tl        : top left corner
    : param b_width , b_height : height and width of the bounding boxes

    '''

    image_height ,image_width , c = image.shape
    X_tl = (X_tl+(b_width/2))/image_width
    Y_tl = (Y_tl+(b_height/2))/image_height
    b_width = b_width/image_width
    b_height = b_height/image_height

    return round(X_tl,6) , round(Y_tl,6) , round(b_width,6) , round(b_height,6)




def vertical_flip(image:np.array,bboxes:List[List],category_ids:List) -> Tuple[np.array,List[List],List]:
    '''
    This function will rotate images 180degree

    : param image        : accepts cv2.imread() image
    : param bboxes       : bounding boxes location
    : param category_ids : ids of each category ( its a dictionary)
    : return             : transformed image , its bounding boxes and their ids

    '''

    transform = A.Compose(
    [A.augmentations.geometric.transforms.VerticalFlip(p=1),
    A.augmentations.geometric.transforms.HorizontalFlip(p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    return transformed['image'] , transformed['bboxes'] , transformed['category_ids']




#it will distort images little bit
# def GridDistortion(image,bboxes,category_ids): 
#     distort = random.choice([0.2,0.3])
#     transform = A.Compose(
#     [A.augmentations.geometric.transforms.GridDistortion(distort_limit=distort,normalized=True,p=1)],
#     bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

#     transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

#     return transformed['image'] , transformed['bboxes'] , transformed['category_ids']        










