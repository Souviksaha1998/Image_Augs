from tornado import concurrent
import cv2
import uuid
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from rich.console import Console
from rich.traceback import install
from rich.progress import track

import json
import os
import glob
import shutil
from typing import NewType , Union

from instance_seg.augment_poly import *
from instance_seg import utils_poly
from instance_seg import yml_writer_poly
import instance_seg.logging_util as logging_util
from utils.data_analyser import DataAnalyser

install()
console = Console()
logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])
keep_aspect_ratio = NewType('keep-aspect-ratio','str')
keep_original_image_height = NewType('keep_original_image_height','str')
keep_original_image_width = NewType('keep_original_image_width','str')


'''
Code logic starts from here.

'''

class PolygonAugmentation():
    def __init__(self,aug_save_folder_name:os.path='polygon_augmentation',yolo:bool=True) -> None:
        self.aug_save_folder_name = aug_save_folder_name
        self.augment = ImageAugmentation()
        self.store_dict:dict = dict()
        self.counter:int = 0
        self.yolo = yolo

        '''
        : param aug_save_folder_name : give a path where you want to save your augmentations 
        : param image_resize : Image resizing for augmentations
        '''
        
        
        if os.path.exists(f'{self.aug_save_folder_name}'):
            raise NotImplementedError(f'"{self.aug_save_folder_name}" folder already exist, please change your augmentation saved folder name..')

        if not os.path.exists(f'{self.aug_save_folder_name}/train/images') or not os.path.exists(f'{self.aug_save_folder_name}/train/labels'):
            os.makedirs(f'{self.aug_save_folder_name}/train/images')
            os.makedirs(f'{self.aug_save_folder_name}/train/labels')
            console.print(f'[bold blue] [+] {self.aug_save_folder_name}/[bold blue] folder - created..')
        
        if not os.path.exists(f'{self.aug_save_folder_name}/test/images') or not os.path.exists(f'{self.aug_save_folder_name}/test/labels'):
            os.makedirs(f'{self.aug_save_folder_name}/test/images')
            os.makedirs(f'{self.aug_save_folder_name}/test/labels')
            
        if type(self.yolo) != bool:
            raise TypeError(f'Expected type for yolo is bool, you provided yolo type as {type(self.yolo)}')  
          
        self.train_images_path = f'{self.aug_save_folder_name}/train/images'
        self.train_labels_path = f'{self.aug_save_folder_name}/train/labels'
        self.test_images_path = f'{self.aug_save_folder_name}/test/images'
        self.test_labels_path = f'{self.aug_save_folder_name}/test/labels'
    
        logger.info('ImageAugments module loaded...')

    def json_converter(self,image:os.path):
        '''
        Json converter will convert json to polygon acceptable format.
        It will return image and converted json

        : param image : require full path of a image 
        
        : return :
        : im_read       : it will return cv2.imread() image.
        : poly on image : it will return json converted polygon points 
        
        
        '''
        try:
            # temporary to store the result
            datas = []
            # first we are checking the extension endswith .json or not 
            # then we read the image and json file
            if not image.endswith('.json'):
                    im_read = cv2.imread(image)
        
                    JSON = image.split('.')[0] + '.json'
                    with open(JSON) as f:
                        data = json.load(f)
                        annotations = data['shapes']
                        for i ,annotation in enumerate(annotations):
                            #taking uuid
                            ids = uuid.uuid4()
                            # getting the label
                            self.labels = annotation['label']
                            # we are saving the labels in the dict
                            if self.labels in self.store_dict:
                                pass
                            
                            else:
                                self.store_dict[self.labels] = self.counter
                                self.counter += 1
                            
                            #getting points from json file
                            poly_points = annotation['points']
                            poly_points = [tuple(lst) for lst in poly_points]
                            # taking this in polygon function and their labels
                            ids = Polygon(poly_points,self.labels)
                            datas.append(ids)
                        # input temp datas and image_shape   
                        poly_on_image = PolygonsOnImage(datas, shape=im_read.shape)
                        
                        del datas
                    
                        yield im_read , poly_on_image

        except Exception as e:
            logger.error(f'problem : Json converter  desc : {e}')       
                   
                  
    def Combined_augmentation(self,im_read , poly_on_image, no_track,image_height=640 , image_width:Union[int,keep_aspect_ratio]=640,  blur=True , blur_f = 0.5 , motionBlur = True , motion_blur_f = 0.5, rotate=True , rotate_f = 0.5 , noise=True,noise_f=0.5,perspective=True,perspective_f = 0.5,affine=True,affine_f=0.5,
                              brightness=True,brightness_f=0.5,hue=True,hue_f=0.5,removesaturation=True,removesaturation_f=0.5,contrast=True,contrast_f=0.5,upflip=True,upflip_f=0.5,
                              shear=True ,shear_f=0.5, rotate90=True,rotate90_f = 0.5,blur_and_noise=True,blur_and_noise_f=0.5,image_cutout = True,image_cutout_f=0.5,
                              mix_aug=True,mix_aug_f=0.5,temperature_change=True,temperature_change_f=0.5,weather_change=True,weather_change_f=0.5):
        
        
        # combining all the augmentations here
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            
            if blur:
           
                frac_data  = round(self.split * blur_f)
                if no_track <= frac_data:
                    try:
                        points =  executor.submit(self.augment.image_blur,im_read,poly_on_image,image_height,image_width)
                        p , im = next(points.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)

                    except Exception as e:
                        logger.warning(f'Blur problem : {e} ')
            
            if rotate:
                frac_data  = round(self.split * rotate_f)
                if no_track <= frac_data:
                    try:
                        points2  = executor.submit(self.augment.image_rotate,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'ROTATION problem : {e} ')
                
            if noise:
                frac_data  = round(self.split * noise_f)
                if no_track <= frac_data:
                    try:
                        points3  = executor.submit(self.augment.image_noise,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points3.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Noise problem : {e} ')
                
            if perspective:
                frac_data  = round(self.split * perspective_f)
                if no_track <= frac_data:
                    try:
                        points4  = executor.submit(self.augment.image_perspective_transform,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points4.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Perspective problem : {e} ')
                
            if affine:
                frac_data  = round(self.split * affine_f)
                if no_track <= frac_data:
                    try:
                        points5  = executor.submit(self.augment.image_affine,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points5.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Affine problem : {e} ')
                
            if brightness:
                frac_data  = round(self.split * brightness_f)
                if no_track <= frac_data:
                    try:
                        points6  = executor.submit(self.augment.image_brightness,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points6.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)

                    except Exception as e:
                        logger.warning(f'Brightness problem : {e} ')
            
            if hue:
                frac_data  = round(self.split * hue_f)
                if no_track <= frac_data:
                    try:
                        points6  = executor.submit(self.augment.image_hue,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points6.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'hue problem : {e} ')
                
            if removesaturation:
                frac_data  = round(self.split * removesaturation_f)
                if no_track <= frac_data:
                    try:
                        points7  = executor.submit(self.augment.image_removeSaturation,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points7.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'remove saturation problem : {e} ')
                
            if contrast:
                frac_data  = round(self.split * contrast_f)
                if no_track <= frac_data:
                    try:
                        points8 = executor.submit(self.augment.image_contrast,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points8.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Contrast problem : {e} ')
                
            if upflip:
                frac_data  = round(self.split * upflip_f)
                if no_track <= frac_data:
                    try:
                        points9  = executor.submit(self.augment.image_upFlip,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points9.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'upflip problem : {e} ')
                    
            if shear:
                frac_data  = round(self.split * shear_f)
                if no_track <= frac_data:
                    try:
                        points10  = executor.submit(self.augment.image_shear,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points10.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)

                    except Exception as e:
                        logger.warning(f'shear problem : {e} ')
                
            if rotate90:
                frac_data  = round(self.split * rotate90_f)
                if no_track <= frac_data:
                    try:
                        points11  = executor.submit(self.augment.image_rotate90,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points11.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'rotate90 problem : {e} ')
                
            if blur_and_noise:
                frac_data  = round(self.split * blur_and_noise_f)
                if no_track <= frac_data:
                    try:
                        points12  = executor.submit(self.augment.blur_and_noise,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points12.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Blur and noise problem : {e} ')
                
            if image_cutout:
                frac_data  = round(self.split * image_cutout_f)
                if no_track <= frac_data:
                    try:
                        points13  = executor.submit(self.augment.image_cutOut,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points13.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'ImageCut problem : {e} ')
            
            if mix_aug:
                frac_data  = round(self.split * mix_aug_f)
                if no_track <= frac_data:
                    try:
                        points14  = executor.submit(self.augment.mixed_aug_1,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points14.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    
                    except Exception as e:
                        logger.warning(f'Mixaug1 problem : {e} ')
                    
                    try: 
                        points15  = executor.submit(self.augment.mixed_aug_2,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points15.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Mixaug 2 problem : {e} ')

                    try:

                        points16  = executor.submit(self.augment.mixed_aug_3,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points16.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    
                    except Exception as e:
                        logger.warning(f'MixAug 3 problem : {e} ')
                    
                    try:
                        points17  = executor.submit(self.augment.mixed_aug_4,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points17.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    
                    except Exception as e:
                        logger.warning(f'Mixaug 4 problem : {e} ')
            
            if temperature_change:
                frac_data  = round(self.split * temperature_change_f)
                if no_track <= frac_data:
                    try:
                        points18  = executor.submit(self.augment.image_change_colorTemperature,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points18.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    except Exception as e:
                        logger.warning(f'Temperature change problem : {e} ')

            if weather_change:
                frac_data  = round(self.split * weather_change_f)
                if no_track <= frac_data:
                    try:
                        points2  = executor.submit(self.augment.image_change_colorTemperature,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)
                    
                    except Exception as e:
                        logger.warning(f'weather change problem : {e} ')

            if motionBlur:
                frac_data  = round(self.split * motion_blur_f)
                if no_track <= frac_data:
                    try:
                        points2  = executor.submit(self.augment.image_motionBlur,im_read,poly_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im ,self.labels,p,self.train_images_path,self.train_labels_path,self.store_dict,yolo=self.yolo)

                    except Exception as e:
                        logger.warning(f'MotionBlur problem : {e} ')

                
             
  
               
    
    def Image_augmentation(self,folder,train_split=1.0,image_height:Union[int,keep_original_image_height]=640 , image_width:Union[int,keep_aspect_ratio,keep_original_image_width]=640,blur=True , blur_f = 0.5 , rotate=True , rotate_f = 0.5 , noise=True,noise_f=0.5,perspective=True,perspective_f = 0.5,affine=True,affine_f=0.5,
                            brightness=True,brightness_f=0.5,hue=True,hue_f=0.5,removesaturation=True,removesaturation_f=0.5,contrast=True,contrast_f=0.5,upflip=True,upflip_f=0.5,
                            shear=True ,shear_f=0.5, rotate90=True,rotate90_f = 0.5,blur_and_noise=True,blur_and_noise_f=0.5,image_cutout = True,image_cutout_f=0.5,
                            mix_aug=True,mix_aug_f=0.5,temperature_change=True,temperature_change_f=0.5,weather_change=True,weather_change_f=0.5,motionBlur=True,motionBlur_f=0.5):
        
        #checking if input folder is a directory or not
        dir_checker = os.path.isdir(folder)
        
        if dir_checker != True:
            shutil.rmtree(self.aug_save_folder_name)
            raise NotADirectoryError(f'Provided Path - {folder} is not a directory...')
       
        # changing other images format to same format '.jpg'
        self.image_format_change(folder=folder)
        
        # getting all json and jpg images
        all_images =  glob.glob(f'{folder}/*jpg')
        all_json =  glob.glob(f'{folder}/*json')
        
        # checking json and images are same or not and checking their len too
        if len(all_images) == len(all_json) and len(all_images) >= 1 and len(all_json) >=1:
            del all_json
            
        else:
            shutil.rmtree(self.aug_save_folder_name)
            raise NotImplementedError(f'Images and Jsons are not equal , recheck your annotation folder! \
                                       Total Images : {len(all_images)}  |  Total Json : {len(all_json)}')
        # checking train is float or not
        if type(train_split) != float:
            shutil.rmtree(self.aug_save_folder_name)
            raise TypeError(f'please provide "train split" as "float" , You provided train split as {type(train_split)}')
            
        # checking train split value > 1.0 or not   
        if train_split > 1.0:
            shutil.rmtree(self.aug_save_folder_name)
            raise ValueError(f'[-] please provide "train split" between "0.5 to 1.0", Your provided train split value is : {train_split}')

        analyser = DataAnalyser(folder,is_json=True)
        analyser.analyse()
        

        #checking the types here    
        type_1 = { 
                'blur_f':blur_f,
                'noise_f':noise_f,
                'Noise_and_blur_f':blur_and_noise_f,
                'hue_f':hue_f,
                'removeSaturation_f':removesaturation_f,
                'bright_f':brightness_f,
                'contrast_f':contrast_f,
                'rotation_f':rotate_f,
                'rotation90_f':rotate90_f,
                'affine_f':affine_f,
                'perspective_f':perspective_f,
                'upflip_f' : upflip_f,
                'shear_f' : shear_f,
                'image_cut_f':image_cutout_f,
                'mix_aug_f' : mix_aug_f,
                'temperaure_change_f' : temperature_change_f,
                'weather_change_f' : weather_change_f,
                'motionBlur_f' : motionBlur_f
                
              }
        
        type_2 = { 
                'blur':blur,
                'noise':noise,
                'NB':blur_and_noise,
                'hue':hue,
                'removeSat':removesaturation,
                'bright':brightness,
                'contrast':contrast,
                'rotation':rotate,
                'rotation90':rotate90,
                'affine':affine,
                'perspective':perspective,
                'upflip' : upflip,
                'shear' : shear,
                'image_cut':image_cutout,
                'mix_aug' : mix_aug,
                'temp_change' : temperature_change,
                'weather_change' : weather_change,
                'motionBlur' : motionBlur
                
              }

        # checking the types 
        for types , val in type_1.items():
            if type(val) != float:
                shutil.rmtree(self.aug_save_folder_name)
                raise TypeError(f' [-] please provide "{types}" as  "float" , You provided "{types}" as {type(val)}')
             
                
            if val > 1.0:
                shutil.rmtree(self.aug_save_folder_name)
                raise ValueError(f'[-] please provide "{types}" value  between "0.1 to 1.0" , Your provided "{types}" value is : "{val}"')
        
        for typess , vals in type_2.items():

            if type(vals) != bool:
                shutil.rmtree(self.aug_save_folder_name)
                raise TypeError(f'Please provide "{typess}" as bool , you provided "{typess}" as "{vals}"')
        

       
        if (type(image_height) , image_width) == (int,'keep_aspect_ratio') or (type(image_height),type(image_width)) == (int,int) or (image_height , image_width) == ('keep_original_image_height','keep_original_image_width'):
            pass

        else:
            shutil.rmtree(self.aug_save_folder_name)
            raise ValueError('You provided a wrong image height and width combination, right combinations are - \n (image height , image width) = (int ,int) \n (image height , image width) = (int , "keep_aspect_ratio") \n (image height , image width) = ("keep_original_image_height","keep_original_image_width") ')
        
        if image_width == 'keep_aspect_ratio' and type(image_height) == int :
            image_width_N = 'keep-aspect-ratio'
            image_height_N = image_height
        

        
        if type(image_height) == int and type(image_width) == int:
            image_height_N = image_height
            image_width_N = image_width
            
        
        #shuffling all images
        random.shuffle(all_images)
        # checking how many images
        self.len_total_images = len(all_images)
        # spliting into training set
        self.split = round(self.len_total_images * train_split)
        
        console.print(f'[bold cyan] [+] Total images : {self.len_total_images}   |   Train Split : {self.split} images   |    Test split : {self.len_total_images-self.split} images [bold cyan]')
        
        if train_split == 1.0:
                    shutil.rmtree(self.aug_save_folder_name)
                    shutil.rmtree(f'{self.aug_save_folder_name}/test')
        
        # if c value less than equal to train split then we will add those images in training data and rest will go for test data            
        for c ,images in enumerate(track(all_images,description='Image Augmentation..',total=len(all_images[:self.split]))):

            if c+1 <= self.split:

                try:
                
                    poly = list(self.json_converter(images))

                    #if we want to keep original image height and width -->
                    if image_height == 'keep_original_image_height' and image_width == 'keep_original_image_width':
                        image_height_N , image_width_N , _ = poly[0][0].shape

                    self.Combined_augmentation(poly[0][0],poly[0][1],c+1,image_height_N,image_width_N,blur=blur , blur_f = blur_f , rotate=rotate , rotate_f = rotate_f , noise=noise,noise_f=noise_f,perspective=perspective,perspective_f = perspective_f,affine=affine,affine_f=affine_f,
                                brightness=brightness,brightness_f=brightness_f,hue=hue,hue_f=hue_f,removesaturation=removesaturation,removesaturation_f=removesaturation_f,contrast=contrast,contrast_f=contrast_f,upflip=upflip,upflip_f=upflip_f,
                                shear= shear ,shear_f=shear_f, rotate90=rotate90,rotate90_f = rotate90_f,blur_and_noise=blur_and_noise,blur_and_noise_f=blur_and_noise_f,image_cutout = image_cutout,image_cutout_f=image_cutout_f,
                                mix_aug=mix_aug,mix_aug_f=mix_aug_f,temperature_change=temperature_change,temperature_change_f=temperature_change_f,weather_change=weather_change,weather_change_f=weather_change_f,motionBlur=motionBlur,motion_blur_f=motionBlur_f)
                
                except Exception as e:
                    console.print('[bold yellow] WARNING [bold yellow] : There is some problem with data , skipping this...')
                    logger.error(f'combined augmentation problem : {e}')
                
                
            elif c+1 > self.split:
              
                try:
                    
                    poly = list(self.json_converter(f'{images}'))
                    utils_poly.create_new_txt(poly[0][0] ,self.labels,poly[0][1],self.test_images_path,self.test_labels_path,self.store_dict,yolo=self.yolo)

                except Exception as e:
                    console.print('[bold yellow] WARNING [bold yellow] : There is some problem with data , skipping this...')
                    logger.error(f'combined augmentation problem : {e}')
                
                   
        console.print(f'[bold green] Labels name : [/bold green] [bold magenta] {list(self.store_dict.keys())} [bold magenta]')
        yml_writer_poly.yaml_writer(len(self.store_dict.keys()),list(self.store_dict.keys()),self.aug_save_folder_name)      
        console.print(f'[bold green] [+] Total augmented images in "{self.train_images_path}" : {len(os.listdir(self.train_images_path))} [bold green]')
      
      
      
    @staticmethod
    def image_format_change(folder):
        for im in os.listdir(folder):
            if im.endswith('.json'):
                continue
            elif im.endswith('.jpg'):
                continue
            else:
                
                im_name = os.path.splitext(im)[0]
                ims = cv2.imread(f'{folder}/{im}')
                cv2.imwrite(f'{folder}/{im_name}.jpg',ims)
                os.remove(f'{folder}/{im}')
                      

         


    










  
        
