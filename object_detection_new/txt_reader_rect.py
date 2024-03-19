from tornado import concurrent
import cv2
import uuid
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from rich.console import Console
from rich.traceback import install
from rich.progress import track
import streamlit as st

import json
import os
import glob
import shutil
from typing import NewType , Union

from object_detection_new.augmentation_rect import *
from object_detection_new import utils_poly
from object_detection_new import yml_writer_poly
import object_detection_new.logging_util as logging_util
# from utils.data_analyser import DataAnalyser


install()
console = Console()
logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])
keep_aspect_ratio = NewType('keep-aspect-ratio','str')
keep_original_image_height = NewType('keep_original_image_height','str')
keep_original_image_width = NewType('keep_original_image_width','str')


'''
Code logic starts from here.

'''

class RectAugmentation():
    def __init__(self,aug_save_folder_name:os.path='rect_augmentation') -> None:
        self.aug_save_folder_name = aug_save_folder_name
        self.augment = ImageAugmentationBox()
        self.store_dict:dict = dict()
        self.counter:int = 0

     

        '''
        : param aug_save_folder_name : give a path where you want to save your augmentations 
        : param image_resize : Image resizing for augmentations
        '''
        
        
        if not os.path.exists(f'{self.aug_save_folder_name}/train/images') or not os.path.exists(f'{self.aug_save_folder_name}/train/labels'):
            os.makedirs(f'{self.aug_save_folder_name}/train/images')
            os.makedirs(f'{self.aug_save_folder_name}/train/labels')
            # console.print(f'[bold blue] [+] {self.aug_save_folder_name}/[bold blue] folder - created..')
        
        if not os.path.exists(f'{self.aug_save_folder_name}/test/images') or not os.path.exists(f'{self.aug_save_folder_name}/test/labels'):
            os.makedirs(f'{self.aug_save_folder_name}/test/images')
            os.makedirs(f'{self.aug_save_folder_name}/test/labels')
            
       
          
        self.train_images_path = f'{self.aug_save_folder_name}/train/images'
        self.train_labels_path = f'{self.aug_save_folder_name}/train/labels'
        self.test_images_path = f'{self.aug_save_folder_name}/test/images'
        self.test_labels_path = f'{self.aug_save_folder_name}/test/labels'
    
        logger.info('RectAugmentation module loaded...')


    def txt_converter(self,image:os.path):
        '''
        txt converter will convert json to polygon acceptable format.
        It will return image and converted json

        : param image : require full path of a image 
        
        : return :
        : im_read       : it will return cv2.imread() image.
        : poly on image : it will return json converted polygon points 
        
        
        '''
        try:

            # temporary to store the result
            new_datas = list()
  
       
            # first we are checking the extension endswith .json or not 
            # then we read the image and json file
            if not image.endswith('.txt'):
                    im_read = cv2.imread(image)
      
                    image_H , image_W , c = im_read.shape
                    TEXT = f"{os.path.splitext(image)[0]}.txt"
                    
                    with open(TEXT) as f:
                        data = f.readlines()
                     
                        for datas in data:
                            strip_data = datas.rstrip().split()
                            strip_data = list(map(float,strip_data))
                            
                            x1,y1,x2,y2 = utils_poly.convert_to_normal_bbox(strip_data[1:],image_W,image_H)
                            self.labels= int(strip_data[0])

                            #taking uuid
                            ids = uuid.uuid4()

                            # taking this in bounding box function and their labels
                            ids = BoundingBox(x1,y1,x2,y2,self.labels)
                           
                            new_datas.append(ids)
                      
                        # input temp datas and image_shape   
                        rect_on_image = BoundingBoxesOnImage(new_datas, shape=im_read.shape)
                      
                        del new_datas
                  
                        yield im_read , rect_on_image

        except Exception as e:
            logger.error(f'problem : Txt converter  desc : {e}')


    def Combined_augmentation(self,im_read , rect_on_image, no_track,image_height=640 , image_width:Union[int,keep_aspect_ratio]=640,
                            blur=True , blur_f = 0.5 ,
                            motionBlur=True, motionBlur_f = 0.5, 
                            rotate=True , rotate_f = 0.5 , 
                            noise=True,noise_f=0.5,
                            perspective=True,perspective_f = 0.5,
                            affine=True,affine_f=0.5,
                            brightness=True,brightness_f=0.5,
                            exposure=True,exposure_f=0.5,
                            removesaturation=True,removesaturation_f=0.5,
                            contrast=True,contrast_f=0.5,
                            shear=True ,shear_f=0.5, 
                            rotate90=True,rotate90_f = 0.5,
                            image_cutout = True,image_cutout_f=0.5,
                            temperature_change=True,temperature_change_f=0.5,
                            vflip=True,vflip_f=0.5,
                            hflip=True,hflip_f=0.5):
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            if rotate:
                frac_data  = round(self.split * rotate_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_rotate,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)
                except Exception as e:
                    logger.warning(f'Image rotation problem : {e}')

            if blur:
                
                frac_data  = round(self.split * blur_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_blur,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path,)
                except:
                  
                    logger.warning(f'Image blur problem : {e}')

            if exposure:
                
                frac_data  = round(self.split * exposure_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_exposure,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path,)
                except:
                  
                    logger.warning(f'Image exposure problem : {e}')


            if vflip:
                
                frac_data  = round(self.split * vflip_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_VFlip,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path,)
                except:
                  
                    logger.warning(f'Image vflip problem : {e}')

            if hflip:
                
                frac_data  = round(self.split * hflip_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_HFlip,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path,)
                except:
                  
                    logger.warning(f'Image hflip problem : {e}')


            if noise:
                frac_data  = round(self.split * noise_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_noise,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image noise problem : {e}')

            if perspective:
                frac_data  = round(self.split * perspective_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_perspective_transform,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image perpective problem : {e}')
            
            if affine:
                frac_data  = round(self.split * affine_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_affine,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image affine problem : {e}')

            if brightness:
                frac_data  = round(self.split * brightness_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_bright_dark,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image brightness problem : {e}')

            # if hue:
            #     frac_data  = round(self.split * hue_f)
            #     try:
            #         if no_track <= frac_data:
                
            #             points2  = executor.submit(self.augment.image_hue,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points2.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

            #     except Exception as e:
            #         logger.warning(f'Image hue problem : {e}')

            if removesaturation:

                frac_data  = round(self.split * removesaturation_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_removeSaturation,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)
                
                except Exception as e:
                    logger.warning(f'Image remove saturation problem : {e}')
            
            if contrast:
                frac_data  = round(self.split * contrast_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_contrast,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image contrast problem : {e}')

      

            if shear:
                frac_data  = round(self.split * shear_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_shear,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image shear problem : {e}')

            if rotate90:
                frac_data  = round(self.split * rotate90_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_rotate90,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image rotation90 problem : {e}')

            # if blur_and_noise:
            #     frac_data  = round(self.split * blur_and_noise_f)
            #     try:
            #         if no_track <= frac_data:
                
            #             points2  = executor.submit(self.augment.blur_and_noise,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points2.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

            #     except Exception as e:
            #         logger.warning(f'Image blur and noise problem : {e}')

            if image_cutout:
                frac_data  = round(self.split * image_cutout_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_cutOut,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image cutout problem : {e}')

            # if mix_aug:
            #     frac_data  = round(self.split * mix_aug_f)
            #     if no_track <= frac_data:
            #         try:
            #             points14  = executor.submit(self.augment.mixed_aug_1,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points14.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)
                    
            #         except Exception as e:
            #             logger.warning(f'Mixaug 1 problem : {e}')

                    
            #         try:
            #             points15  = executor.submit(self.augment.mixed_aug_2,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points15.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

            #         except Exception as e:
            #             logger.warning(f'Mixaug 2 problem : {e}')


            #         try:
            #             points16  = executor.submit(self.augment.mixed_aug_3,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points16.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

            #         except Exception as e:
            #             logger.warning(f'Mixaug 3 problem : {e}')

                    
            #         try:
            #             points17  = executor.submit(self.augment.mixed_aug_4,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points17.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

            #         except Exception as e:
            #             logger.warning(f'Mixaug 4 problem : {e}')
            
            if temperature_change:
                frac_data  = round(self.split * temperature_change_f)
                try:
                    if no_track <= frac_data:
                
                        points2  = executor.submit(self.augment.image_change_colorTemperature,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points2.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)

                except Exception as e:
                    logger.warning(f'Image temperature problem : {e}')

            if motionBlur:
                frac_data  = round(self.split * motionBlur_f)
                try:
                    if no_track <= frac_data:
                        
                        points20  = executor.submit(self.augment.image_motionBlur,im_read,rect_on_image,image_height,image_width) 
                        p , im = next(points20.result())
                        utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)
                        
                except Exception as e:
                    logger.warning(f'Image motion_blur problem : {e}')


            # if weather_change:
            #     frac_data  = round(self.split * weather_change_f)
            #     try:
            #         if no_track <= frac_data:
                        
            #             points20  = executor.submit(self.augment.image_weatherChange,im_read,rect_on_image,image_height,image_width) 
            #             p , im = next(points20.result())
            #             utils_poly.create_new_txt(im,p,self.train_images_path,self.train_labels_path)
                        
            #     except Exception as e:
            #         logger.warning(f'Image weather chnage problem : {e}')

        
            


            



    
    def Image_augmentation(self,folder,train_split=1.0,image_height:Union[int,keep_original_image_height]=640 , image_width:Union[int,keep_aspect_ratio,keep_original_image_width]=640,
                            blur=True , blur_f = 0.5 , motionBlur=True , motionBlur_f = 0.5 ,  rotate=True , rotate_f = 0.5 , 
                            noise=True,noise_f=0.5,perspective=True,perspective_f = 0.5,affine=True,affine_f=0.5,
                            brightness=True,brightness_f=0.5,exposure=True,exposure_f=0.5,removesaturation=True,removesaturation_f=0.5,
                            contrast=True,contrast_f=0.5,Hflip=True,Hflip_f=0.5,Vflip=True,Vflip_f=0.5,
                            shear=True ,shear_f=0.5, rotate90=True,rotate90_f = 0.5,image_cutout = True,image_cutout_f=0.5,
                            temperature_change=True,temperature_change_f=0.5):
        
      
        
        # getting all json and jpg images
        all_images =  glob.glob(f'{folder}/*jpg')
        all_txt =  list(map(lambda x : os.path.splitext(x)[0] + '.txt',all_images))
        classes_ = glob.glob(f'{folder}/classes*txt')

   
  
        
        with open(classes_[0]) as f:
            data = f.readlines()
            for index ,classes_name in enumerate(data):
                self.store_dict[index] = classes_name.rstrip()
        
            
        ##plotting labels data
        # analyser = DataAnalyser(folder,is_json=False)
        # analyser.analyse()
            
       
        if image_width == 'keep-aspect-ratio' and type(image_height) == int :
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
        
        if train_split == 1.0:
                    shutil.rmtree(f'{self.aug_save_folder_name}/test')
       
        my_bar = st.progress(0, text="Augmenting data..",)      
        for c ,images in enumerate(track(all_images,description='Augmenting Images',total=len(all_images[:self.split]))):
            my_bar.progress(int(c /len(all_images) * 100),text="Augmenting data..")
            if c+1 <= self.split:
  
                try:
                    poly = list(self.txt_converter(images))
            
                    #if we want to keep original image height and width -->
                    if image_height == 'keep_original_image_height' and image_width == 'keep_original_image_width':
                        image_height_N , image_width_N , channel = poly[0][0].shape
                    
                    # print(poly)

                    self.Combined_augmentation(poly[0][0],poly[0][1],c+1,image_height_N,image_width_N,
                                               blur=blur , blur_f = blur_f , 
                                               rotate=rotate , rotate_f = rotate_f ,
                                                noise=noise,noise_f=noise_f,
                                                perspective=perspective,perspective_f = perspective_f,
                                                affine=affine,affine_f=affine_f,
                                                brightness=brightness,brightness_f=brightness_f,
                                                removesaturation=removesaturation,removesaturation_f=removesaturation_f,
                                                contrast=contrast,contrast_f=contrast_f,
                                                hflip=Hflip,hflip_f=Hflip_f,
                                                vflip=Vflip,vflip_f=Vflip_f,
                                                exposure=exposure,exposure_f=exposure_f,
                                                shear= shear ,shear_f=shear_f, 
                                                rotate90=rotate90,rotate90_f = rotate90_f,
                                                image_cutout = image_cutout,image_cutout_f=image_cutout_f,
                                                temperature_change=temperature_change,temperature_change_f=temperature_change_f,
                                                motionBlur=motionBlur,motionBlur_f=motionBlur_f)
                
                except Exception as e:
                    console.print('[bold yellow] WARNING [bold yellow] : There is some problem with data , skipping this...')
                    logger.error(f'combined augmentation problem : {e}')
                
                
                
            elif c+1 > self.split:
               
                try:
                
                    poly = list(self.txt_converter(f'{images}'))

                    utils_poly.create_new_txt(poly[0][0] ,poly[0][1],self.test_images_path,self.test_labels_path)

                except Exception as e:
                    console.print('[bold yellow] WARNING [bold yellow] : There is some problem with data , skipping this...')
                    logger.error(f'Test data  problem : {e}')
                   
        # console.print(f'[bold green] Labels name : [/bold green] [bold magenta] {list(self.store_dict.values())} [bold magenta]')
        my_bar.empty()
        yml_writer_poly.yaml_writer(len(self.store_dict.keys()),list(self.store_dict.values()),self.aug_save_folder_name)   

        with open(f'{self.aug_save_folder_name}/classes.txt','w') as f:
            f.write('\n'.join(list(self.store_dict.values()))) 
        
        # console.print(f'[bold green] [+] Total augmented images in "{self.train_images_path}" : {len(os.listdir(self.train_images_path))} [bold green]')
      
      
      
    # @staticmethod
    # def image_format_change(folder):
    #     for im in os.listdir(folder):
    #         if im.endswith('.txt'):
    #             continue
    #         elif im.endswith('.jpg'):
    #             continue
    #         else:
    #             # im_name = im.split('.')[0]
    #             im_name = os.path.splitext(im)[0]
    #             ims = cv2.imread(f'{folder}/{im}')
    #             cv2.imwrite(f'{folder}/{im_name}.jpg',ims)
    #             os.remove(f'{folder}/{im}')