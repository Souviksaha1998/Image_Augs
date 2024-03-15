import imgaug.augmenters as iaa
import imgaug as ia
from rich.console import Console
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import os
import random
from typing import Tuple
import secrets

import instance_seg.logging_util as logging_util
from object_detection_new.yml_writer_poly import load_yaml




console  = Console()
logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])


'''
ImageAugmentationBox class used only for Image augmentation , there are different augmentation available including blur , noise , rotation and many more.

'''

class ImageAugmentationBox():
    def __init__(self) -> None:
        
        # self.height = height
        
        # console.print('[bold green] [+] Image augmentation Bounding box module loaded....[bold green]')
        logger.info('Image Augmentation Box module loaded successfully...')

        try:
            self.yaml_ = load_yaml("config.yaml")
        except Exception as e:
            logger.warning("yaml config not found , depending on args..")

   
    # done
    def image_rotate(self,image:np.array,bbox:BoundingBoxesOnImage,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:

        '''
        Image Rotate will rotate an image and its polygon points to a desired angle

        : param image                   : accepts cv2.imread() images
        : param boundingBoxesOnImage    : BoundingBox points of that image
        : param rotation angle          : desired angle you want to rotate (-angle , angle)

        : return :
        : new polygon points    : it will return after rotation polygon points
        : aug rotated image     : it will return rotated image
        
        '''
        if args:
            rotation_angle = args[0] 
        else:
            rotation_angle = tuple(map(int,tuple(self.yaml_['rotate']["rotate_range"])))
        
        
        try:
            if args:
                aug = iaa.Sequential([iaa.Rotate(rotation_angle,fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.Rotate((-rotation_angle[0],rotation_angle[1]),fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
                
                
            aug_rotated_image, new_points_bbox  = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            
            yield new_points_bbox , aug_rotated_image
        except Exception as e:
            logger.error(f'problem: Image rotation    desc : {e}')

    # done
    def image_affine(self,image:np.array,bbox:BoundingBoxesOnImage,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            scale_range = args[0] 
        else:
            scale_range = self.yaml_['affine']["affine_range"]
        try:
            if args:
                aug = iaa.Sequential([iaa.Affine(scale=scale_range,fit_output=False,),
                                iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.Affine(scale=scale_range,fit_output=False,),
                                iaa.Resize({"height": H, "width": W})])
        
            aug_affine_image , new_points_bbox  = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image affine    desc : {e}')


    # done
    def image_perspective_transform(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            perspective = args[0] 
            
        else:
            perspective = self.yaml_['perspective']["perspective_range"]
        try:
            if args:
                aug = iaa.Sequential([iaa.PerspectiveTransform(scale=perspective,fit_output=False,),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.PerspectiveTransform(scale=perspective,fit_output=False,),
                                    iaa.Resize({"height": H, "width": W})])
                
            aug_affine_image,new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image perspective transform    desc : {e}')
    # done
    def image_noise(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            noise_val = args[0]
            
        else:
            noise_val = self.yaml_['noise']["noise_range"]

        
        try:
            choice = random.choice([0,1])
       
            if choice == 0:

                if args:
                    aug = iaa.Sequential([iaa.SaltAndPepper(p=noise_val),
                                    iaa.Resize({"height": H, "width": W})])
                else:
                    aug = iaa.Sequential([iaa.SaltAndPepper(p=noise_val),
                                    iaa.Resize({"height": H, "width": W})])
            else:

                if args:
                    aug = iaa.Sequential([iaa.SaltAndPepper(p=noise_val,per_channel=True),
                                    iaa.Resize({"height": H, "width": W})])
                else:
                    aug = iaa.Sequential([iaa.SaltAndPepper(p=noise_val,per_channel=True),
                                    iaa.Resize({"height": H, "width": W})])
                    
            aug_affine_image,new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image noise    desc : {e}')

    # done
    def image_blur(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            blur_val = args[0]
            
        else:
            blur_val = self.yaml_['blur']["blur_range"]
        
        # print(f'blur val : {blur_val}')
        try:
            if args:
                aug = iaa.Sequential([iaa.GaussianBlur(sigma=blur_val),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.GaussianBlur(sigma=blur_val),
                                    iaa.Resize({"height": H, "width": W})])
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        
        except Exception as e:
            # print(e)
            logger.error(f'problem: Image blur    desc : {e}')

    # done
    def image_motionBlur(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            motion_val = args[0]
            
        else:
            motion_val = self.yaml_['motionblur']["motionblur_range"]
        try:
            if args:
                aug = iaa.Sequential([iaa.MotionBlur(k=15, angle=motion_val),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.MotionBlur(k=15, angle=motion_val),
                                    iaa.Resize({"height": H, "width": W})])
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        
        except Exception as e:
            logger.error(f'problem: Image motion blur    desc : {e}')
    # done
    def image_bright_dark(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            hue_val = args[0]
        else:
            hue_val = self.yaml_['bright&dark']["bright&dark_range"]

      
        try:
            if args:
                aug = iaa.Sequential([iaa.Multiply(hue_val, per_channel=0.2,),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.Multiply(hue_val, per_channel=0.2,),
                                    iaa.Resize({"height": H, "width": W})])
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image hue    desc : {e}')


    # def image_hue(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
    #     if args:
    #         hue_val = args[0]
         
    #     else:
    #         hue_val = self.yaml_['bright&dark']["bright&dark_range"]
      
    #     try:
    #         if tuple_:
    #             aug = iaa.Sequential([iaa.WithColorspace(
    #             to_colorspace="HSV",
    #             from_colorspace="RGB",
    #             children=iaa.WithChannels(
    #                 0,
    #                 iaa.Add(hue_val)
    #             )
    #         ),
    #                                 iaa.Resize({"height": H, "width": W})])
    #         else:
    #             aug = iaa.Sequential([iaa.WithColorspace(
    #             to_colorspace="HSV",
    #             from_colorspace="RGB",
    #             children=iaa.WithChannels(
    #                 0,
    #                 iaa.Add(hue_val)
    #             )
    #         ),
    #                                 iaa.Resize({"height": H, "width": W})])
                
    #         aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
    #         new_points_bbox = new_points_bbox.bounding_boxes
    #         yield new_points_bbox , aug_affine_image
    #     except Exception as e:
    #         logger.error(f'problem: Image hue    desc : {e}')

    # done
    def image_exposure(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            bright = args[0]
            

        else:
            bright = self.yaml_['exposure']["exposure_range"]
        try:
            
            if args:
       
                aug = iaa.Sequential([iaa.WithBrightnessChannels(iaa.Add(bright)),
                                iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.WithBrightnessChannels(iaa.Add(bright)),
                                iaa.Resize({"height": H, "width": W})])
     
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image brightness    desc : {e}')

    # done
    def image_change_colorTemperature(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            temp = args[0]

        else:
            temp = self.yaml_['temperature']["temperature_range"]
        try:
            if args:
                aug = iaa.Sequential([iaa.ChangeColorTemperature(temp),
                                    
                                    iaa.Resize({"height": H, "width": W})])
                
            else:
                aug = iaa.Sequential([iaa.ChangeColorTemperature(temp),
                                    iaa.Resize({"height": H, "width": W})])
            
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image color temp    desc : {e}')

    # done
    def image_removeSaturation(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            sat = args[0]
        else:
            sat = self.yaml_['remove_saturation']["remove_saturation_range"]
        # print(list(range(*sat)))
        try:

            if args:
                
                aug = iaa.Sequential([iaa.RemoveSaturation(sat/10),
                                    iaa.Resize({"height": H, "width": W})])
                
            else:
                sat = list(range(*sat))
                sat = list(map(lambda x : x/10,sat))
                # print(sat)
                choices = random.choice(sat)
                aug = iaa.Sequential([iaa.RemoveSaturation(choices),
                                    iaa.Resize({"height": H, "width": W})])
                
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image removeSaturation    desc : {e}')

    # done
    def image_contrast(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            con = args[0]
   
        else:
            con = self.yaml_['contrast']["contrast_range"]
        try:
            if args:
                aug = iaa.Sequential([iaa.LinearContrast(con),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                aug = iaa.Sequential([iaa.LinearContrast(con),
                                    iaa.Resize({"height": H, "width": W})])
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image contrast    desc : {e}')
    # done
    def image_HFlip(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            flip = args[0]
        else:
            flip = 1
        try:
   
            aug = iaa.Sequential([iaa.Fliplr(flip),
                                iaa.Resize({"height": H, "width": W})])
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image Hlfip    desc : {e}')

    # done
    def image_VFlip(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            flip = args[0]
        else:
            flip = 1
        try:
   
            aug = iaa.Sequential([iaa.Flipud(flip),
                                iaa.Resize({"height": H, "width": W})])
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image Vlfip    desc : {e}')

    
    # done
    def image_shear(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            shear = args[0]
        else:
            shear =  self.yaml_['shear']["shear_range"]
        try:
            if args:
              
                aug = iaa.Sequential([iaa.ShearX(shear=shear,fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
        
            else:
                aug = iaa.Sequential([iaa.ShearX(shear=shear,fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image shear    desc : {e}')
    
    # done
    def image_rotate90(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:
        if args:
            rotate90 = args[0]
           
        else:
            rotate90 = 90
        try:
            
            choice = random.choice([0,1])
            if args:
                aug = iaa.Sequential([iaa.Rotate(rotate=rotate90,fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
                
            else:
                if choice == 0:
                    
                    aug = iaa.Sequential([iaa.Rotate(rotate=rotate90,fit_output=False,),
                                    iaa.Resize({"height": H, "width": W})])
                else:
                    
                    aug = iaa.Sequential([iaa.Rotate(rotate=-rotate90,fit_output=False),
                                    iaa.Resize({"height": H, "width": W})])
                
                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image rotate90   desc : {e}')

    #beta
    def image_weatherChange(self,image:np.array,bbox:BoundingBox,H,W,rain_speed:range=((0.09, 0.2))) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            choice = secrets.choice([0,1,2])
        
            if choice == 0:
     
                aug = iaa.Sequential([iaa.imgcorruptlike.Fog(severity=2),
                                iaa.Resize({"height": H, "width": W})])
            elif choice == 1:
         
                aug = iaa.Sequential([iaa.Rain(speed=rain_speed,drop_size=(0.009,0.01),),
                                iaa.Resize({"height": H, "width": W})])
            elif choice == 2:
                # aug =  aug = iaa.imgcorruptlike.Snow(severity=1)        
                aug = iaa.Sequential([iaa.imgcorruptlike.Snow(severity=1),
                                iaa.Resize({"height": H, "width": W})])

            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image weatherchange    desc : {e}')
    # done
    def image_cutOut(self,image:np.array,bbox:BoundingBox,H,W,*args) -> Tuple[BoundingBoxesOnImage,np.array]:

        if args:
            square = args[0]
        else:
            square = self.yaml_['box']["box_range"]
        # print(f'square :{square}')
        try:
            
            if args:
                # aug =  iaa.Cutout(fill_mode="constant", cval=random.choice([128,255]),size=size,nb_iterations=number_of_square)
                aug = iaa.Sequential([iaa.Cutout(fill_mode="constant", cval=random.choice([128,255]),size=0.1,nb_iterations=square),
                                    iaa.Resize({"height": H, "width": W})])
            else:
                # print("yes")
                aug = iaa.Sequential([iaa.Cutout(fill_mode="constant", cval=random.choice([128,255]),size=0.1,nb_iterations=square),
                                    iaa.Resize({"height": H, "width": W})])
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)

            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image cutout    desc : {e}')












            

    def blur_and_noise(self,image:np.array,bbox:BoundingBox,H,W) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            aug = iaa.Sequential([iaa.GaussianBlur(sigma=(0.8, 3.5)),
                                iaa.Resize({"height": H, "width": W}),
                                iaa.SaltAndPepper(p=(0.02,0.07))])
            
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image blur and noise    desc : {e}')

    
    def mixed_aug_1(self,image:np.array,bbox:BoundingBox,H,W) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            aug = iaa.Sequential([iaa.Affine(scale=(0.5,1.6),fit_output=False),
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.Resize({"height": H, "width": W})])
            
                                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Mix aug 1    desc : {e}')


    def mixed_aug_2(self,image:np.array,bbox:BoundingBox,H,W) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            choices = random.choice([0.4,0.5,0.6,0.7,0.8])
            # aug = iaa.RemoveSaturation(choices)
            aug = iaa.Sequential([iaa.Rotate((-13,13),fit_output=False),
                                iaa.WithBrightnessChannels(iaa.Add((-40,40))),
                                iaa.RemoveSaturation(choices),
                                iaa.PerspectiveTransform(scale=(0.08,0.12),fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            
                                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        except Exception as e:
            logger.error(f'problem: mix aug 2    desc : {e}')

    def mixed_aug_3(self,image:np.array,bbox:BoundingBox,H,W) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            choice = random.choice([0,1])
            if choice == 0:
                aug = iaa.Sequential([iaa.ShearX(shear=(-15,15),fit_output=False),
                                    iaa.GaussianBlur(sigma=(0.8, 2.8)),
                                    iaa.ChangeColorTemperature((1100, 7000)),
                                    iaa.Resize({"height": H, "width":W})])
            
            else:
                aug = iaa.Sequential([iaa.ShearY(shear=(-15,15),fit_output=False),
                                    iaa.LinearContrast((0.8, 1.6)),
                                    iaa.Resize({"height": H, "width":W})])
    
                            
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image
        
        except Exception as e:
            logger.error(f'problem: Mix aug 3    desc : {e}')

    def mixed_aug_4(self,image:np.array,bbox:BoundingBox,H,W) -> Tuple[BoundingBoxesOnImage,np.array]:
        try:
            choice = random.choice([-90,90])
            choices =  random.choice([0,1])
            
            if choices == 0:
            
                aug = iaa.Sequential([iaa.Rotate(rotate=choice,fit_output=False),
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.SaltAndPepper(p=(0.01,0.06)),
                                iaa.Resize({"height": H, "width": W})])
                
            else:
                
                aug = iaa.Sequential([iaa.Rotate(rotate=choice,fit_output=False),
                                iaa.WithBrightnessChannels(iaa.Add((-20,20))),
                                iaa.SaltAndPepper(p=(0.01,0.06)),
                                iaa.Resize({"height": H, "width": W})])
            
                                
            aug_affine_image, new_points_bbox = aug(image=image,bounding_boxes=bbox)
            new_points_bbox = new_points_bbox.remove_out_of_image().clip_out_of_image()
            new_points_bbox = new_points_bbox.bounding_boxes
            yield new_points_bbox , aug_affine_image

    

        except Exception as e:
            logger.error(f'problem: Mixed aug 4    desc : {e}')