import imgaug.augmenters as iaa
from rich.console import Console
import numpy as np
from imgaug.augmentables.polys import Polygon

import os
import warnings
import random
from typing import Tuple

import instance_seg.logging_util as logging_util

warnings.filterwarnings('ignore')



console  = Console()
logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])


'''
ImageAugmentation class used only for Image augmentation , there are different augmentation available including blur , noise , rotation and many more.

'''


class ImageAugmentation():
    def __init__(self) -> None:
        
        # self.height = height
        
        console.print('[bold green] [+] Image augmentation module loaded....[bold green]')
        logger.info('Image Augmentation module loaded successfully...')
           
    def image_rotate(self,image:np.array,polygons:Polygon,H,W,rotation_angle:int=16) -> Tuple[Polygon,np.array]:

        '''
        Image Rotate will rotate an image and its polygon points to a desired angle

        : param image           : accepts cv2.imread() images
        : param polygons        : Polygon points of that image
        : param rotation angle  : desired angle you want to rotate (-angle , angle)

        : return :
        : new polygon points    : it will return after rotation polygon points
        : aug rotated image     : it will return rotated image
        
        '''
        try:
            aug = iaa.Sequential([iaa.Rotate((-rotation_angle,rotation_angle),fit_output=False),
                                iaa.SaltAndPepper(p=(0.01,0.03)),
                                iaa.Multiply((0.2, 1.1), per_channel=1.0,),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_rotated_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_rotated_image
        except Exception as e:
            logger.error(f'problem: Image rotation    desc : {e}')
        
        
    def image_affine(self,image:np.array,polygons:Polygon,H,W,scale_range:range=(0.5,1.5)) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.Affine(scale=scale_range,fit_output=False,),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image affine    desc : {e}')

    def image_perspective_transform(self,image:np.array,polygons:Polygon,H,W,scale_range:range=(0.10,0.12)) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.PerspectiveTransform(scale=scale_range,fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image perspective transform    desc : {e}')


    def image_noise(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            choice = random.choice([0,1])
            if choice == 0:
                aug = iaa.Sequential([iaa.SaltAndPepper(p=(0.01,0.05)),
                                iaa.Resize({"height": H, "width": W})])
            else:
                # aug = iaa.AdditiveLaplaceNoise(scale=(5,0.1*200), per_channel=True)
                aug = iaa.Sequential([iaa.SaltAndPepper(p=(0.01,0.05),per_channel=True),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image noise    desc : {e}')


    def image_blur(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.GaussianBlur(sigma=(1.2, 4.0)),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        
        except Exception as e:
            logger.error(f'problem: Image blur    desc : {e}')

    
    #chnag thissssssss
    def image_hue(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            # aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)
            aug = iaa.Sequential([iaa.Multiply((0.5, 1.5), per_channel=1.0,),
                                  iaa.WithBrightnessChannels(iaa.Add((-40,40))),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image hue    desc : {e}')

    def image_brightness(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            ranges = random.choice([0,1])
            if ranges == 0:
                # aug = iaa.WithBrightnessChannels(iaa.Add((10,40)))
                aug = iaa.Sequential([iaa.WithBrightnessChannels(iaa.Add((10,50))),
                                iaa.Resize({"height": H, "width": W})])
            else:
                # aug = iaa.WithBrightnessChannels(iaa.Add((-40,-10)))
                aug = iaa.Sequential([iaa.WithBrightnessChannels(iaa.Add((-50,-10))),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image brightness    desc : {e}')

    def image_change_colorTemperature(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            # aug =  iaa.ChangeColorTemperature((1100, 10000))
            aug = iaa.Sequential([iaa.ChangeColorTemperature((1100, 6500)),
                                  iaa.GaussianBlur(sigma=(1.2, 2.7)),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image color temp    desc : {e}')

    def image_removeSaturation(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            choices = random.choice([0.3,0.4,0.5,0.6,0.7,0.8,0.9])
            aug = iaa.Sequential([iaa.RemoveSaturation(choices),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image removeSaturation    desc : {e}')

    def image_contrast(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
        
            aug = iaa.Sequential([iaa.LinearContrast((0.8, 1.7)),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image contrast    desc : {e}')

    def image_upFlip(self,image:np.array,polygons:Polygon,H,W,flip_percentage:float=1.0) -> Tuple[Polygon,np.array]:
        try:
            # aug =  iaa.Flipud(flip_percentage)
            aug = iaa.Sequential([iaa.Flipud(flip_percentage,),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image uplfip    desc : {e}')

    def image_shear(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            choice = random.choice([0,1])
            if choice == 0:
                # aug =  iaa.ShearX(shear=(-12,12))
                aug = iaa.Sequential([iaa.ShearX(shear=(-18,18),fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            else:
                # aug =  iaa.ShearY(shear=(-12,12))
                aug = iaa.Sequential([iaa.ShearX(shear=(-18,18),fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image shear    desc : {e}')

    def image_rotate90(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            choice = random.choice([0,1])
            if choice == 0:
                # aug =  iaa.Rotate(rotate=90,fit_output=True)
                aug = iaa.Sequential([iaa.Rotate(rotate=90,fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            else:
                # aug =  iaa.Rotate(rotate=-90,fit_output=True)
                aug = iaa.Sequential([iaa.Rotate(rotate=-90,fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image rotate90   desc : {e}')
            

    # beta mode
    def image_weatherChange(self,image:np.array,polygons:Polygon,H,W,rain_speed:range=(0.09, 0.2)) -> Tuple[Polygon,np.array]:
        try:
            choice = random.choice([0,1,2])
            if choice == 0:
     
                aug = iaa.Sequential([iaa.imgcorruptlike.Fog(severity=2),
                                iaa.Resize({"height": H, "width": W})])
            elif choice == 1:
         
                aug = iaa.Sequential([iaa.Rain(speed=rain_speed,drop_size=(0.009,0.01)),
                                iaa.Resize({"height": H, "width": W})])
            elif choice == 2:
                # aug =  aug = iaa.imgcorruptlike.Snow(severity=1)        
                aug = iaa.Sequential([iaa.imgcorruptlike.Snow(severity=1),
                                iaa.Resize({"height": H, "width": W})])

            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image weatherchange    desc : {e}')

    def image_cutOut(self,image:np.array,polygons:Polygon,H,W,number_of_square:int=2,size:float=0.1) -> Tuple[Polygon,np.array]:
        try:
            # aug =  iaa.Cutout(fill_mode="constant", cval=random.choice([128,255]),size=size,nb_iterations=number_of_square)
            aug = iaa.Sequential([iaa.Cutout(fill_mode="constant", cval=random.choice([128,255]),size=size,nb_iterations=number_of_square),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Image cutout    desc : {e}')
    
    def blur_and_noise(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.GaussianBlur(sigma=(0.8, 3.5)),
                                iaa.Resize({"height": H, "width": W}),
                                iaa.SaltAndPepper(p=(0.02,0.07))])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Image blur and noise    desc : {e}')
    
    def mixed_aug_1(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.Affine(scale=(0.5,1.6),fit_output=False),
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.Resize({"height": H, "width": W})])
            
                                
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: Mix aug 1    desc : {e}')
    
    def mixed_aug_2(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            choices = random.choice([0.4,0.5,0.6,0.7,0.8])
            # aug = iaa.RemoveSaturation(choices)
            aug = iaa.Sequential([iaa.Rotate((-13,13),fit_output=False),
                                iaa.WithBrightnessChannels(iaa.Add((-40,40))),
                                iaa.RemoveSaturation(choices),
                                iaa.PerspectiveTransform(scale=(0.08,0.12),fit_output=False),
                                iaa.Resize({"height": H, "width": W})])
            
                                
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        except Exception as e:
            logger.error(f'problem: mix aug 2    desc : {e}')
    
    def mixed_aug_3(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
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
    
                            
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        
        except Exception as e:
            logger.error(f'problem: Mix aug 3    desc : {e}')
    
    def mixed_aug_4(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
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
            
                                
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.remove_out_of_image().clip_out_of_image()
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image

        except Exception as e:
            logger.error(f'problem: Mixed aug 4    desc : {e}')
    
    def image_motionBlur(self,image:np.array,polygons:Polygon,H,W) -> Tuple[Polygon,np.array]:
        try:
            aug = iaa.Sequential([iaa.MotionBlur(k=15, angle=[-45, 45]),
                                iaa.Resize({"height": H, "width": W})])
            new_points_polygons, aug_affine_image = aug(polygons=polygons,image=image)
            new_points_polygons = new_points_polygons.polygons
            yield new_points_polygons , aug_affine_image
        
        except Exception as e:
            logger.error(f'problem: Image motion blur    desc : {e}')
        
    
    
    
    


  