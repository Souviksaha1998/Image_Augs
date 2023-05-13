import albumentations as A
import cv2
from rich.console import Console
import numpy as np
from skimage.util import random_noise
from PIL import Image , ImageEnhance

import os
import secrets

from classification import logging_util





logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])

console = Console()

class ClassificationAugmentation():
    def __init__(self,height:int=256,width:int=256) -> None:
        self.height = height
        self.width  = width
        console.print('[bold green] [+] Classification Augmentation module loaded..[/bold green]')
        logger.info('Classification module loaded successfully..')

    @property
    def Height(self):
        return self.height
    
    @Height.setter
    def Height(self,value):
        self.height = value
    
    @property
    def Width(self):
        return self.width
    
    @Width.setter
    def Width(self,value):
        self.width = value

    def horizontalFlip(self,image):
        try:
            transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),A.HorizontalFlip(p=1.0),])
            
            augmented_image = transform(image=image)['image']

            yield augmented_image 

        except Exception as e:
            logger.error(f'horizontal flip problem  , desc : {e}')

    def blur(self,image):
        '''
        This function will blur images.
        Blur values are selected by random. (3,3) , (5,5)

        : param image : accepts cv2.imread() image
        : return      : blurred image
        '''
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            blurrr = [(3,3),(5,5),(7,7)]
            blurrr = secrets.choice(blurrr)
            yield cv2.GaussianBlur(image ,blurrr,0) 
        
        except Exception as e:
            logger.error(f'blur  problem  , desc : {e}')


    def noise(self,image):

        '''
        This function will add noise to the images.
        Noise values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : noisy image

        '''
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            noise = [0.001,0.002,0.003,0.004]
            noise_1 = secrets.choice(noise)
            noise_img = random_noise(image, mode='s&p',amount=noise_1)
            noise_img = np.array(255*noise_img, dtype = 'uint8')
            yield noise_img 
        except Exception as e:
            logger.error(f'noise problem  , desc : {e}')
    
    def randomBrightness(self,image):
        try:
            transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
                                A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, brightness_by_max=True, always_apply=False, p=1)])
        
            augmented_image = transform(image=image)['image']
            yield augmented_image

        except Exception as e:
            logger.error(f'randomBrightness problem  , desc : {e}')
    
    def randomContrast(self,image):

        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            level = secrets.choice(range(30,55))
            factor = (259 * (level + 255)) / (255 * (259 - level))
            def contrast(c):
                return 128 + factor * (c - 128)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(rgb)
            con = im_pil.point(contrast)
            bgr = np.asarray(con)
            contrast_im = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            yield contrast_im
        except Exception as e:
            logger.error(f'randomConrast problem  , desc : {e}')
    
    #need to change value
    # def fog(self,image):
    #     transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
    #                            A.augmentations.transforms.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1)])

    #     augmented_image = transform(image=image)['image']
    #     return augmented_image
    
    #need to change value
    # def rain(self,image):
    #     transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
    #                            A.augmentations.transforms.RandomRain (slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=1)])
    
    #     augmented_image = transform(image=image)['image']
    #     return augmented_image

    # def randomSunflare(self,image):
    #     transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
    #                             A.augmentations.transforms.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=1)])
    #     random.seed(7)
    #     augmented_image = transform(image=image)['image']
    #     return augmented_image

    def randomShadow(self,image):
        try:
            transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
                                    A.augmentations.transforms.RandomShadow (shadow_roi=(0, 0.8, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1)])

            augmented_image = transform(image=image)['image']
            yield augmented_image
        except Exception as e:
            logger.error(f'randomShadow problem  , desc : {e}')
    
    def sharpen(self,image):
        try:
            transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
                                    A.augmentations.transforms.Sharpen (alpha=(0.1, 0.5), lightness=(0.3, 0.6), always_apply=False, p=1)])
    
            augmented_image = transform(image=image)['image']
            yield augmented_image

        except Exception as e:
            logger.error(f'sharpen problem  , desc : {e}')

    def hue(self,image):
        '''
        This function will add hue to the images.
        Hue values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : hue image

        '''
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            hue_choice = secrets.choice(range(10,30))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            hnew = np.mod(h + int(hue_choice), 180).astype(np.uint8)
            hsv_new = cv2.merge([hnew,s,v])
            bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
            yield bgr_new
        except Exception as e:
            logger.error(f'HUE problem  , desc : {e}')
    
    def saturation(self,image):
        '''
        This function will add saturation to the images.
        Hue values are selected by random.

        : param image : accepts cv2.imread() image
        : return      : saturated image

        '''
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            sat_level = secrets.choice([1,2,3])
            # print(sat_level)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            converter = ImageEnhance.Color(im_pil)
            img2 = converter.enhance(sat_level)
            saturated_image = np.asarray(img2)
            saturated_image = cv2.cvtColor(saturated_image, cv2.COLOR_RGB2BGR)
            yield saturated_image

        except Exception as e:
            logger.error(f'saturation problem  , desc : {e}')

    def perspective(self,image):
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
            rows,cols = image.shape[:2]
            pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
            pts2 = np.float32(pts1 + np.random.normal(0, 50, size=pts1.shape))
            M = cv2.getPerspectiveTransform(pts1, pts2)


            transformed_img = cv2.warpPerspective(image, M, (cols,rows))
            yield transformed_img
        except Exception as e:
            logger.error(f'Perspective problem  , desc : {e}')
    
    def zoom(self,image):

        try:
            zoom = secrets.choice([0.6,0.7,0.8,0.9,1.2,1.3,1.4])
            transform = A.Compose([A.augmentations.geometric.resize.Resize(self.height,self.width),
                                    A.augmentations.geometric.transforms.Affine(shear=None,p=1.0,scale=zoom)])
        
            augmented_image = transform(image=image)['image']
            yield augmented_image
        except Exception as e:
            logger.error(f'zoom problem  , desc : {e}')
    
    def translation(self,image):
        try:
            image = cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)    
            rows,cols = image.shape[:2]
            dx,dy = np.random.randint(-75,75,2)
            M = np.float32([[1,0,dx],[0,1,dy]])
            translated_img = cv2.warpAffine(image, M, (cols,rows))
            yield translated_img
        except Exception as e:
            logger.error(f'translation problem  , desc : {e}')


    @staticmethod
    def visualize(image):
        cv2.imshow('im',image)
        cv2.waitKey(0)


# if __name__ == '__main__':
#     cls1 = ClassificationAugmentation(height=512,width=512)
#     im = cv2.imread('classfication_aug_test/cat/cat1.jpg')


    
#     aug_im = cls1.zoom(im)
#     # print(aug_im)
#     aug_im = next(aug_im)

#     print(aug_im.shape)
#     cls1.visualize(aug_im)
# cls1.visualize(im)