import cv2
from tornado import concurrent
from rich.console import Console
from rich.traceback import install
from rich.progress import track

import os
import shutil
import uuid

from classification.classification_ import ClassificationAugmentation
from classification import logging_util






install()
console  = Console()
logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])

class ImageAugmentHelper(ClassificationAugmentation):
    def __init__(self,source_folder,aug_save_folder_name,train_split=.90,height=256,width=256) -> None:

        

        self.source_folder = source_folder
        self.aug_save_folder_name = aug_save_folder_name
        self.train_split = train_split
        super().__init__(height,width)


        dir_check = os.path.isdir(self.source_folder)

        if dir_check != True:
            shutil.rmtree(self.aug_save_folder_name)
            raise NotADirectoryError(f'{self.source_folder} is not a directory....')

        if type(self.train_split) != float:
            shutil.rmtree(self.aug_save_folder_name)
            raise TypeError(f'{self.train_split} is not float, Please provide train split as float. Your provided train split as {type(self.train_split)}')
        
        if self.train_split > 1.0:
            shutil.rmtree(self.aug_save_folder_name)
            raise ValueError(f'please provide "train split" between "0.5 to 1.0", Your provided train split value is : {train_split}')
        
        if not type(height) == int and type(width) == int:
            shutil.rmtree(self.aug_save_folder_name)
            raise TypeError('Please provide height and width as "INT"')
        
        if  height >= 2056:
            shutil.rmtree(self.aug_save_folder_name)
            raise ValueError('Please provide height value less than 2056')
        
        if  width >= 2056:
            shutil.rmtree(self.aug_save_folder_name)
            raise ValueError('Please provide width value less than 2056')

        if os.path.exists(f'{self.aug_save_folder_name}'):
            shutil.rmtree(self.aug_save_folder_name)
            raise NotImplementedError(f'"{self.aug_save_folder_name}" folder already exist, please change your augmentation saved folder name..')

        if not os.path.exists(f'{self.aug_save_folder_name}/train/') or not os.path.exists(f'{self.aug_save_folder_name}/train/'):
            os.makedirs(f'{self.aug_save_folder_name}/train/')
            console.print(f'[bold blue] [+] {self.aug_save_folder_name}/[bold blue] folder - created..')
            self.train_images_path = f'{self.aug_save_folder_name}/train/'

        if self.train_split < 1.0:
            if not os.path.exists(f'{self.aug_save_folder_name}/test/') or not os.path.exists(f'{self.aug_save_folder_name}/test/'):
                os.makedirs(f'{self.aug_save_folder_name}/test/')
                self.test_images_path = f'{self.aug_save_folder_name}/test/'

        
        


    def __len__(self):
        len_check = len(os.listdir(self.source_folder))
        return len_check
    
    def augmentations(self,save_raw_images=True,blur=True,blur_f=0.5,noise=True,noise_f=0.5,horizontalFlip=True,horizontalFlip_f=0.5,brightness=True,brightness_f=0.5,
                      contrast=True,contrast_f=0.5,hue=True,hue_f=0.5,saturation=True,saturation_f=0.5,zoom=True,zoom_f=0.5,
                      perspective=True,perspective_f=0.5,translation=True,translation_f=0.5,sharpen=True,sharpen_f=0.5,randomShadow=True,randomShadow_f=0.5):
        



        type_1 = {
            'save_raw' : save_raw_images,
            'blur' :blur,
            'noise' : noise,
            'horizontalFlip' : horizontalFlip,
            'brightness' : brightness,
            'contrast' : contrast,
            'hue' : hue,
            'saturation' : saturation,
            'zoom' : zoom,
            'perspective' : perspective,
            'translation' : translation,
            'sharpen' : sharpen,
            'randomShadow' : randomShadow


        }

        type_2 = {
            'blur_f' :blur_f,
            'noise_f' : noise_f,
            'horizontalFlip_f' : horizontalFlip_f,
            'brightness_f' : brightness_f,
            'contrast_f' : contrast_f,
            'hue_f' : hue_f,
            'saturation_f' : saturation_f,
            'zoom_f' : zoom_f,
            'perspective_f' : perspective_f,
            'translation_f' : translation_f,
            'sharpen_f' : sharpen_f,
            'randomShadow_f' : randomShadow_f


        }

        for types , val in type_2.items():
            if type(val) != float:
                shutil.rmtree(self.aug_save_folder_name)
                raise TypeError(f'Please provide "{types}" as  float , You provided "{types}" as {type(val)}')
             
                
            if val > 1.0:
                shutil.rmtree(self.aug_save_folder_name)
                raise ValueError(f'please provide "{types}" value  between 0.2 to 1.0 , Your provided "{types}" value is : {val}')
        
        for typess , vals in type_1.items():

            if type(vals) != bool:
                shutil.rmtree(self.aug_save_folder_name)
                raise TypeError(f'Please provide "{typess}" as bool , you provided "{typess}" as {vals}')


        total_no_of_images = None
        images = os.listdir(self.source_folder)
        console.print(f'[bold green]Total classes [/bold green]: [bold red]{len(images)}[/bold red]')
        console.print(f'[bold green]Class names[/bold green] : [bold blue]{images}[/bold blue]')

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # checking number of images in each folder
            for i in range(len(images)):

                ims = os.listdir(os.path.join(self.source_folder,images[i]))

                if total_no_of_images is None:
                    total_no_of_images = len(ims)
                
                if len(ims) == total_no_of_images:
                    pass
                else:
                    shutil.rmtree(self.aug_save_folder_name)
                    raise NotImplementedError('Number of images in each folder is not same...')
            
            
            console.print(f'[bold green]Total number of images in each folder[/bold green]: [bold red]{total_no_of_images}[/bold red]')
            train_split = int(total_no_of_images * self.train_split)
            console.print(f'[bold green]Train split[/bold green] :[bold blue] {self.train_split}[/bold blue]    |   [bold green]Train images[/bold green] : [bold blue]{train_split}[/bold blue]    |   [bold green]Test images[/bold green] : [bold blue]{total_no_of_images - train_split}[/bold blue]') 
        

            for i in range(len(images)):

                full_train_path = f'{self.train_images_path}/{images[i]}'
                os.makedirs(full_train_path)

                if self.train_split < 1.0:
                    full_test_path = f'{self.test_images_path}/{images[i]}'
                    os.makedirs(full_test_path)

                for j,im in enumerate(track(os.listdir(os.path.join(self.source_folder,images[i])),description=images[i])):

              
                    full_path = os.path.join(os.path.join(self.source_folder,images[i]),im)
                    # print(full_path)
                    # print(train_split)
                    if j+1 <= train_split:

                        # random.seed(j+1)
                
                        read_image = cv2.imread(full_path)

                        if save_raw_images:
                            try:
                                ims = cv2.resize(read_image,(self.width,self.height),interpolation=cv2.INTER_CUBIC)
                                cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}.jpg',ims)
                            except Exception as e:
                                logger.warning(f'saving raw images problem : {e}')

                        if blur:
                            
                            frac_data= int(blur_f * train_split)
                            # print(j+1,frac_data)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().blur,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_blur.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'blur problem : {e}')
                        
                        if noise:
                         
                            frac_data= int(noise_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().noise,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_noise.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'noise problem : {e}')
                                # print(f'{full_train_path}/{uuid.uuid4()}_noise.jpg')

                        if horizontalFlip:
                            frac_data= int(horizontalFlip_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().horizontalFlip,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_horizontal.jpg',aug_res)
                                
                                except Exception as e:
                                    logger.warning(f'horizontal problem : {e}')
                        
                        if brightness:
                            frac_data= int(brightness_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().randomBrightness,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_bright.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'brightness problem : {e}')

                        if contrast:
                            frac_data= int(contrast_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().randomContrast,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_contrast.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'contrast problem : {e}')

                        if hue:
                            frac_data= int(hue_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().hue,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_hue.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'hue problem : {e}')

                        if saturation:
                            frac_data= int(saturation_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().saturation,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_sat.jpg',aug_res)

                                except Exception as e:
                                    logger.warning(f'saturation problem : {e}')

                        if zoom:
                            frac_data= int(zoom_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().zoom,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_zoom.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'zoom problem : {e}')
                        
                        if perspective:
                            frac_data= int(perspective_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().perspective,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_perspective.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'perpective problem : {e}')

                        if translation:
                            frac_data= int(translation_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().translation,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_trans.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'translation problem : {e}')
                        
                        if sharpen:
                            frac_data= int(sharpen_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().sharpen,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_sharpen.jpg',aug_res)

                                except Exception as e:
                                    logger.warning(f'sharpen problem : {e}')
                        
                        if randomShadow:
                            frac_data= int(randomShadow_f * train_split)
                            if j+1 <= frac_data:
                                try:
                                    aug_res =  executor.submit(super().randomShadow,read_image)
                                    aug_res = next(aug_res.result())
                                    cv2.imwrite(f'{full_train_path}/{uuid.uuid4()}_shadow.jpg',aug_res)
                                except Exception as e:
                                    logger.warning(f'Random problem : {e}')
                        

                    if j+1 > train_split:
                        
                        shutil.copy(full_path,full_test_path)


        





# if __name__ == '__main__':
#     im = ImageAugmentHelper(source_folder='classfication_aug_test',aug_save_folder_name='tests',train_split=1,height=512,width=512)
#     # print(len(im))
#     im.augmentations()