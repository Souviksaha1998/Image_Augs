
# Image Augmentations 🚀
***
<img src='images\logo.png'>

***
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/Souviksaha1998/Image_augmentations">

<img alt="GitHub code size in bytes" src="https://static.pepy.tech/personalized-badge/image-augs?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads">

<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Souviksaha1998/image_augmentations?style=social">
<img alt="GitHub followers" src="https://img.shields.io/github/followers/Souviksaha1998?style=plastic">

***

This is a augemntation tool for Object Detection , Image classification and Instance Segmentation , it can perform 14 annotations. The important ones are rotation, affine, zooming in and out, noise, and blur. The augmentations were applied to a fraction of the data (40 - 50 percent of the images out of 100 can be augmented).When rotating or zooming in and out, the bounding box coordinates will also change as the image is rotated or zoomed.

***
##  Code Starts from here 

### 👩🏼‍💻Create a folder first, inside that folder keep your image annotation folder 👩🏼‍💻
***
<img src='images\3.jpg'>

### 👩🏼‍💻 Create a virtual environment 👩🏼‍💻

```python

pip install virtualenv
#name your environment
python3 -m venv <your env name>

#activate the environment --> for linux user
source <your env name>/bin/activate

#activate the environment  --> for windows user
<your env name>/Source/activate.ps1

```

### Installation (for pip installation) 🚀

```python
pip install image_augs
```

## After installation 🎯
***

**Create a .py script inside your created folder**

**This Script is for OBJECT DETECTION**

```python
#import these modules in your created <scriptname>.py file


from object_detection_new.txt_reader_rect import RectAugmentation

################# image height and width combination  ##################

# first combination --> for custom image size
#  image_height = < custom image size > ; 640
#  image_width = < custom image size > ; 320


# second combination --> keep aspect ratio of the image
#  image_height = 640
#  image_width  = 'keep_aspect_ratio_False'

# Third combination --> keeping original image height and width
#  image_height = 'keep_original_image_height'
#  image_width = 'keep_original_image_width'



annotation_folder = 'your folder'
new_aug_saved_folder = 'new saved folder'
train_split = 0.90
image_H = 640  #check above for height and width setting
image_W = 'keep_aspect_ratio'


rect_aug = RectAugmentation(new_aug_saved_folder)

rect_aug.Image_augmentation(annotation_folder,
                                 
                                train_split=train_split,
                                 image_height= image_H,
                                 image_width= image_W,


                                 blur=False,  blur_f=0.8,

                                 motionBlur= False , motionBlur_f= 0.8 ,

                                 rotate=False, rotate_f = 0.8, 

                                 noise=False, noise_f=0.8,

                                 perspective=False, perspective_f = 0.8,

                                 affine=False, affine_f=0.8,

                                 brightness=False, brightness_f=0.8,
                                    
                                 hue=False, hue_f=0.8,

                                 removesaturation=False, removesaturation_f=0.8,

                                 contrast=False, contrast_f=0.8,

                                 upflip=False, upflip_f=0.8,

                                 shear=False, shear_f=0.8, 

                                 rotate90=False, rotate90_f =0.8,

                                 blur_and_noise=False, blur_and_noise_f=0.8,

                                 image_cutout = False, image_cutout_f=0.8,
                                    
                                 mix_aug= False, mix_aug_f=0.8, 
                                    
                                 temperature_change= False, temperature_change_f=0.8,  # change color temperature from cool to warm color

                                 weather_change=True,weather_change_f=0.8), # add rain , fog , snow in your images
                               
                                

#results will be saved in < your given folder >
```
***

**This Script is for INSTANCE SEGMENTATION**

```python
#import these modules in your created <scriptname>.py file
from instance_seg.json_reader_poly import PolygonAugmentation


################# image height and width combination  ##################

# first combination --> for custom image size
#  image_height = < custom image size > ; 640
#  image_width = < custom image size > ; 320


# second combination --> keep aspect ratio of the image
#  image_height = 640
#  image_width  = 'keep_aspect_ratio_False'

# Third combination --> keeping original image height and width
#  image_height = 'keep_original_image_height'
#  image_width = 'keep_original_image_width'


#### yolo ####
# if yolo False then it will normalize all images and save it as txt , if false augmentations will be saved as json.

annotation_folder = 'your data'
new_aug_saved_folder = 'new saved dataset name'
train_split = 0.70
image_H = 640  #check above for height and width setting
image_W = 'keep_aspect_ratio'
yolo = True


im_aug_helper = PolygonAugmentation(aug_save_folder_name=new_aug_saved_folder,
                                    yolo=yolo)

im_aug_helper.Image_augmentation(annotation_folder,
                                 
                                 train_split=train_split,
                                 image_height= image_H,
                                 image_width= image_W,


                                 blur=True,  blur_f=0.8,

                                 motionBlur= False , motionBlur_f= 0.5,

                                 rotate=True, rotate_f = 0.8, 

                                 noise=True, noise_f=0.6,

                                 perspective=True, perspective_f = 0.6,

                                 affine=True, affine_f=0.6,

                                 brightness=True, brightness_f=0.6,
                                    
                                 hue=True, hue_f=0.6,

                                 removesaturation=True, removesaturation_f=0.6,

                                 contrast=True, contrast_f=0.6,

                                 upflip=True, upflip_f=0.8,

                                 shear=True , shear_f=0.7, 

                                 rotate90=True, rotate90_f =1.0,

                                 blur_and_noise=True, blur_and_noise_f=0.6,

                                 image_cutout = True, image_cutout_f=0.6,
                                    
                                 mix_aug=True, mix_aug_f=0.7,
                                    
                                 temperature_change=True, temperature_change_f=0.5,
                                 
                                 weather_change=True,weather_change_f=0.3)
#results will be saved in < your given folder >
***

```

**This Script is for IMAGE CLASSIFICATION**

```python
#import these modules in your created <scriptname>.py file
from classification.classification_combined import ImageAugmentHelper


### PARAMS ###
source_folder = '<source folder>'
aug_saved_folder = '<augmentation saved folder>'
train_split = 0.5
image_height = 512
image_width = 512

classification_aug = ImageAugmentHelper(source_folder=source_folder,
                                        aug_save_folder_name=aug_saved_folder,
                                        train_split=train_split,
                                        height=image_height,
                                        width=image_width)


classification_aug.augmentations(

    save_raw_images=True,

    blur=True, blur_f=1.0,

    noise=True,noise_f=1.0,

    horizontalFlip=True, horizontalFlip_f=1.0,

    brightness=True, brightness_f=1.0,

    contrast=True, contrast_f=1.0,

    hue=True, hue_f=1.0,

    saturation=True, saturation_f=1.0,

    zoom=True, zoom_f=1.0,

    perspective=True, perspective_f=1.0,

    translation=True, translation_f=1.0,

    sharpen=True, sharpen_f=1.0,
    
    randomShadow=True, randomShadow_f=1.0
)
```
***
Use github to clone [image_augmentations](https://github.com/Souviksaha1998/Image_augmentations) repo 🖥️
Use instanceSeg_aug_script.py / classification_aug_script.py / objectDetection_augScript.py according to your needs.

***


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)


