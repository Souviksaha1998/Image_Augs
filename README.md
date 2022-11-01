# Image Augmentations 🚀
***
<img src='images\2.jpg'>

***
This is a **bounding box level image augmentation tool**, it can perform 14 annotations. The important ones are rotation, affine, zooming in and out, noise, and blur. The augmentations were applied to a fraction of the data (40 - 50 percent of the images out of 100 can be augmented).When rotating or zooming in and out, the bounding box coordinates will also change as the image is rotated or zoomed.

***
## Code Starts from here
***

## 👩🏼‍💻 Create a folder first, inside that folder keep your image annotation folder 👩🏼‍💻 
<img src='images\3.jpg'>

***
## 🧑🏼‍💻 Create a virtual environment  🧑🏼‍💻 

```python

pip install virtualenv
#name your environment
python3 -m venv <your env name>

#activate the environment --> for linux user
source <your env name>/bin/activate

#activate the environment  --> for windows user
<your env name>/Source/activate.ps1

```

## 🚀 Installation (for pip install) 🚀

```python
pip install image_augs
```

## 🎯 After installation 🎯 

**create a .py script inside your created folder**

```python
#import these modules in your created <scriptname>.py file
from image_augs import utils_py
from image_augs import converter_for_txtToYolo , combined
from image_augs import draw_bbox

# first step -- >   create a destination folder to save augmented results.
saved_folder_name =  utils_py.folder_creation("<give any folder name>") 
    
#2nd step --> # give your image annotation folder(images , .txt , classes.txt)
output = converter_for_txtToYolo.converter('<give your image annotation folfer path>',keep_aspect_ratio=True,resize_im=640)
    
#3rd step -->
#different augmentation and their fraction  -- > 100 images dataset ,  if True and fraction =0.3 --> it will take random 30 images from your dataset
#raw images True means, it will save raw images too.
    
dicc = combined.main(folder=saved_folder_name,

    raw_images_ok=True,
    train_test_split=0.10,
    blurs=True,blur_f=.2,
    noise=True,noise_f=.5,
    NB=True,NB_f=.5,
    hue=True,hue_f=.5,
    sat=True,sat_f=.5,
    bright=False,bright_f=.5,
    contrast=False , contrast_f=0.5,
    rotation=True,rotation_f=0.5,
    zoom=False,zoom_f=.8,
    affine=False,affine_f=0.7,
    translation=False,translation_f= 0.7,
    vertical_flip=False,vertical_f=0.2)

#results will be saved in < your given folder >
```

## 🖥️ Github (if you clone the repo from github) 🖥️ 

Use github to clone [Image_augmentations](https://github.com/Souviksaha1998/Image_augmentations) repo.

```python
git clone git@github.com:Souviksaha1998/Image_augmentations.git
cd Image_augmentations

```
## 🧑🏼‍💻 change in hyperparameters.ini file  🧑🏼‍💻 
***
**open hyperparameters.ini file**

<img src='images\4.jpg'>

- give your source folder name (image annotation folder)
- destination folder, where you want to save your results , just provide any folder name
- resize image, default 640. Keep aspect ratio will  keep one side's (Height or width) same according to your resize image
- train test split of your data
- augmentations, if *True*, it will apply those effects.
- percentage of data should use for augmentation.. 0.5 --> 50% of total data , 0.1 --> 10% of total data

## 🚀 Now, run this file 🚀

```python
python mainfile.py --configFile hyperparameters.ini
#results will be saved in destination folder
```
## Terminal output 🧑🏼‍💻
<img src='images\1.png'>

### To view the augmented images

```python
aug_image = draw_bbox.bbbox_viewer('<image_path_from_train/test_folder>','<ids.pickle file path from pickle_files folder>')
cv2.imshow(f'aug_image',aug_image)
cv2.waitKey(0)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

- [@SouvikSaha](https://github.com/Souviksaha1998)
