import cv2
import streamlit as st
import numpy as np
import imutils
import os
import numpy as np
from glob import glob
from PIL import Image

import random
import yaml
import shutil
import random

from apps.utils import get_bbox_coor_by_image ,map_data , image_format_change
from object_detection_new.txt_reader_rect import RectAugmentation
from utils.load_yaml import load_yaml
from utils.data_analyser import DataAnalyser



def view_logo():
    st.set_page_config(
    page_title="Image Augmentations",
    page_icon="🚀",)
    st.image("images/logo.png")


def start_project():
    project_names = st.text_input("**Provide a name for the project**",value="",)
    if st.button("Next ➡️"):
        if project_names is "":
            st.warning("⚠️ Please provide a project name..")
        else:
            if os.path.exists(project_names):
                st.warning(f"project name already exist, Please choose a different project name.")
            else:
                st.session_state.project_name = project_names
                st.success(f"Successfully saved project name as -- **{st.session_state.project_name}** --")



def augmentation_type():
    aug_type = st.selectbox("**Project Type**",
                     ['Detection', 'Segmentation', 'Classification'],)
    
    if st.button("Continue ➡️"):
        if aug_type == "Detection":  ### exception
            st.session_state.model = aug_type
            st.success(f"✅ Selected project type  -- **{st.session_state.model}** --")
        else:
            st.info(f"ℹ️ Selected project type -- **{aug_type}** -- is not supported yet")
            st.session_state.model = None
            



def file_upload(detection):
    
    upload,plot = st.columns([7,5])
    with upload:
        # uploaded_files = st.file_uploader("**Upload Images and Txts**", accept_multiple_files=True, type=["png","jpg","jpeg", "tif","txt"],)
        uploaded_files = st.text_input(label="Enter Full Path Of Your Data..")

        if st.button("Upload 📤"):
            if uploaded_files == "":
                st.warning("⚠️ Please provide full path..")
            else: 
                if os.path.exists(uploaded_files):
                    st.session_state.data_path = uploaded_files
        
                    st.session_state.button_pressed = True
                    image_format_change(folder=st.session_state.data_path)

                    all_images = glob(f"{st.session_state.data_path}/*.jpg")
                    st.session_state.total_image = len(all_images) # saving total image len
                    all_txt =  list(map(lambda x : os.path.splitext(x)[0] + '.txt',all_images))
                    

                    classes_ = glob(f"{st.session_state.data_path}/classes.txt")
                    if classes_ != []:
                        with open(classes_[0]) as f:
                            data = f.readlines()
                            data = list(map(lambda x : x.replace("\n",""),data))
                            st.write(f"**Class names : {data}**")
                        
                        for image_file, txt_file in zip(all_images, all_txt):
                            if os.path.exists(image_file) and os.path.exists(txt_file):
                                pass
                            else:
                                st.warning(f"⚠️ Either **{image_file}**  or  **{txt_file}** does not exist.")
                                st.session_state["button_pressed"] = False
                        
            
                    else:
                        st.error("⚠️ **classes.txt** not found!!")
                        st.session_state["button_pressed"] = False
                    
                    if st.session_state["button_pressed"]:
                        st.success(f"🚀 {len(os.listdir(uploaded_files))} file(s) uploaded successfully!")
                        analyse = DataAnalyser(st.session_state.data_path,is_json=False)
                        save = f"{st.session_state.project_name.strip()}_{st.session_state.model}"
                        analyse.analyse(save_folder=save)

                        print(f"upload dir : {st.session_state.data_path}")
                        ### select a random photo and apply augmentation
                        RANDOM_IMAGE = random.choice(glob(f"{st.session_state.data_path}/*.jpg"))
                        BBOX_COOR = get_bbox_coor_by_image(RANDOM_IMAGE,detection)

                        img = Image.open(RANDOM_IMAGE)
                        st.session_state["random_image"] = img # random selected image
                        st.session_state["bbox_coor"] = BBOX_COOR # and its box coor

                else:
                    st.error("⚠️ Path Does not exist, Please provide full path..")
                    st.session_state.button_pressed = False

    with plot:
        if os.path.exists(f"plots/{st.session_state.project_name.strip()}_{st.session_state.model}.png"):
            st.write(f"-- **Labels Distribution** --")
            st.image(f"plots/{st.session_state.project_name.strip()}_{st.session_state.model}.png",width=1000)


def plot():
    analyse = DataAnalyser(st.session_state.project_name.strip(),is_json=False)
    save = f"{st.session_state.project_name.strip()}_{st.session_state.model}"
    analyse.analyse(save_folder=save)
       

def image_resize():
    st.markdown("### --- Image resize ---")
    image_col, resize_col  = st.columns([5,5])

    with image_col:
        resize = st.selectbox("Resize Image",
                     ['Custom Size','Keep Same Size', 'Keep Aspect Ratio'],)
        
   
        if resize == "Custom Size":
            height = st.number_input('Height',value=640)
            width = st.number_input('Width',value=640)
            st.session_state['size'] = (int(height),int(width))
        elif resize == "Keep Aspect Ratio":
            height = st.number_input('Height',value=640)
            width = "keep-aspect-ratio"
            st.session_state['size'] = (int(height),width)
        else:
            st.session_state['size'] = ("keep_original_image_height","keep_original_image_width")
    
    with resize_col:
         
        val = st.session_state['size']
        
        
        if val is not None:

            if val[0] == "keep_original_image_height" and val[1] == "keep_original_image_width":
           
                im = np.array(st.session_state.random_image) # from oil fromat for array
                h,w,c = im.shape
                st.session_state['size'] = (int(h),int(w))
                st.session_state['resize_image'] = im

            if type(val[0]) == int and type(val[1]) == int and val[0] >= 28 and val[1] >= 28:
                resize_image = cv2.resize(np.array(st.session_state.random_image),(val[0],val[1]),)
                st.session_state['resize_image'] = resize_image
         

            if val[1] == "keep-aspect-ratio" and val[0] >= 28:
        
                resize_image = imutils.resize(np.array(st.session_state.random_image),height=val[0])
                st.session_state['resize_image'] = resize_image

            
            
            st.image(st.session_state.resize_image)

def split_size():
    st.markdown("### --- Split Data ---")
    size_col , count_col = st.columns([7,5])
    with size_col:
        train_split = st.slider(f"**Train split size**", 0.10, 1.0,step=0.05,value=0.80)
        st.session_state.train_split = train_split
    with count_col:
        if st.session_state.total_image is not None and st.session_state.train_split is not None:
            
            st.markdown(f'<span style="color:red; font-size:20px; ">Total Images -> {st.session_state.total_image} </span> ', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#48ed1f; font-size:20px">Train Images -> {round(st.session_state.total_image*st.session_state.train_split)}</span> ', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#12ebff; font-size:20px">Test Images -> {round(st.session_state.total_image-(st.session_state.total_image*st.session_state.train_split))}</span> ', unsafe_allow_html=True)
            


# multiselect aug
def multiselect():
    options = st.multiselect("**Select augmentations you want to apply on you images**",
                             st.session_state.aug)
    
    if st.button("Fire Up augmentations"):
        if options != []:
            st.session_state.selectd_aug = options
            st.session_state.fire_up = True

   
        else:
            st.warning("⚠️ Please select augmentations")
            st.session_state.selectd_aug = None
            st.session_state.fire_up = False
        
    if st.session_state.fire_up:
        st.info('**NOTE : IF YOU WANT TO REMOVE ANY AUGMENTATIONS, PLEASE USE "X" ON SELECTED AUGMENTATION (ABOVE), THEN PRESS --FIRE UP AUGMENTATIONS--**')
            



def boilerplate(function,type='blur',low=0.0,high=10.0,step=0.1,max_limit=10,min_limit=None,format=None,value=None,msg=None,show_radio=True):
    st.markdown(f"### {type}")
    image_col,  aug_col  = st.columns([5,5])
    with aug_col:
            if format is not None:
                format = format

            if show_radio:  
                status = st.radio(f"Select {type} low and high value ", ('Low Value', 'High Value'))
            # print(value)
            max_val  = st.slider(f"Select {type} level", low, high,step=step,format=format,value=value)
            


            if msg is not None:
                st.markdown(f"##### {msg}")
            if max_limit is not None:
                if max_val >= max_limit:
                    st.warning(f"Too much {type} will reduce model performance.")
            
            if min_limit is not None:
                if min_limit >= max_val:
                    st.warning(f"Too much {type} will reduce model performance.")
    

            if show_radio:
                if status == "Low Value":
                    if (max_val) != st.session_state[f'{type}_low']:
                        st.session_state[f'{type}_low'] = max_val
                    
                        
                
                if status == "High Value":
                    if (max_val) != st.session_state[f'{type}_high']:
                        st.session_state[f'{type}_high'] = max_val
                    
                        

            if (max_val) != st.session_state[f'{type}']:
                st.session_state[f"{type}_im"] = st.session_state.resize_image.copy()
                st.session_state[f'{type}'] = max_val
                st.session_state[f'{type}_value_ch'] = True
            else:
                 st.session_state[f'{type}_value_ch'] = False
            
            # print
            if st.session_state[f'{type}_low'] != None:
                st.write(f":blue[**{type} low value** : {st.session_state[f'{type}_low']}]")
            # else:
            #     st.warning(f"⚠️ Please select {type}_low value.")

            if st.session_state[f'{type}_high'] != None:
                st.write(f":green[**{type} high value** : {st.session_state[f'{type}_high']}]")
                # value = st.session_state[f'{type}_high']
            if st.session_state[f'{type}_high'] ==None:
                st.warning(f"⚠️ Please select {type}_high value.")
            
            percentage = st.slider(f":red[**Fraction of data you want to use for {type} augmentation**]",0.0,1.0,value=0.30,step=0.05)
            st.session_state[f"{type}_%"] = percentage
            
            train_len = round(st.session_state.total_image*st.session_state.train_split)
            data = round(train_len*st.session_state[f"{type}_%"])

            st.info(f"Out of **{train_len}** train images, **{data}** images will affected by {type}.")
            st.session_state.train_split

    with image_col:
            if st.session_state[f"{type}_im"] is None and st.session_state[f'{type}'] is None :
                
                st.session_state[f'{type}'] = 0.00

                augs = function(st.session_state.resize_image,st.session_state.bbox_coor,st.session_state['size'][0],st.session_state['size'][1],st.session_state[f'{type}'])
                st.session_state[f"{type}_im"] = list(augs)[0][1]

            if st.session_state[f'{type}_value_ch']:
                augs = function(st.session_state[f"{type}_im"],st.session_state.bbox_coor,st.session_state['size'][0],st.session_state['size'][1],st.session_state[f'{type}'])
                st.session_state[f"{type}_im"] = list(augs)[0][1]

            st.image(st.session_state[f"{type}_im"])





def combine_user_data():
    data = {}
    for augs in st.session_state.aug:
        if augs in st.session_state.selectd_aug:
            augs_ = map_data(aug_name=augs)
           
            if augs_ is not  None:
                augs_ = augs_.lower()
                data[augs_] = {"stat" : True}
                data[augs_][f"{augs_}_range"] = [st.session_state[f"{augs_}_low"],st.session_state[f"{augs_}_high"]]
                data[augs_][f"{augs_}_%"] = st.session_state[f"{augs_}_%"]
            else:
                augs = augs.lower()
                data[augs] = {"stat" : True}
                data[augs][f"{augs}_range"] = [st.session_state[f"{augs}_low"],st.session_state[f"{augs}_high"]]
                data[augs][f"{augs}_%"] = st.session_state[f"{augs}_%"]
        else:
            
            augs_ = map_data(aug_name=augs)
            if augs_ is not  None:
                data[augs_.lower()] = {"stat" : False}
                data[augs_.lower()][f"{augs_.lower()}_%"] = 0.0
            else:
                data[augs.lower()] = {"stat" : False}
                data[augs.lower()][f"{augs.lower()}_%"] = 0.0

    yaml_filename = 'config.yaml'
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    # print(f"The configs have been written to {yaml_filename}")



def annotate():
    yaml_data = load_yaml("config.yaml") #loading yaml
    im_H = st.session_state['size'][0] # custom size
    im_W = st.session_state['size'][1] # custom size 
    split = st.session_state.train_split # train split
    project_save_folder = f"{st.session_state.project_name.strip()}_{st.session_state.model}" 
    annotation_folder = st.session_state.data_path
    rect_aug = RectAugmentation(project_save_folder)
    rect_aug.Image_augmentation(folder=annotation_folder,
                                train_split=split,
                                image_height=im_H,
                                image_width=im_W,
                                blur=yaml_data["blur"]["stat"],  blur_f=yaml_data["blur"]["blur_%"],

                                 motionBlur= yaml_data["motionblur"]["stat"] , motionBlur_f= yaml_data["motionblur"]["motionblur_%"],

                                 rotate=yaml_data["rotate"]["stat"], rotate_f = yaml_data["rotate"]["rotate_%"], 

                                 noise=yaml_data["noise"]["stat"], noise_f=yaml_data["noise"]["noise_%"],

                                 perspective=yaml_data["perspective"]["stat"], perspective_f =yaml_data["perspective"]["perspective_%"],

                                 affine=yaml_data["affine"]["stat"], affine_f=yaml_data["affine"]["affine_%"],

                                 brightness=yaml_data["bright&dark"]["stat"], brightness_f=yaml_data["bright&dark"]["bright&dark_%"],

                                 exposure=yaml_data["exposure"]["stat"] , exposure_f= yaml_data["exposure"]["exposure_%"],
   
                                 removesaturation=yaml_data["remove_saturation"]["stat"], removesaturation_f= yaml_data["remove_saturation"]["remove_saturation_%"],

                                 contrast=yaml_data["contrast"]["stat"], contrast_f=yaml_data["contrast"]["contrast_%"],

                                 shear=yaml_data["shear"]["stat"], shear_f=yaml_data["shear"]["shear_%"], 

                                 rotate90=yaml_data["rotate90"]["stat"], rotate90_f =yaml_data["rotate90"]["rotate90_%"],

                                 image_cutout = yaml_data["box"]["stat"], image_cutout_f=yaml_data["box"]["box_%"],
                                    
                                 Hflip= yaml_data["hflip"]["stat"] , Hflip_f= yaml_data["hflip"]["hflip_%"],

                                 Vflip= yaml_data["vflip"]["stat"] , Vflip_f= yaml_data["vflip"]["vflip_%"],
                                    
                                 temperature_change= yaml_data["temperature"]["stat"], temperature_change_f=yaml_data["temperature"]["temperature_%"],)
    

  
    st.success(f"##### Augmentations saved in --> {os.path.basename(os.getcwd())}/{project_save_folder} <-- folder ")
    st.success(f"##### Please refresh the page to start new augmentations..")

    
    total_im = len(os.listdir(f"{project_save_folder}/train/images"))
    st.markdown(f"##### Total Augmented Images --> {total_im} ")
    
    if os.path.exists(f"{project_save_folder}/test/images"):
        total_im_test = len(os.listdir(f"{project_save_folder}/test/images"))
        st.markdown(f"##### Total Test Images  --> {total_im_test} ")



def cleanup():
    os.remove("config.yaml")
    shutil.rmtree("rect_augmentation")
    st.session_state.button_pressed = False
    st.session_state.apply = False
    st.session_state.aug_select = False
    st.session_state.augment_data = False
    st.session_state.project_name = None
    st.session_state.model = None
    st.session_state.total_image = None
    st.session_state['random_image'] = None
    st.session_state.selectd_aug = None


# additional info
    
def sidebar():
    if st.session_state.project_name is not None:
        st.sidebar.write(f"**Project Name :** {st.session_state.project_name}")
    if st.session_state.model is not None:
        st.sidebar.write(f"**Project type :** {st.session_state.model}")
        st.sidebar.write(f"**Project Save Directory :** {st.session_state.project_name.strip()}_{st.session_state.model}")


def creator_name():
    st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 5px;
        background-color: #f1f1f1;
        color: #333;
        font-size:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    st.markdown(
        """
        <div class="footer">
            Created By Souvik Saha @2024
        </div>
        """,
        unsafe_allow_html=True,
    )