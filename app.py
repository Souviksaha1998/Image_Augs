import streamlit as st

from object_detection_new.txt_reader_rect import RectAugmentation
from object_detection_new.augmentation_rect import ImageAugmentationBox
from apps.modules import *
from apps.sessions import sessions




view_logo()
creator_name()
sessions()
detection = RectAugmentation()
augmentations_detection = ImageAugmentationBox()
start_project()
sidebar()


if st.session_state["project_name"] is not None:
    augmentation_type()
    if st.session_state["model"] is not None:
        file_upload(st.session_state.project_name.strip(),detection)
        if st.session_state["button_pressed"]:
            image_resize()
            split_size()
            if st.button("Apply"):
                st.session_state.apply = True
            if st.session_state.apply:
                multiselect()
                
                if st.session_state["selectd_aug"] is not None and st.session_state["selectd_aug"] != []:
                    if "Blur" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_blur,type="blur",low=0.0,high=15.0,step=0.1,max_limit=7)
                    if "Noise" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_noise,type="noise",low=0.0000, high=0.25,step=0.0001,max_limit=0.10,format="%.5f")
                    if  "Rotate Upto 20deg" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_rotate,type="rotate",low=0.0,high=20.0,step=1.0,max_limit=15,format=None,value=None,msg="rotation will be +/-") 
                    if  "Affine" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_affine,type="affine",low=0.5,high=2.0,step=0.1,max_limit=1.50,value=1.0) 
                    if  "Exposure" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_exposure,type="exposure",low=-100,high=100,step=5,max_limit=50,min_limit=-40,value=0) 
                    if  "Remove_saturation" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_removeSaturation,type="remove_saturation",low=1,high=10,step=1,max_limit=None,min_limit=None,value=0) 
                    if  "Contrast" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_contrast,type="contrast",low=0.3,high=2.0,step=0.05,max_limit=1.50,min_limit=0.70,value=1.0) 
                    if  "Shear" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_shear,type="shear",low=-25,high=25,step=1,max_limit=12,min_limit=-12,value=1)
                    if  "Rotate90" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_rotate90,type="rotate90",low=-90,high=90,step=90,max_limit=None,min_limit=None,value=0,show_radio=False,msg="Rotation will be +/- 90deg")
                    if  "Boxes On Image" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_cutOut,type="box",low=1,high=10,step=1,max_limit=5,min_limit=None,value=1,show_radio=True) 
                    if  "Perspective" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_perspective_transform,type="perspective",low=0.05,high=0.20,step=0.001,max_limit=0.5,min_limit=None,value=None,show_radio=True,format="%.3f",msg="Apply perspective only if necessary..")
                    if  "Motion Blur" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_motionBlur,type="motionblur",low=-90,high=90,step=5,max_limit=45,min_limit=-45,value=0,show_radio=True,)
                    if  "Bright&Dark" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_bright_dark,type="bright&dark",low=0.5,high=2.0,step=0.01,max_limit=1.20,min_limit=0.85,value=1.0,show_radio=True,)
                    if  "Color_Temperature" in st.session_state.selectd_aug:
                        boilerplate(augmentations_detection.image_change_colorTemperature,type="temperature",low=1100,high=15000,step=100,max_limit=None,min_limit=None,value=6500,show_radio=True,msg="Note : Too much change in color temperature will reduce model performance") 
                    if  "Horizontal Flip" in st.session_state.selectd_aug:  
                        boilerplate(augmentations_detection.image_HFlip,type="hflip",low=0,high=1,step=1,max_limit=None,min_limit=None,value=0,show_radio=False,msg="Keep Hflip value to 1")
                    if  "Vertical Flip" in st.session_state.selectd_aug: 
                        boilerplate(augmentations_detection.image_VFlip,type="vflip",low=0,high=1,step=1,max_limit=None,min_limit=None,value=0,show_radio=False,msg="Keep Vflip value to 1") #need to chnage
                
                    if st.button("Augment Data"):
                        if st.session_state.button_pressed:
                            st.session_state.augment_data = True

                    if st.session_state.augment_data:
                        combine_user_data()
                        annotate()
                        cleanup()





    
         









