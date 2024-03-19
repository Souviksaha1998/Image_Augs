import streamlit as st



def sessions():
    if 'button_pressed' not in st.session_state:
        st.session_state['random_image'] = None
        st.session_state.aug  = ["Blur","Noise","Rotate Upto 20deg","Rotate90","Affine","Bright&Dark","Exposure","Remove_saturation","Contrast","Shear","Color_Temperature","Perspective","Boxes On Image","Vertical Flip","Horizontal Flip","Motion Blur"]
        st.session_state.selectd_aug = None

        st.session_state['blur'] = None
        st.session_state['blur_%'] = None
        st.session_state['blur_low'] = None
        st.session_state['blur_high'] = None
        st.session_state['blur_im'] = None
        st.session_state['blur_value_ch'] = False

        st.session_state['noise'] = None
        st.session_state['noise_%'] = None
        st.session_state['noise_low'] = None
        st.session_state['noise_high'] = None
        st.session_state['noise_im'] = None
        st.session_state['noise_val_ch'] = False

        st.session_state['rotate'] = None
        st.session_state['rotate_%'] = None
        st.session_state['rotate_low'] = None
        st.session_state['rotate_high'] = None
        st.session_state['rotate_im'] = None
        st.session_state['rotate_val_ch'] = False

        st.session_state['affine'] = None
        st.session_state['affine_%'] = None
        st.session_state['affine_low'] = None
        st.session_state['affine_high'] = None
        st.session_state['affine_im'] = None
        st.session_state['affine_val_ch'] = False

        st.session_state['bright&dark'] = None
        st.session_state['bright&dark_%'] = None
        st.session_state['bright&dark_low'] = None
        st.session_state['bright&dark_high'] = None
        st.session_state['bright&dark_im'] = None
        st.session_state['bright&dark_val_ch'] = False

        st.session_state['exposure'] = None
        st.session_state['exposure_%'] = None
        st.session_state['exposure_low'] = None
        st.session_state['exposure_high'] = None
        st.session_state['exposure_im'] = None
        st.session_state['exposure_val_ch'] = False

        st.session_state['remove_saturation'] = None
        st.session_state['remove_saturation_%'] = None
        st.session_state['remove_saturation_low'] = None
        st.session_state['remove_saturation_high'] = None
        st.session_state['remove_saturation_im'] = None
        st.session_state['remove_saturation_val_ch'] = False

        st.session_state['contrast'] = None
        st.session_state['contrast_%'] = None
        st.session_state['contrast_low'] = None
        st.session_state['contrast_high'] = None
        st.session_state['contrast_im'] = None
        st.session_state['contrast_val_ch'] = False

        st.session_state['shear_%'] = None
        st.session_state['shear'] = None
        st.session_state['shear_low'] = None
        st.session_state['shear_high'] = None
        st.session_state['shear_im'] = None
        st.session_state['shear_val_ch'] = False

        st.session_state['temperature'] = None
        st.session_state['temperature_%'] = None
        st.session_state['temperature_low'] = None
        st.session_state['temperature_high'] = None
        st.session_state['temperature_im'] = None
        st.session_state['temperature_val_ch'] = False

        st.session_state['perspective'] = None
        st.session_state['perspective_%'] = None
        st.session_state['perspective_low'] = None
        st.session_state['perspective_high'] = None
        st.session_state['perspective_im'] = None
        st.session_state['perspective_val_ch'] = False

        st.session_state['box'] = None
        st.session_state['box_%'] = None
        st.session_state['box_low'] = None
        st.session_state['box_high'] = None
        st.session_state['box_im'] = None
        st.session_state['box_val_ch'] = False

        st.session_state['hflip'] = None
        st.session_state['hflip_%'] = None
        st.session_state['hflip_low'] = None
        st.session_state['hflip_high'] = None
        st.session_state['hflip_im'] = None
        st.session_state['hflip_val_ch'] = False

        st.session_state['vflip'] = None
        st.session_state['vflip_%'] = None
        st.session_state['vflip_low'] = None
        st.session_state['vflip_high'] = None
        st.session_state['vflip_im'] = None
        st.session_state['vflip_val_ch'] = False

        st.session_state['motionblur'] = None
        st.session_state['motionblur_%'] = None
        st.session_state['motionblur_low'] = None
        st.session_state['motionblur_high'] = None
        st.session_state['motionblur_im'] = None
        st.session_state['motionblur_val_ch'] = False

        st.session_state['rotate90'] = None
        st.session_state['rotate90_%'] = None
        st.session_state['rotate90_low'] = None
        st.session_state['rotate90_high'] = None
        st.session_state['rotate90_im'] = None
        st.session_state['rotate90_val_ch'] = False

        st.session_state['size'] = None
        st.session_state['resize_image'] = None
        st.session_state['bbox_coor'] = None

        st.session_state.button_pressed = False
        st.session_state.apply = False
        st.session_state.aug_select = False
        st.session_state.augment_data = False
        st.session_state.train_split = .80
        st.session_state.project_name = None
        st.session_state.model = None
        st.session_state.total_image = None
        st.session_state.fire_up = False
        st.session_state.data_path = None