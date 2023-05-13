from instance_seg.json_reader_poly import PolygonAugmentation




################# image height and width combination  ##################

# first combination --> for custom image size
#  image_height = < custom image size > ; 640
#  image_width = < custom image size > ; 320


# second combination --> keep aspect ratio of the image
#  image_height = 640
#  image_width  = 'keep_aspect_ratio_False'

# Third combination --> keeping original image height and width
#  image_height = < keep_original_image_height >
#  image_width = < keep_original_image_width >


#### yolo ####
# if yolo False then it will normalize all images and save it as txt , if false augmentations will be saved as json.


annotation_folder = 'indian_data_together'
new_aug_saved_folder = 'test_det_1'
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

                                 motionBlur= True , motionBlur_f= 0.5,

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