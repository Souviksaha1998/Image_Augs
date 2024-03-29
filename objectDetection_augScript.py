from object_detection_new.txt_reader_rect import RectAugmentation





################# image height and width combination  ##################

# first combination --> for custom image size
#  image_height = < custom image size > ; 640
#  image_width = < custom image size > ; 320


# second combination --> keep aspect ratio of the image
#  image_height = 640
#  image_width  = 'keep_aspect_ratio_True'

# Third combination --> keeping original image height and width
#  image_height = < keep_original_image_height >
#  image_width = < keep_original_image_width >



annotation_folder = 'lp_dataset_subset'
new_aug_saved_folder = 'lp_dataset_subset_new'
train_split = 0.60
image_H = 640  #check above for height and width setting
image_W = 'keep_aspect_ratio'


rect_aug = RectAugmentation(new_aug_saved_folder)

rect_aug.Image_augmentation(annotation_folder,
                                 
                                train_split=train_split,
                                 image_height= image_H,
                                 image_width= image_W,


                                 blur=True,  blur_f=0.8,

                                 motionBlur= True , motionBlur_f= .8 ,

                                 rotate=True, rotate_f = 0.8, 

                                 noise=True, noise_f=0.8,

                                 perspective=True, perspective_f = 0.8,

                                 affine=True, affine_f=0.8,

                                 brightness=True, brightness_f=0.8,
   
                                 removesaturation=True, removesaturation_f=0.8,

                                 contrast=True, contrast_f=0.8,

                                 shear=True, shear_f=0.8, 

                                 rotate90=True, rotate90_f =0.8,

                                 image_cutout = True, image_cutout_f=0.8,
                                    
                                 Hflip= True , Hflip_f= 0.5,

                                 Vflip= True , Vflip_f= 0.5,
                                    
                                 temperature_change= True, temperature_change_f=0.8,  # change color temperature from cool to warm color

                            ), # add rain , fog , snow in your images
                               