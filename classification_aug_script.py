from classification.classification_combined import ImageAugmentHelper




### PARAMS ###
source_folder = 'classfication_aug_test'
aug_saved_folder = 'test_classification_augs'
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
