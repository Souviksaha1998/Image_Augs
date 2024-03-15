import os
import cv2

def map_data(aug_name):
    data = {"Rotate Upto 20deg" : "rotate",
            "Remove saturation" : "remove_saturation",
            "Boxes On Image":"box",
            "Vertical Flip" : "Vflip",
            "Horizontal Flip" : "Hflip",
            "Motion Blur" : "motionBlur",
            "Color_Temperature": "temperature",
            "Bright&Dark" : "bright&dark"
            }
    return data.get(aug_name)


def get_bbox_coor_by_image(image,detection):
    return list(detection.txt_converter(image))[0][1]


def image_format_change(folder):
    for im in os.listdir(folder):
        if im.endswith('.txt'):
            continue
        elif im.endswith('.jpg'):
            continue
        else:
                # im_name = im.split('.')[0]
            im_name = os.path.splitext(im)[0]
            ims = cv2.imread(f'{folder}/{im}')
            cv2.imwrite(f'{folder}/{im_name}.jpg',ims)
            os.remove(f'{folder}/{im}')