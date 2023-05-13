import uuid
import cv2
import os
import labelme
from numpyencoder import NumpyEncoder

import base64
import json

import instance_seg.logging_util as logging_util

logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])

def create_new_txt(image , label ,  points,train_path_images , train_path_labels,dict,yolo=True):

    if yolo:

        try:
            save_name = uuid.uuid4()
            
            # if type(image) == str:
            #     image = cv2.imread(image)

            # cv2.imwrite(f'{train_path_images}/{save_name}.jpg',image)
            txt_path=f'{train_path_labels}/{save_name}.txt'
            
            new_height , new_width , c = image.shape
        
            for p in points: 
                labels = p.label 
                label = dict[labels]
                new_name = str(label) + ' '
                points = []

                for ps in p: 

                    points.append(list(ps))
                    x , y = list(ps)
                    normalize_X = x/new_width
                    normalize_Y = y/new_height

                    
                    new_name += f'{normalize_X} {normalize_Y}'
                    new_name += ' '
                
                if new_name != '':
                    if type(image) == str:
                        image = cv2.imread(image)

                    cv2.imwrite(f'{train_path_images}/{save_name}.jpg',image)
                 
               
                    with open(txt_path,'a+') as f:
                    
                        f.write(new_name)
                        f.write('\n')
                    
                        
                del new_name
            
        except Exception as e:
            logger.warning(f'problem : create new json  desc : {e}')

    else:
        try:
            save_name = uuid.uuid4()
            annot_dict={"version": "4.6.0","flags": {},"shapes":[]}
            if type(image) == str:
                image = cv2.imread(image)

            cv2.imwrite(f'{train_path_images}/{save_name}.jpg',image)
            for p in points: 
                labels = p.label 
                label = dict[labels]
                new_name = str(label) + ' '
                points = []

                for ps in p: 

                    points.append(list(ps))
                
                if points != []:

                    sh_dict={"label": labels,"points":points , "group_id": None, "shape_type": "polygon", "flags": {}}
                    annot_dict["shapes"].append(sh_dict)
                    annot_dict["imagePath"] = f'{save_name}.jpg'
                    img_data = labelme.LabelFile.load_image_file(f'{train_path_images}/{save_name}.jpg')
                    img_data = base64.b64encode(img_data).decode('utf-8')
                    height , width , _ = image.shape
                    annot_dict["imageData"]=img_data
                    annot_dict["imageHeight"]=height
                    annot_dict["imageWidth"]=width
                    json_file=f'{train_path_labels}/{save_name}.json'
                                
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(annot_dict, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)
                
            
        except Exception as e:
            logger.warning(f'problem : create new text  desc : {e}')


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    #         # print(x/new_width,y/new_height)
    #         # a = (x/new_width).ravel()
            
         
        
    #     print('***') 
        # print(points)
    
        # sh_dict={"label": label,"points":points , "group_id": None, "shape_type": "polygon", "flags": {}} 
        # annot_dict["shapes"].append(sh_dict)
        # annot_dict["imagePath"] = f'{save_name}.jpg'
        
        # img_data = labelme.LabelFile.load_image_file(f'{train_path_images}/{save_name}.jpg')
        # img_data = base64.b64encode(img_data).decode('utf-8')
        # height , width , _ = image.shape
        # annot_dict["imageData"]=img_data
        # annot_dict["imageHeight"]=height
        # annot_dict["imageWidth"]=width
        # # print(train_path_labels)
        # json_file=f'{train_path_labels}/{save_name}.json'
                        
        # with open(json_file, 'w', encoding='utf-8') as f:
        #         json.dump(annot_dict, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)