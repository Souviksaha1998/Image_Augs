import uuid
import cv2
import os



import instance_seg.logging_util as logging_util

logger = logging_util.get_logger(os.path.basename(__file__).split('.')[0])

def create_new_txt(image ,points,train_path_images , train_path_labels):

   

    try:
            save_name = uuid.uuid4()
            
            
            txt_path=f'{train_path_labels}/{save_name}.txt'
            
            new_height , new_width , c = image.shape
        
            for p in points: 
                
                labels = p.label 
                new_x1 , new_y1 , new_x2, new_y2 = p.x1 , p.y1 , p.x2 , p.y2
                
                new_name = str(labels) + ' '

                x1norm , y1norm , x2norm , y2norm = convert_to_yolo_bbox([new_x1,new_y1,new_x2,new_y2],new_width,new_height)
                new_name += f'{x1norm} {y1norm} {x2norm} {y2norm}'
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
            logger.warning(f'problem : create new txt  desc : {e}')


def convert_to_normal_bbox(yolo_bbox, image_width, image_height):
    """
    Convert YOLO's normalized bounding box coordinates to normal bounding box format.
    yolo_bbox: Tuple of (x, y, w, h) normalized bounding box coordinates from YOLO's output.
    image_width: Width of the input image.
    image_height: Height of the input image.
    Returns: Tuple of (x_min, y_min, x_max, y_max) normal bounding box coordinates.
    """
    # Get the normalized coordinates
    x_norm, y_norm, w_norm, h_norm = yolo_bbox
    
    # Convert the normalized coordinates to pixel coordinates
    x_pixel = int(x_norm * image_width)
    y_pixel = int(y_norm * image_height)
    w_pixel = int(w_norm * image_width)
    h_pixel = int(h_norm * image_height)
    
    # Calculate the normal bounding box coordinates
    x_min = x_pixel - (w_pixel // 2)
    y_min = y_pixel - (h_pixel // 2)
    x_max = x_pixel + (w_pixel // 2)
    y_max = y_pixel + (h_pixel // 2)
    
    # Return the normal bounding box coordinates
    return (x_min, y_min, x_max, y_max)
        
        
def convert_to_yolo_bbox(normal_bbox, image_width, image_height):
    """
    Convert normal bounding box coordinates to YOLO's normalized bounding box format.
    normal_bbox: Tuple of (x_min, y_min, x_max, y_max) normal bounding box coordinates.
    image_width: Width of the input image.
    image_height: Height of the input image.
    Returns: Tuple of (x, y, w, h) normalized bounding box coordinates.
    """
    # Get the normal bounding box coordinates
    x_min, y_min, x_max, y_max = normal_bbox
    
    # Calculate the pixel coordinates
    x_pixel = (x_min + x_max) // 2
    y_pixel = (y_min + y_max) // 2
    w_pixel = x_max - x_min
    h_pixel = y_max - y_min
    
    # Convert the pixel coordinates to normalized coordinates
    x_norm = x_pixel / image_width
    y_norm = y_pixel / image_height
    w_norm = w_pixel / image_width
    h_norm = h_pixel / image_height
    
    # Return the normalized bounding box coordinates
    return (x_norm, y_norm, w_norm, h_norm)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
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