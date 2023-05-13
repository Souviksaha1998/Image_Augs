

def yaml_writer(total_no_of_detection , labels_list,augmentation_folder_name):


    template = '''
# ImageAugs Support augmentation for Object detection , Instance segmentation and Classification.
 
train: ../train/images
test: ../test/images
val: ../test/images

nc: {}
names: {}


ImageAugs:
    type: segmentation
    
        
    '''

    template_str = template.format(total_no_of_detection,labels_list)
    with open(f"{augmentation_folder_name}/data.yaml", "w") as f:
        f.write(template_str)
    