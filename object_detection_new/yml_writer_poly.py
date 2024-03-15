import yaml

def yaml_writer(total_no_of_detection , labels_list,augmentation_folder_name):


    template = '''
# ImageAugs Support augmentation for Object detection , Instance segmentation and Classification.
 
train: ../train/images
test: ../test/images
val: ../test/images

nc: {}
names: {}


ImageAugs:
    type: Detection
    created by: Souvik Saha
    year: 2030-24
    version: 2.4.20
    
        
    '''

    template_str = template.format(total_no_of_detection,labels_list)
    with open(f"{augmentation_folder_name}/data.yaml", "w") as f:
        f.write(template_str)


def load_yaml(yaml_filename):
    with open(yaml_filename, 'r') as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
    return loaded_data



    