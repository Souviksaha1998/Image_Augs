import yaml

def load_yaml(yaml_filename):
    with open(yaml_filename, 'r') as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
    return loaded_data