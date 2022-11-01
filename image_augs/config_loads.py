from configparser import ConfigParser

import os

import image_augs.logging_util as logging_util


#logging
logger = logging_util.get_logger(os.path.basename(__file__))

def read_config(file_name:os.path) -> object: 

    '''
    This function will read config file
    
    : param read_config : it accepts config file full path
    : return            : it returns read config file
    
    '''

    
    config = ConfigParser()

    try:

        with open(file_name) as fh:

            config.read_file(fh)
            
        return config

    except Exception as e:

        logger.error(f'ConfigFile not found!,{e}')
        raise FileNotFoundError(f'Config file not found, Please provide full path of config file,{e}')
        




