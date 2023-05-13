import setuptools


with open('requirements.txt','r') as f:
    requirements = f.read().splitlines()

with open('README.md','r') as li:
    readme = li.read()
    
with open('VERSION','r') as li:
    version = li.read()

classifiers = [
      'Intended Audience :: Developers'
    , 'Operating System :: OS Independent'
    , 'Programming Language :: Python'
    , 'Programming Language :: Python :: 3.8'
    , 'License :: OSI Approved :: MIT License'
]

setuptools.setup(
include_package_data=True,
name='image_augs',
version=version,
description='Image Augs supports Augmentation for Object Detection , Instance Segmentation and classification tasks.',
long_description =readme,
long_description_content_type='text/markdown',
author='Souvik Saha',
packages=setuptools.find_packages(),
install_requires=requirements,
keywords="machine_learning,development,data_augmentations",
python_requires=">=3.7",
classifiers= classifiers,
author_email='ssouvik.191@gmail.com',
license_files = 'LICENSE'

)


