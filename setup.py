from setuptools import setup, find_packages

setup(
    name='manolo',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'torch', 'torchvision', 'numpy', 'tqdm', 'wandb', 'IPython'
    ],
    description='MANOLO database. v0.0: Demo version for library usage',
    
)
