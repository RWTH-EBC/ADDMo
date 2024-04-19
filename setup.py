from setuptools import setup, find_packages
import os

'''''
installations done when setting up on VM python 3.11
pandas 2.2.1
scikit-learn 1.4.1
hyperopt 0.2.7
optuna 3.6.1
matplotlib 3.8.4
pyod 1.1.3
wandb 0.16.6
pydantic 2.6.4
onnx 1.16.0
skl2onnx 1.16.0
seaborn 0.13.2
plotly 5.20.0
'''''

if os.path.exists('requirements.txt'):

        with open('requirements.txt', 'r') as f:
            packages= f.read().splitlines()

        try:
            with open('README.md', 'r', encoding='utf-8') as f:
                long_description = f.read()
        except FileNotFoundError:
            long_description = "Automated data driven modeling using regression machine learning"


        setup(
            name='ADDMo',
            version='0.1',
            packages=find_packages(),
            url='https://git.rwth-aachen.de/EBC/Team_BA/Data_Driven_Modeling',
            license=' ',
            author='mre',
            author_email=' ',
            description='Automated data driven modeling using regression machine learning',
            long_description=long_description,
            install_requires=packages,
        )
else:
    raise ValueError('requirements.txt file not found')