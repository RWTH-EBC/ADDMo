import os
from setuptools import setup, find_packages

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        packages = f.read().splitlines()
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            long_description = f.read()
    except FileNotFoundError:
        long_description = "Automated system_data driven modeling using regression machine learning"

    setup(
        name='ADDMo',
        version='0.1',
        packages=find_packages(),
        url='https://git.rwth-aachen.de/EBC/Team_BA/Data_Driven_Modeling',
        license=' ',
        author='mre',
        author_email=' ',
        description='Automated system_data driven modeling using regression machine learning',
        long_description=long_description,
        install_requires=packages,
    )
else:
    raise ValueError('requirements.txt file not found')
