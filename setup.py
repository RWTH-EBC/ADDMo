from setuptools import setup

pkg_list = [
"sklearn-pandas",
"hyperopt",
"scikit-learn",
"openpyxl",
"PyForms",
"remi",
"statsmodels",
"numpy",
"xlrd",
"pillow",
"matplotlib",
"pandas",
"networkx",
]


setup(
    name='ADDMo',
    version='0.1',
    packages=[''],
    url='https://git.rwth-aachen.de/EBC/Team_BA/Data_Driven_Modeling',
    license='',
    author='mre',
    author_email='',
    description='Automated data driven modeling using regression machine learning',

    install_requires=pgk_list,
)
