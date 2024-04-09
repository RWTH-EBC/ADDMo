from setuptools import setup

pkg_lite = """sklearn-pandas   ==  1.8.0,
hyperopt         ==  0.1.2,
scikit-learn     ==  0.20.0,
openpyxl         ==  2.5.4,
PyForms          ==  4.0.3,
remi             == 2019.4,
statsmodels      ==  0.11.0,
numpy             == 1.15.4,
xlrd            ==  1.2.0,
pillow          == 6.0.0,
matplotlib      == 2.2.2,
pandas        == 0.25.3, 
networkx        ==   1.11"""

'''
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


'''

pgk_list = pkg_lite.replace(" ", "").split(",\n")

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
