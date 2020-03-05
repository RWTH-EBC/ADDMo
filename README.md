# ADDMo functional overview

__ADDMo is an automated machine learning tool for regression tasks__

__ADDMo faces the following challenges:__

- Preprocessing of initial data (in some extent)
- Selection of proper training and test data periods 
- Selection and creation of optimal features 
- Selection of a model, the model configuration 
- Hyperparameter tuning 
- Overfitting & underfitting 

__The used methods for facing those challenges are:__

Preprocessing:
- Resampling* to the desired resolution.
- Initial custom feature selection* before tuning the data.
- Data cleaning: Replaces NaNs and infinite values.
- Scaling and normalizing: RobustScaler, StandardScaler &
no scaling.

Period selection:
- Time series plotting: Visualization of the signal’s time series
for detecting extraordinary patterns or mistakes in the data
via matplotlib.
- Custom period selection*

Feature creation:
- Cross-, cloud- and autocorrelation plotting: Visualization
of correlations for detecting influential lags and gathering
insight into the system’s dependencies via matplotlib.
- Creation of differences*: Creation of feature derivatives.
- Custom featurelag creation*
- Automatic featurelag creation*: Wrapper for automatic creation
of the best lag per feature within a custom lag range.
Each lag is only created if beneficial. The BBOM is based
on the assumption that only one lag per feature has real
informative value.
- Custom ownlag creation*
- Automated time series ownlag creation*: Wrapper for creating
the optimal number of time series ownlags. Ownlags
are added as long as they improve the score. The selection
is based on the assumption that the score is monotonically
increasing with the number of ownlags, till it reaches the
global optimum.

Feature selection:
- Low variance filter: Deletes features with low variance.
- Custom feature selection* of created features.
- Independent component analysis: Separating of superimposed
features.
- Univariate filter: Several search and rating strategies for
univariate filters.
- Embedded recursive feature selection: Embedded multivariate
feature selection, see scikit-learn.org for further information. Number
of features can be both found automatically or set manually.
- Embedded feature selection by threshold: Univariate feature
selection by a custom threshold of importance.
- Wrapper recursive feature selection*: Multivariate wrapper
which iteratively deletes the worst feature of the respective
feature subset as long as it improves the score.

Sample processing:
- Individual model “hourly”*: One model per hour of the day.
- Individual model “weekday & weekend”*: One model for
weekdays and one for weekends.
- Individual model “by feature”*: One model for all samples
with the feature’s values below and one for values above a
certain threshold. The respective feature is user-defined.
- Shuffle: Random shuffle of samples.

Model tuning:
- Model selection*: Exhaustive wrapper for selecting the best
out of all implemented models.
- Recursive prediction*: Enabling multistep ahead prediction
for models with time series and inertia ownlags. Iteratively
feeding the forecast back to the ownlag slots of the input
features.
- Hyperparameter tuning: via bayesian optimization or grid search
- Cross-validation: prevent overfitting

All those methods are applied sequentially while each method
is optional, thus ensuring that any combination can be selected.

The implemented models are 
- “multi layer perceptron” (ANN) 
- “epsilon support vector regression” (SVR)
- “random forest” (RF)
- “gradient tree boosting” (GB)
- “lasso”

The methodology of train & test set differentiation, hyperparameter
tuning and cross-validation is depicted in "ModelTuningFlowchart.vsdx". Moreover,
the figure illustrates, how “individual model” and “sample
shuffle” are implemented. The implementation includes a
comprehensive documentation of all settings and results via
tables and plots. Additionally, it enables insight to all changes
conducted, while tuning the data, by documenting the data set
after each method field.

The tool is mainly designed to perform modelling on time series data,
via regression and time series analysis.
Nevertheless it can also be used to handle data indexed by an id, 
simply converting the id into a timestamp (pandas.datetimeindex convention).

06.06.2018 what is the tool not able to do:
The tool is single output only (no MIMO).
It has no natively recurrent model, means it only uses ownlags as a regular input for regression analysis (A native recurrent model would be e.g. long short term memory neural networks)

# How to set it up - two options

1. using exclusively the GUI via docker container 
2. full spectrum via your python environment and editor (suggested) 

__1: Docker__

Install docker - make sure it works properly

Open CMD

`$cd <Path to ADDMo Repo>`\
`$docker image build -t addmo .`\
`$docker container run --publish 8081:8081 -it -v D:/Git_Repos/Data_Driven_Modeling:/ADDMo --name addmocontainer addmo`


Open your browser and enter the URL: [http://127.0.0.1:8081/]
1. The GUI should open up
2. Use the tool
3. Interact with ADDMo via the CMD
4. If a script runs into an error you have to restart the container 

To restart the container in cmd:\
"strg+c" for stopping the program

`$ docker container stop addmocontainer` \
`$ docker container rm addmocontainer` \
`$ docker container run --publish 8081:8081 -it -v D:/Git_Repos/Data_Driven_Modeling:/ADDMo --name addmocontainer addmo `



__2: Python + Editor__

Via setup.py:\
Install Anaconda (conda version: 4.8.0)

Open command line and create an python 3.6 environment via:
`$ conda create --name ADDMo python=3.6 `

Type y for accepting to install first packages
`$ y`

Activate environment
`$ conda activate ADDMo`

Install required packages via:
`$ pip install -e <Path to your local ADDMo repo>`

Set the conda environment "ADDMo" as interpreter for e.g. in PyCharm

Via pip by your own:\
The used Python version is 3.6.2

Except the regular packages you need to install:\
sklearn-pandas   ==  1.8.0,     (https://github.com/scikit-learn-contrib/sklearn-pandas) \
hyperopt         ==  0.1.2,     (http://hyperopt.github.io/hyperopt-sklearn/) \
scikit-learn     ==  0.20.0,    (https://scikit-learn.org/stable/install.html) \
openpyxl         ==  2.5.4,     (https://openpyxl.readthedocs.io/en/stable/) \
PyForms          ==  4.0.3,     (https://pyforms.readthedocs.io/en/v3.0/) \
remi             == 2019.4,     (https://github.com/dddomodossola/remi) \
statsmodels      ==  0.11.0, \
numpy             == 1.15.4, \
xlrd            ==  1.2.0, \
pillow          == 6.0.0, \
matplotlib      == 2.2.2, \
pandas        == 0.25.3,  \
networkx        ==   1.11       (https://networkx.github.io/) 



# How to use it- two options

1. via the GUI
2. accessing the python files and run them directly 

__Using the GUI:__

Run GUI_remi.py and see the information within the GUI

Select the respective "tool" via the tabs:

1.The automatic procedure (Auto final bayes: Importing data, tuning data, training the model while automatically\
selecting the best: "Model", "Individual Model", "Features" and "Hyperparameters of the model". Also evaluate the models via out-of-sample prediction), \
Necessary steps "Auto final bayes":\
	1.Upload input data\
	2.Define settings\
	3.Run

2.Data tuning: Importing input data, tuning data.\
Necessary steps "Data tuning":\
	1.Upload input data\
	2.Define settings\
	3.Run

3.Model tuning: Importing the previously tuned data, training the model with optimizing the hyperparameters and evaluate the model via out-of-sample prediction.\
Necessary steps "Model tuning":\
	1.Define the folder from which the tuned data shall be loaded\
	2.Define settings\
	3.Run

4.Only predict: Import previously trained models and their underlying tuned data, predict and evaluate the model with a more sophisticated evaluation method.\
Necessary steps "Predict only":\
	1.Define the folder from which the trained models (and their respective tuned data) shall be imported\
	2.Define settings\
	3.Run


__Running the scripts directly via the python console:__

*Executive scripts are:*\
- DataTuning.py for tuning the data (achieving the tuned data)
- ModelTuning.py for tuning the model (with the tuned data as input). 
In the final lines of ModelTuning.py one can define via commmenting and uncommenting, whether the automatic procedure (final bayes: training the model while automatically selecting the best: 
		"Model", "Individual Model", "Features" and "Hyperparameters of the model"), the regular procedure (optimizing the hyperparameter of the model), or the procedure for using previously trained models to only predict.


Set a name of the data and a name of the experiment in order to save your documentation and results (final input data) in a folder named as the data and a subfolder named as the name of experiment. This allows to go back to this final input data whenever you want. 

*Define all variables in SharedVariables.py:*\
Advises on how to understand the entry section in SharedVariables, per Method you´ll find:\
1st Line: A comment about what the method is or does\
2st Line: A variable that decides whether this method will be used or not. (possible entries are: True or False)\
Following lines: Only if additional attributes need to be set: The respective attributes, read the comments to understand which entries are valid. 
Empty lines separate the methods

Check for the order of how the methods are executed, as each method´s input is the output of the method conducted before.

-------------------------------------------

__Information about the required input shape:__
- Input ExcelFile has to be named: "InputData" and saved in the Folder "Data"
- Sheet to read in must be the first sheet, with time as first column and all signals and features thereafter (one per column)
- The time must be in the format of "pandas.datetimeindex"
- Columns must have different names
- Each columns has to have a unit, which should be written like: [kwh] if no unit is available write []
- The index should be continuously counting(no missing steps)

__Understanding the handling of saving the results:__\
A folder called results is created within the directory of the python files. Within that folder a four layered folder system is used, the next layer is a subfolder of the respective previous layer. The folder are created by the program, only their names must be defined:
- Layer0: "Results", general folder for all results)
- Layer1: "NameOfData", name of the folder used to declare which input data is used for the results within.
- Layer2: "NameOfExperiment", name of the folder in which the results of "DataTuning" are saved, including the "tuned data" which will be the input for model tuning.
- Layer3: "NameOfSubTest", name of the folder in which the results of "ModelTuning" are saved, including the trained models which will be the input for only predicting.
- Layer4: "NameOfOnlyPredict", name of the folder in which the results of "OnlyPredict" are saved.



# Scheme of ADDMo:

The program is built like the mainconcept (file in the readme folder), take it as guideline. Read the comments in the code or the GUI to get more information.

After reading the above instructions, check all documents in the readme-folder as supplemental documents.
- MainConcept - Verknüpfung : Here the theoretical concept of the program is depicted.
- ProgramFlowchart.vsdx : Here you can see which methods are available in the program and in which section they are executed.
- MethodDescription.doc : This is a list of all methods plus their Input/Output, the theoretical function and their practical function.
- DetailedMethodsDescription_CodingPointOfView.xlsx : This is a list of all methods with their attributes and their meaning, and a more detailed description of each method

# Cite ADDMo:

If you use ADDMo in scientific publication, we would appreciate citations to the following paper:

Automated data-driven modeling of building energy systems via machine learning algorithms, Rätz et al., Energy and Buildings, Volume 202, 2019.

Published at Energy&Buildings Journal:
[Link to article](https://doi.org/10.1016/j.enbuild.2019.109384)

If you are not granted access to the paper, you may find the pre-print at:
[Link to pre-print article at researchgate](https://www.researchgate.net/publication/335424562_Automated_Data-driven_Modeling_of_Building_Energy_Systems_via_Machine_Learning_Algorithms)
