from pydantic import BaseModel, Field


class ModelMetadata(BaseModel): #TODo: move this to abstractmodel.py and add docstring
    addmo_class: str = Field(description='ADDMo model class type, from which the regressor was saved.')
    addmo_commit_id: str = Field ( description= 'Current commit id when the model is saved.')
    library: str = Field(description='ML library origin of the regressor')
    library_model_type: str = Field(description='Type of regressor within library')
    library_version: str = Field(description='library version used')
    target_name: str = Field(description="Name of the target variable")
    features_ordered: list = Field(description='Name and order of features')
    preprocessing: list = Field('StandardScaler for all features',
                                description="Preprocessing steps applied to the features.")
    instructions: str = Field('Pass a single or multiple observations with features in the order listed above',
                              description="Instructions for passing input data for making predictions.")

