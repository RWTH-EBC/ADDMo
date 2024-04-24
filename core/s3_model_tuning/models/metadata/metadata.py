from pydantic import BaseModel, Field


class Metadata(BaseModel):
    addmo_class: str = Field(description='model method called for training')
    addmo_commit_id: str = Field ( description= 'commit id')
    library: str = Field(description='library used')
    library_model_type: str = Field(description='type of model within library')
    library_version: str = Field(description='library version used')
    target_name: str = Field("FreshAir Temperature", description="Name of the target variable")
    feature_order: list = Field(description='name and order of features')
    preprocessing: list = Field('StandardScaler for all features',
                                description="Preprocessing steps applied to the features.")
    instructions: str = Field('Pass a single or multiple observations with features in the order listed above',
                              description="Instructions for passing input data for making predictions.")

