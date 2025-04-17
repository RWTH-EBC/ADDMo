import os
import json
from addmo.s4_model_testing.model_testing import model_test
from addmo.util.load_save_utils import root_dir
from addmo.util.definitions import  return_results_dir_model_tuning

if __name__ == "__main__":

    # Define path of saved model directory, don't specify arguments in the function below if model is saved to default path
    dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning_fixed')
    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Define new input data path
    input_data_path = os.path.join(root_dir(),'addmo_examples','raw_input_data','InputData.xlsx')
    # Saving directory of new tuned data test
    input_data_exp_name = 'model_test'
    error, saving_dir = model_test(dir,model_config, input_data_path, input_data_exp_name, 'fixed' )
    print("error is: ", error)
    print("saving_dir is: ", saving_dir)
