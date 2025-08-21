import os
import json
from addmo.s4_model_testing.model_testing import model_test
from addmo.util.load_save_utils import root_dir
from addmo.util.definitions import  return_results_dir_model_tuning

def exe_model_test(dir, input_data_path, data_tuning_type, input_tuned_data_path):

    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Saving directory of new tuned data test
    input_data_exp_name = 'model_test'
    # Specify the type of tuning used for training the model in model_tuning kwarg below.
    # This ensures that the input data structure is consistent throughout the training and prediction.
    model_config['abs_path_to_data']=input_tuned_data_path
    error, saving_dir = model_test(dir, model_config, input_data_path, input_data_exp_name, data_tuning_type)
    print("error is: ", error)
    print("saving_dir is: ", saving_dir)


if __name__ == "__main__":

    # By default, the model after exe_model_tuning is saved at 'addmo_examples/results/test_raw_data/test_data_tuning/test_model_tuning'. Simply pass the function
    # return_results_dir_model_tuning() to set this directory. Change the arguments only when the default name_of_raw_data, name_of_data_tuning_experiment or
    # name_of_model_tuning_experiment is changed during model_tuning, since the saving directory of model depends on it.
    dir =  r'C:\Users\mre-rpa\PycharmProjects\ADDMo\addmo_examples\results\test_raw_data_auto\test_data_tuning\test_model_tuning'
    # dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning_fixed')
    # Define new input data path
    input_data_path = os.path.join(root_dir(), 'addmo_examples', 'raw_input_data', 'InputData.xlsx')
    data_tuning_type = 'auto'
    input_tuned_data_path= r'C:\Users\mre-rpa\PycharmProjects\ADDMo\addmo_examples\results\test_raw_data\data_tuning_experiment_auto\tuned_xy_auto.csv'
    exe_model_test(dir, input_data_path, data_tuning_type, input_tuned_data_path)
