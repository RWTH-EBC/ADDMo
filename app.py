import streamlit as st
import json
import os
from streamlit_pydantic import pydantic_input, pydantic_output, pydantic_form
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo_examples.executables.exe_data_tuning_auto import exe_data_tuning_auto
from addmo_examples.executables.exe_data_tuning_fixed import exe_data_tuning_fixed
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo_examples.executables.exe_model_tuning import exe_model_tuning
from addmo.util.load_save_utils import root_dir
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from streamlit_pdf_viewer import pdf_viewer
from addmo.util.definitions import results_dir_data_tuning_auto,results_dir_data_tuning_fixed,return_results_dir_model_tuning, results_model_streamlit_testing
from addmo_examples.executables.exe_data_insights import exe_time_series_plot,exe_parallel_plot,exe_carpet_plots
from addmo.s4_model_testing.model_testing import model_test, data_tuning_recreate_fixed, data_tuning_recreate_auto




def exe_streamlit_data_tuning_auto():
    st.header("Auto Data Tuning Configuration")
    st.markdown("""
       
        This setup allows you configure and launch a fully-automated data tuning pipeline.
        Use the form below to customize:
        - Preprocessing & feature engineering (e.g., lag generation, variance filtering)
        - Feature selection strategies
        - Model tuning setup (required for automatically generating feature lags based on model performance improvement)

        """)

    st.subheader("Data Tuning Configuration")
    st.markdown("""

    ####  General Setup
    - **`name_of_raw_data`**: Name of results data folder (used in output paths).
    - **`name_of_tuning`**: Name of the tuning experiment.
    - **`abs_path_to_data`**: Full path to your input Excel file.
    - **`name_of_target`**: The name of your target column to be predicted.

    - The default results data folder is: `addmo-automated-ml-regression\\addmo_examples\\results\\test_raw_data`
    - The default name of tuning experiment is: `data_tuning_experiment_auto`
    
    - The information about every feature is provided in the help tab of each feature along with default values

    #### Feature Construction
    Control how new features (like lags or derivatives) are created:
    - **`create_differences`**: Create first-order differences (like feature derivatives).
    - **`create_manual_target_lag`**: Add lags for the target variable manually.
    - **`target_lag`**: Define which lags to add for the target (e.g., [1, 2] = t-1, t-2).
    - **`create_automatic_timeseries_target_lag`**: Enable automatic lag generation.
    - **`create_manual_feature_lags`**: Add lags for specific input variables.
    - **`feature_lags`**: Define feature-specific lags: e.g., `{feature: [1, 2]}`. If unsure, set default to: FreshAir Temperature: [1, 2], Total active power: [1, 2]
    - **`create_automatic_feature_lags`**: Let a model choose optimal lags.
    - **`minimum_feature_lag` / `maximum_feature_lag`**: Range of lags to consider between 1 and 20.

    """)

    st.header("Guide for Data Tuning")
    st.markdown("""
    ### Manual Configuration

    Use these options if you want to explicitly define features or lags.
    Note: No need to specify model config if these all features are constructed manually using the fields below
    
    ##### Manually Select Features
    - ✅ Enable `manual_feature_selection` to select the features which will be included in tuned data
    -  Provide the feature names in `selected_features`, e.g.: ["FreshAir Temperature", "Total active power"]
    ##### Manually Create Feature Lags
    - ✅ Enable `create_manual_feature_lags` to create feature lags 
    - Define lag values for each feature in `feature_lags`, e.g.: 
    {
        "FreshAir Temperature": [1, 2],
        "Total active power": [1, 2, 3]
    }
    ##### Create Manual Target Lags
    - ✅ Enable `create_manual_target_lag` for manual construction of target lags.
    - Define lag values for target feature here in `target_lag`, e.g.: [1,2]
    
    ### Automated Configuration
    
    Skip filling the field values for the above mentioned attributes if you would like to proceed with automated configuration.
    Note: Filling the Model config below is required in case these options are selected.
    #### Automatically Create Feature Lags
    - ✅ Enable create_automatic_feature_lags
    - Configure: `minimum_feature_lag` , `maximum_feature_lag`, `config_model_tuning` (defines the model used to evaluate lags) 
    #### Automatically Create Target Lags
    - ✅ Enable create_automatic_timeseries_target_lag
    - Set minimum_target_lag only
    
    #### Further features: 
    These options are **used only in automated tuning workflows** where model-based evaluation is performed (e.g., during automatic feature lag or target lag construction).  
    They are **not used** when you're manually selecting features or creating lags.
    - **`filter_low_variance`**: Pre-Filter removing features with low variance.
    - **`low_variance_threshold`**: Minimum variance required.
    - **`filter_ICA`**: Apply Independent Component Analysis.
    - **`filter_univariate`**: Filter univariate with scoring function `f-test` or `mutual information` and search mode : `‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’`.
    - **`univariate_score_function`**: `'mutual_info_regression'` or `'f_regression'`.
    - **`univariate_search_mode`**: Choose top features via `‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’`.
    - **`univariate_filter_params`**: Percentage or count of features to keep.
        
    #### Embedded / Wrapper Selection
    Learns which features matter while the model is training:
    - **`embedded_model`**: Estimator used for model-based selection (e.g., RF). It scores the importance of each feature based on how useful it is for making predictions.
    - **`filter_recursive_embedded`**: Recursive feature elimination.
    - **`recursive_embedded_number_features_to_select`**: Number of features to select in recursive feature elimination.
    - **`wrapper_sequential_feature_selection`**: Forward/backward wrapper selection.
    - **`sequential_direction`**: `'forward'` or `'backward'`.
    - **`min_increase_4_wrapper`**: Minimum performance gain needed to accept a feature.
    
    """)
    auto_tuning_config = pydantic_input("Auto", DataTuningAutoSetup)
    st.header("Model Tuner Configuration")

    st.markdown("""
     **Model Tuner** automatically generates feature lags and evaluates features during embedded/wrapper selection.

    This helps the pipeline understand:
    - Which feature lags improve model performance the most
    - Which features to keep based on their contribution to predictions
    ---
    Recommendations about some hyperparameters:
    - Use **`ScikitMLP_TargetTransformed`** for non-linear patterns in time series.
    - **`trainings_per_model`**: `3–5` for deep models (e.g. MLP), `1–2` for simple models (e.g. linear)
    - **`hyperparameter_tuning_kwargs`**: `n_trials = 2-5`. Do not leave this field empty.
    - **`validation_score_mechanism`**: **`cv`** for consistent and robust evaluation.
    - **`validation_score_mechanism_kwargs`**: {"test_size": 0.2}
    - **`validation_score_splitting`**: `KFold` *(default, recommended)*, `PredefinedSplit`
    - **`validation_score_splitting_kwargs`**: For KFold, example:{"n_splits": 5, "shuffle": True}
    - **`validation_score_metric`**: Scoring function to decide which lags or features are valuable.
    
        - `r2`: Score from 0–1 *(higher is better)*
        - `neg_root_mean_squared_error` *(default)*
        - `neg_mean_absolute_error`

        *Use `neg_root_mean_squared_error` to prioritize precision in regression.*

    - **`validation_score_metric_kwargs`**: Advanced tweaks for the metric (rarely needed).

    ---

    """)
    model_tuner_config = pydantic_input("ModelTuner", ModelTunerConfig)
    if not model_tuner_config['hyperparameter_tuning_kwargs']:
        model_tuner_config['hyperparameter_tuning_kwargs'] = {"n_trials": 2}
    # auto_tuning_config._config_model_tuning = model_tuner_config

    auto_tuning_config["_config_model_tuning"] = model_tuner_config
    # Output strategy
    st.subheader("Output Directory Strategy")
    st.markdown("""
        Define how to handle existing results:
        - **y**: Overwrite the contents  
        - **d**: Delete and recreate the directory
    """)
    overwrite_strategy = st.radio(
        "Choose strategy for existing results directory:",
        ["y", "d"],
        index=0,
        help="Select how to handle existing results directory at: `addmo_examples/results/test_raw_data/data_tuning_experiment_auto`",
    )


    # Submit button
    if st.button("Run Auto Data Tuning"):
    # Run when user submits the config
        missing_fields = []
        for field_name, field_value in auto_tuning_config.items():
            # Check if field is missing or empty
            if isinstance(field_value, (list)) and not field_value:
                missing_fields.append(field_name)
            elif field_value in [None, ""]:
                missing_fields.append(field_name)

        # If any fields are missing, show an error
        if missing_fields:
            st.error(f"❌ The following required fields are missing or empty: {', '.join(missing_fields)}")
            return

        auto_tuning_config = DataTuningAutoSetup(**auto_tuning_config,)
        # Save config to the expected JSON location
        config_path = os.path.join(
            root_dir(), 'addmo', 's1_data_tuning_auto', 'config', 'data_tuning_auto_config.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            f.write(auto_tuning_config.model_dump_json(indent=4))

        st.success("✅ Configuration saved!")


        # Run the tuning process
        with st.spinner("Running data tuning..."):
            exe_data_tuning_auto(overwrite_strategy)
            # Load default saving path for plot
            plot_image_path = os.path.join(results_dir_data_tuning_auto("test_raw_data"), "tuned_xy_auto.pdf")
            st.markdown("### Auto-Tuned Data Plot")
            # Display the saved plot PDF
            pdf_viewer(plot_image_path, width= "80%", height= 855)
            st.success("✅ Data tuning completed!")

def exe_streamlit_data_tuning_fixed():
    st.header("Data Tuning Fixed Configuration")
    st.markdown("""
            This setup allows you tune the system data in a fixed manner without randomness
            """)

    st.subheader("Data Tuning Configuration")
    st.markdown("""

        ####  General Setup
        - **`name_of_raw_data`**: Name of results data folder (used in output paths).
        - **`name_of_tuning`**: Name of the tuning experiment.
        - **`abs_path_to_data`**: Full path to your input Excel file.
        - **`name_of_target`**: The name of your target column to be predicted.

        - The default results data folder is: `addmo-automated-ml-regression\\addmo_examples\\results\\test_raw_data`
        - The default tuning experiment is: `data_tuning_experiment_fixed`

        - The information about every feature is provided in the help tab of each feature along with default values
        
        ### Feature Construction
        - **`create_lag`**: Creates a lagged version of the input series.
        - **`create_diff`**: Creates a differenced version of the input series.
        - **`create_sqaured`**: Creates a squared version of the input series.
        
        ### How to define custom feature names:
        To define a new feature, use the format: `column name__feature name`
        - `FreshAir Temperature__lag3`: 3-step lag of the "FreshAir Temperature"
        - `Total active power__diff`: First difference of "Total active power"
        ###  Lag Naming Convention
        To create lagged features:
        - Use the format **`lagX`** where **X** is the number of time steps to shift.
        - Example: `Temperature__lag1` = Temperature at previous timestep          """)
    fixed_config_data = pydantic_input(key="Config Setup", model=DataTuningFixedConfig)

    st.subheader("Output Directory Strategy")
    st.markdown("""
       Define how to handle existing results:
       - **y**: Overwrite the contents
       - **d**: Delete and recreate the directory
    """)
    overwrite_strategy = st.radio(
        "Choose strategy for existing results directory:",
        ["y", "d"],
        index=0,
        help="Select how to handle existing results directory at: addmo_examples/results/test_raw_data/data_tuning_experiment_fixed",
    )

    if st.button("Run Fixed Data Tuning"):
        missing_fields = []
        for field_name, field_value in fixed_config_data.items():
            # Check if field is missing or empty
            if isinstance(field_value, (list)) and not field_value:
                missing_fields.append(field_name)
            elif field_value in [None, ""]:
                missing_fields.append(field_name)
        # If any fields are missing, show an error
        if missing_fields:
            st.error(f"❌ The following required fields are missing or empty: {', '.join(missing_fields)}")
            return

        fixed_config_data  = DataTuningFixedConfig(**fixed_config_data)
        # Save config to the expected JSON location
        config_path = os.path.join(
            root_dir(), 'addmo', 's2_data_tuning', 'config', 'data_tuning_config.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            f.write(fixed_config_data.model_dump_json(indent=4))

        st.success("✅ Configuration saved!")

        # Run the tuning process
        with st.spinner("Running data tuning.."):
            exe_data_tuning_fixed(overwrite_strategy)
            # Load default saving path for plot
            plot_image_path = os.path.join(results_dir_data_tuning_fixed("test_raw_data"), "tuned_xy_fixed.pdf")
            st.markdown("### Tuned Fixed Data Plot")
            # Display the saved plot PDF
            pdf_viewer(plot_image_path, width="80%", height=855)
            st.success("✅ Data tuning completed!")

def exe_streamlit_model_tuning():
    st.header("Model Tuning")
    st.markdown("""
    
     This setup allows you to:
    - Select and train one or multiple machine learning models.
    - Tune their hyperparameters using techniques like **Optuna** or **Grid Search**.
    - Use **cross-validation** or **holdout splits** to validate performance.
    - Choose the best-performing model from multiple runs to avoid local minima
    
    
    ####  General Setup
    - **`name_of_raw_data`**: Name of results data folder (used in output paths).
    - **`name_of_tuning`**: Name of the tuning experiment.
    - **`abs_path_to_data`**: Full path to your input Excel file.
    - **`name_of_target`**: The name of your target column to be predicted.

    - The default results data folder is: `addmo-automated-ml-regression\\addmo_examples\\results\\test_raw_data`
    - The default tuning experiment is: `test_data_tuning`
    - The default model tuning experiment is: `test_model_tuning`
    - Overall, the default results directory is: `addmo-automated-ml-regression\\addmo_examples\\results\\test_raw_data\\test_data_tuning\\test_model_tuning`

    
   
    
    Recommendations about some hyperparameters:
    - Use **`ScikitMLP_TargetTransformed`** for non-linear patterns in time series.
    - **`trainings_per_model`**: `3–5` for deep models (e.g. MLP), `1–2` for simple models (e.g. linear)
    - **`hyperparameter_tuning_kwargs`**: `n_trials = 2-5`. Do not leave this field empty.
    - **`validation_score_mechanism`**: **`cv`** for consistent and robust evaluation.
    - **`validation_score_mechanism_kwargs`**: {"test_size": 0.2}
    - **`validation_score_splitting`**: `KFold` *(default, recommended)*, `PredefinedSplit`
    - **`validation_score_splitting_kwargs`**: For KFold, example:{"n_splits": 5, "shuffle": True}
    - **`validation_score_metric`**: Scoring function to decide which lags or features are valuable.
    
        - `r2`: Score from 0–1 *(higher is better)*
        - `neg_root_mean_squared_error` *(default)*
        - `neg_mean_absolute_error`

        *Use `neg_root_mean_squared_error` to prioritize precision in regression.*

    - **`validation_score_metric_kwargs`**: Advanced tweaks for the metric (rarely needed).

    ---
    
    """)

    st.subheader("Model Configuration")
    model_config_data = pydantic_input("ModelConfig", ModelTuningExperimentConfig)


    st.subheader("Model Tuning Configuration")
    model_tuner = pydantic_input("ModelTunerConfig", ModelTunerConfig)
    if not model_tuner["hyperparameter_tuning_kwargs"]:
        model_tuner["hyperparameter_tuning_kwargs"] = {"n_trials": 2}
    model_config_data["config_model_tuner"] = model_tuner

    st.subheader("Input data tuning type")
    st.markdown("Please ensure that the data tuning process is completed and the files are saved at correct paths. "
                "By default, the experiment names and results directory are synced for data tuning and model tuning. "
                "Choose the default paths if the experiment names are not changed throughout the process.")
    type_of_data = st.selectbox("Would you like to use tuned data for model tuning?", ["choose", "Yes", "No"])
    st.session_state.use_tuned_data = type_of_data
    if type_of_data == "Yes":
            type_of_tuning = st.selectbox("Which tuning would you like to use?", ["choose","Auto", "Fixed"])

            if type_of_tuning == "Auto":
                st.text("The default directory used for loading the data is: addmo_examples/results/test_raw_data/data_tuning_experiment_auto")
                path_type = st.selectbox(
                    "Would you like to use the default saving path for loading the input data?",
                    ["Select an option", "Yes", "No"],
                    help= "The path for loading input data depends on the experiment names defined above. "
                           "If the default saving paths and experiment folder names are not changed during data tuning and model config, choose Default")
                if path_type== "Yes":
                    model_config_data["abs_path_to_data"] = os.path.join(
                        results_dir_data_tuning_auto(model_config_data['name_of_raw_data']),
                        "tuned_xy_auto.csv")
                    st.success("✅ Tuned data path set in config.")
                elif path_type == "No":
                    model_config_data["abs_path_to_data"] = st.text_input('path')
                    st.success("✅ Tuned data path set in config.")

            elif type_of_tuning == "Fixed":
                st.text("The default directory used for loading the data is: addmo_examples/results/test_raw_data/data_tuning_experiment_fixed")

                path_type = st.selectbox(
                    "Would you like to use the default saving path for loading the input data?",
                    ["Select an option", "Yes", "No"]
                )
                if path_type == "Yes":
                    model_config_data["abs_path_to_data"] = os.path.join(
                        results_dir_data_tuning_fixed(model_config_data['name_of_raw_data']),
                        "tuned_xy_fixed.csv")
                    st.success("✅ Tuned data path set in config.")

                elif path_type == "No":
                    model_config_data["abs_path_to_data"] = st.text_input('path')
                    st.success("✅ Tuned data path set in config.")
    if st.button("Save Model Config"):
        st.session_state.model_config_data = model_config_data
        st.success("Model configuration saved!")

    st.subheader("Output Directory Strategy")
    st.markdown("""
           Define how to handle existing results:
           - **y**: Overwrite the contents
           - **d**: Delete and recreate the directory
        """)
    overwrite_strategy = st.radio(
        "Choose strategy for existing results directory:",
        ["y", "d"],
        index=0,
        help="Select how to handle existing results directory at: addmo_examples/results/test_raw_data/data_tuning_experiment_fixed",
    )

    if st.button("Run Model Tuning"):
            model_config_data = st.session_state.model_config_data

            if "model_config_data" not in st.session_state:
                st.error("🚫 Model configuration is missing. Please fill out the model config first.")
            else:
            # Save and execute
                config_path = os.path.join(
                    root_dir(), 'addmo', 's3_model_tuning', 'config', 'model_tuning_config.json'
                )
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                model_config_data_obj = ModelTuningExperimentConfig(**model_config_data)
                with open(config_path, 'w') as f:
                    f.write(model_config_data_obj.model_dump_json(indent=4))

                st.success("✅ Configuration saved!")

                with st.spinner("Running model tuning..."):
                    exe_model_tuning(overwrite_strategy, model_config_data_obj)
                    plot_image_path = os.path.join(return_results_dir_model_tuning(model_config_data_obj.name_of_raw_data,model_config_data_obj.name_of_data_tuning_experiment, model_config_data_obj.name_of_model_tuning_experiment), "model_fit_scatter.pdf")

                    st.markdown("### Model fit Plot")

                    # Display the saved plot PDF
                    pdf_viewer(plot_image_path, width="80%", height=855)
                    st.success("✅ Model tuning completed!")
    st.markdown("---")
    if st.button("Run another model tuning"):
        st.rerun()

def exe_streamlit_data_insights():

    st.header('Generate Insights for the tuned saved models')
    st.markdown("""
    This setup allows you to **generate insightful visualizations** based on the results of previously trained and saved models. It's especially useful for **analyzing model performance** and **understanding the effect of features** through various plots.
    Once you load a previously saved model's config file and result directory, you can:

    - **Visualize Time Series Performance**  
      Plot predictions vs actual target values across time for the training data.
    
    - **Create Carpet Plots**  
      These 2D plots help visualize the interaction between multiple features and the predicted values.
    
    - **Draw Parallel Plots**  
      Visualize the relationships between feature combinations and their contribution to the prediction.
    
    **Note**:
    - Ensure that your selected directory contains a valid trained model config and model results.
    - The plots depend on a successful model tuning run — you must have completed that step first.
     """)
    if "dir_submitted" not in st.session_state:
        st.session_state.dir_submitted = False
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = ""

    if st.session_state.dir_submitted is False:
        st.subheader("Load previously saved Model for generating insights:")
        with st.form("Model Directory"):
            option = st.radio(
                "Select results directory option for loading previously saved model:",
                ("Default", "Custom")
            )

            if option == "Custom":
                directory = st.text_input("Enter the custom results directory path")
            else:
                directory = return_results_dir_model_tuning()

            submitted = st.form_submit_button("Submit")

            if submitted:
                st.session_state.model_dir = directory
                st.session_state.dir_submitted = True
                st.rerun()

    else:
        directory = st.session_state.model_dir
        st.write(f"Using directory: {directory}")

        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        plot_dir = os.path.join(directory, 'plots')

        with st.form("Choose plots for generating insights"):
            plots_selections = st.multiselect("Select the plots which you'd like to see",
                                              options=['Time Series plot for training data',
                                                       'Predictions carpet plot',
                                                       'Predictions parallel plot'])
            submitted = st.form_submit_button("Confirm plots")

            if submitted:
                st.session_state.plots_selections = plots_selections

                st.write(f"Plots will be saved in {plot_dir}")

                if 'Time Series plot for training data' in plots_selections:
                    exe_time_series_plot(model_config, "training_data_time_series", plot_dir, save=True)
                    st.markdown("### Time Series Data Plot")

                    # Display the saved plot PDF
                    path = os.path.join(plot_dir, "training_data_time_series.pdf")
                    pdf_viewer(path, width="80%", height=855)

                if 'Predictions carpet plot' in plots_selections:
                    exe_carpet_plots(model_config, "predictions_carpet_new", plot_dir, save=True)
                    st.markdown("### Carpet Plot")

                    # Display the saved plot PDF
                    path = os.path.join(plot_dir, "predictions_carpet_new.pdf")
                    pdf_viewer(path, width="80%")

                if 'Predictions parallel plot' in plots_selections:
                    exe_parallel_plot(model_config, "parallel_plot", plot_dir, save=True)
                    st.markdown("### Parallel Plot")

                    # Display the saved plot PDF
                    path = os.path.join(plot_dir, "parallel_plot.pdf")
                    pdf_viewer(path, width="80%", height=1400)

        st.markdown("---")
        if st.button("Generate other insights"):
            st.session_state.dir_submitted = False
            st.session_state.model_dir = ""
            st.session_state.pop("plots_selections", None)
            st.rerun()

def exe_streamlit_model_testing():
    st.header("Model Testing")
    st.markdown("""
    This setup allows you **test a previously trained and saved model** using **new or unseen input data**. This step helps you **evaluate model performance** beyond the training phase and validate generalization.

    Once a trained model is selected, this tab lets you:

    - Load the trained model's saved config
    - Provide new input data for testing
    - Reapply the correct **data tuning procedure** if required
    - Run model predictions and display a scatter plot for evaluation
    - Save and display the results

    ---

    **Select Tuning Type**
    This is **critical** if the model was trained on **tuned data**:
    - `None`: Use raw data without any tuning (only if the model was trained that way).

    ⚠️ **Make sure the tuning type and input structure match the training phase!**  
    If the trained model used tuned data, you must choose the **same tuning procedure** here. The same tuning process will be recreated on input data.

   The default results directory is: `addmo-automated-ml-regression/addmo_examples/results/model_streamlit_test/`

   
    """)
    for key in ["dir_submitted", "input_submitted", "tuning_submitted"]:
        if key not in st.session_state:
            st.session_state[key] = False

    for key in ["model_dir", "input_data", "tuning_type"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    if not st.session_state.dir_submitted:
        with st.form("Model Directory"):
            option = st.radio(
                "Select directory option for loading previously saved model for testing:",
                ("Default", "Custom")
            )

            if option == "Custom":
                directory = st.text_input("Enter the custom results directory path, the path should contain the experiment folder as well. For example: addmo_examples/results/test_raw_data/test_data_tuning/test_model_tuning ")

            submitted = st.form_submit_button("Submit")

            if submitted:
                if option == "Default":
                    st.session_state.model_dir = return_results_dir_model_tuning()
                    st.session_state.dir_submitted = True

                elif option == "Custom" and directory.strip():
                    st.session_state.model_dir = directory
                    st.session_state.dir_submitted = True

    if st.session_state.dir_submitted:
        directory = st.session_state.model_dir
        st.write(f"Using directory: {directory}")

        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        if not st.session_state.input_submitted:
            with st.form("Input Data"):
                option = st.radio("Select raw input data path for testing the saved model:",
                                  ("Default", "Custom"))
                st.text('The default raw data path is: addmo_examples/raw_input_data/InputData.xlsx')
                if option == "Default":
                    input_data_path = os.path.join(root_dir(), 'addmo_examples', 'raw_input_data', 'InputData.xlsx')
                else:
                    input_data_path = st.text_input("Enter the input data path for testing the saved model:")

                submitted = st.form_submit_button("Submit")

                if submitted:
                    st.session_state.input_data = input_data_path
                    st.session_state.input_submitted = True

        if st.session_state.input_submitted:
            with st.form("Tuning Type"):
                tuning_type = st.radio("Select type of tuning for the raw input data:",
                                       ("Auto", "Fixed", "None"))
                submitted = st.form_submit_button("Submit")

                if submitted:
                    st.session_state.tuning_type = tuning_type
                    st.session_state.tuning_submitted = True

        # Run model test once all inputs are gathered
        if st.session_state.tuning_submitted:
            error, saving_dir = model_test(st.session_state.model_dir, model_config, st.session_state.input_data,
                                           "model_streamlit_test", st.session_state.tuning_type)
            st.write(f"Results saved in: ", saving_dir)
            st.write(f"Error is: ", error)
            pdf_viewer(os.path.join(saving_dir, "model_fit_scatter.pdf"), width="80%")

        # Button to test another model
        st.markdown("---")
        if st.button("Generate other insights"):
            st.session_state.dir_submitted = False
            st.session_state.model_dir = ""
            st.session_state.pop("plots_selections", None)  # Optional: remove previous selections
            st.rerun()

def exe_streamlit_data_tuning_recreate():
    st.header("Recreate data tuning for new data")
    st.markdown("""

    This setup allows you to **recreate the exact data tuning process** applied during a previous experiment, using the saved tuning configuration.  

    Although data tuning is **automatically handled** in the **Model Testing tab**, this tab gives you **manual control** over the recreation of tuned datasets.  
    It's useful when:
    - You need a standalone tuned dataset for further analysis or visualization.
    - You want to verify how tuning was applied.
    - You want to reproduce preprocessing outside the model testing flow.

    > ⚠️ This tab **only supports tuning configurations saved by this app**.  
    > It cannot recreate tuning from externally trained models or configurations.

    ---""")

    if "tuning_type" not in st.session_state:
        st.session_state.tuning_type = None

    if "tuning_submitted" not in st.session_state:
        st.session_state.tuning_submitted = False
        st.session_state.dir_submitted = False
        st.session_state.model_dir = None

    if not st.session_state.tuning_submitted:
        st.subheader("Choose data tuning type")
        with st.form("Type of tuning"):
            tuning_type = st.radio(
                "Choose tuning recreate type for dataset, based on the existing saved tuning config file",
                ["Auto", "Fixed"],
                index=0,
                key="tuning_type_radio"
            )
            submitted = st.form_submit_button("Confirm")

        if submitted:
            st.session_state.tuning_type = tuning_type
            st.session_state.tuning_submitted = True

    if st.session_state.tuning_submitted:

            with st.form("Model Directory"):
                option = st.radio(
                "Select directory option for loading previously saved config for tuned data:",
                ("Default", "Custom"))

                directory = None
                if option == "Custom":
                    directory = st.text_input("Enter the custom results directory path. The path should contain the experiment folder as well. For example: addmo_examples/results/test_raw_data/test_data_tuning/test_model_tuning")

                submitted = st.form_submit_button("Submit")

            if submitted:
                if option == "Default" and st.session_state.tuning_type == "Auto":
                    st.session_state.model_dir = results_dir_data_tuning_auto()
                    st.session_state.dir_submitted = True

                elif option == "Default" and st.session_state.tuning_type == "Fixed":
                    st.session_state.model_dir = results_dir_data_tuning_fixed()
                    st.session_state.dir_submitted = True

                elif option == "Custom" and directory:
                    st.session_state.model_dir = directory
                    st.session_state.dir_submitted = True

    if st.session_state.dir_submitted:
        with st.form("Raw Input data"):
            input_data_path = st.text_input("Enter the raw input data path:")
            submitted = st.form_submit_button("Submit")
        if submitted:
            # Load data tuning config
            config_path = os.path.join(st.session_state.model_dir, "config.json")
            with open(config_path, 'r') as f:
                data_config = json.load(f)
                input_data_exp_name = data_config.get("name_of_raw_data")
                data_config["name_of_raw_data"] = "model_streamlit_test"

            if st.session_state.tuning_type == "Auto":
                tuned_x_new, y_new, new_config = data_tuning_recreate_auto(data_config, input_data_path,input_data_exp_name)
            else:
                tuned_x_new, tuned_y_new, new_config = data_tuning_recreate_fixed(data_config, input_data_path,input_data_exp_name)

            st.write(tuned_x_new)
            result_dir = results_model_streamlit_testing(input_data_exp_name)
            st.write('The tuned data is saved at:', result_dir)




    st.markdown("---")
    if st.button("Generate another tuning type"):
        st.session_state.tuning_type = None
        st.session_state.tuning_submitted = False
        st.session_state.dir_submitted = False
        st.session_state.model_dir = None
        st.rerun()

# Streamlit UI

st.set_page_config(
    page_title="ADDMO",
    page_icon="🔧",
    layout="wide"
)

st.title("ADDMO - Automated Data & Model Optimization")
st.markdown("""
Welcome to **ADDMO**, a AutoML toolkit designed for **time series regression tasks**.

This application helps you configure, run, and analyze the full machine learning pipeline — from **data preprocessing** to **model tuning**, with full documentation and visualization at each step.

---

### Workflow Overview

1. **Data Tuning (Auto/Fixed)**  
   Automates feature creation and selection, lag generation, scaling, and cleaning.

2. **Model Tuning**  
   Select models, configure hyperparameters, and run cross-validation.

3. **Data Insights**  
   Visualize time series input data, model predictions across combinations of two selected features and analyze relationships between features, target, and predictions.

4. **Model Testing**  
   Evaluate trained models on test data using intuitive metrics and plots.

5. **Data Tuning Recreate**  
   Rebuild previously used data tuning pipelines for reproducibility and fine-tuning.

---

Each module is modular and optional, enabling flexible experimentation and analysis.  
""")
tab = st.radio("Choose Tab", ["Data Tuning", "Model Tuning", "Insights", "Model Testing","Data Tuning Recreate"], horizontal=True)
if tab == "Data Tuning":
    st.header("Choose Data Tuning type")
    if "tuning_type" not in st.session_state:
        st.session_state.tuning_type = None

    if "tuning_submitted" not in st.session_state:
        st.session_state.tuning_submitted = False

    if not st.session_state.tuning_submitted:
        with st.form("Type of tuning"):
            tuning_type = st.radio(
                "Choose tuning type for dataset",
                ["Auto", "Fixed"],
                index=0,
                key="tuning_type_radio"
            )
            submitted = st.form_submit_button("Confirm")
        if submitted:
            st.session_state.tuning_type = tuning_type
            st.session_state.tuning_submitted = True
            st.rerun()

    # After submission
    else:
        if st.session_state.tuning_type == "Auto":
            exe_streamlit_data_tuning_auto()
        elif st.session_state.tuning_type == "Fixed":
            exe_streamlit_data_tuning_fixed()

        # Add a reset button to allow switching tuning types
        st.markdown("---")
        if st.button("Run another tuning type"):
            st.session_state.tuning_submitted = False
            st.session_state.tuning_type = None
            st.rerun()



if tab == "Model Tuning":
    exe_streamlit_model_tuning()

if tab=="Insights":
    exe_streamlit_data_insights()

if tab=="Model Testing":
    exe_streamlit_model_testing()


if tab=="Data Tuning Recreate":
    exe_streamlit_data_tuning_recreate()









