import base64
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
from addmo.util.load_save_utils import root_dir, create_or_clean_directory
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from streamlit_pdf_viewer import pdf_viewer
from addmo.util.definitions import results_dir_data_tuning_auto, results_dir_data_tuning_fixed, return_results_dir_model_tuning, results_model_streamlit_testing, results_dir_data_tuning, results_dir
from addmo_examples.executables.exe_data_insights import exe_time_series_plot, exe_parallel_plot, exe_carpet_plots, exe_scatter_carpet_plots, exe_interactive_parallel_plot
from addmo.s4_model_testing.model_testing import model_test, data_tuning_recreate_fixed, data_tuning_recreate_auto
from addmo.util.load_save import load_data


def exe_streamlit_data_tuning_auto():
    st.header("Auto Data Tuning Configuration")
    st.markdown("""
       
        This setup allows you configure and launch a fully-automated data tuning pipeline.
        Use the form below to customize:
        - Preprocessing & feature engineering (e.g., lag generation, variance filtering)
        - Feature selection strategies
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
    - **`create_manual_feature_lags`**: Add lags for specific input variables.
    - **`feature_lags`**: Define feature-specific lags: e.g., `{feature: [1, 2]}`. 
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
    
        
    #### Automated Feature Selection
    Learns which features matter while the model is training:
    - **`recursive_embedded_number_features_to_select`**: Number of features to select in recursive feature elimination.
    - **`wrapper_sequential_feature_selection`**: Forward/backward wrapper selection.
    - **`sequential_direction`**: `'forward'` or `'backward'`.
    - **`min_increase_for_wrapper`**: Minimum performance gain needed to accept a feature.
    
    """)
    auto_tuning_config = pydantic_input("Auto", DataTuningAutoSetup)

    # Output strategy
    auto_tuning_config_obj = DataTuningAutoSetup(**auto_tuning_config)
    st.session_state.auto_tuning_config = auto_tuning_config_obj

    st.subheader("Automatic feature selection")
    st.markdown(
        """
        This step performs **recursive feature elimination (RFE)** using a Random Forest model to automatically select the most relevant features.

        You can choose between two selection strategies:

        - **By minimum number of features**:  
          Features are eliminated one at a time until only the specified number (`recursive_embedded_number_features_to_select`) remain.

        - **By performance improvement**:  
          Features are eliminated until the model's cross-validation score no longer improves significantly, based on a threshold (`min_increase_for_wrapper`).
        """
    )
    feature_selection_strategy = st.selectbox(
        "Select the feature selection strategy:",
        ["Select an option", "Minimum number of features", "Minimum score improvement"]
    )

    if feature_selection_strategy == "Minimum number of features":
        st.session_state.auto_tuning_config.filter_recursive_by_count = True
    elif feature_selection_strategy == "Minimum score improvement":
        st.session_state.auto_tuning_config.filter_recursive_by_score = True


    output_dir = results_dir_data_tuning(auto_tuning_config_obj)
    st.subheader("Output Directory Strategy")
    st.write("The default directory for saving the tuned data is : " )
    st.code(output_dir)
    strategy = st.selectbox(
        "Choose how to handle existing results in the output directory:",
        ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
    )

    if strategy != "Select an option":
        overwrite_strategy = strategy.split()[-1].strip("()")


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
            return None

        # Save config to the expected JSON location
        config_path = os.path.join(
            root_dir(), 'addmo', 's1_data_tuning_auto', 'config', 'data_tuning_auto_config.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            f.write(st.session_state.auto_tuning_config.model_dump_json(indent=4))

        st.success("✅ Configuration saved!")


        # Run the tuning process
        with st.spinner("Running data tuning..."):
            exe_data_tuning_auto(overwrite_strategy)
            # Load default saving path for plot
            base_path = os.path.join(output_dir, "tuned_xy_auto.pdf")
            pdf_viewer(base_path, width="80%", height=855)

            # Path to the optional 2-week plot
            two_weeks_path = os.path.join(output_dir, "tuned_xy_auto_2weeks.pdf")
            if os.path.exists(two_weeks_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(two_weeks_path, width="80%", height=855)
            st.success("✅ Data tuning completed!")

    return output_dir

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
    fixed_tuning_config = pydantic_input(key="Config Setup", model=DataTuningFixedConfig)
    fixed_config_tuning_obj = DataTuningFixedConfig(**fixed_tuning_config)
    output_dir = results_dir_data_tuning(fixed_config_tuning_obj)
    st.subheader("Output Directory Strategy")
    st.write("The default directory for saving the tuned data is : ")
    st.code(output_dir)
    strategy = st.selectbox(
        "Choose how to handle existing results in the output directory:",
        ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
    )

    if strategy != "Select an option":
        overwrite_strategy = strategy.split()[-1].strip("()")

    if st.button("Run Fixed Data Tuning"):
        missing_fields = []
        for field_name, field_value in fixed_tuning_config.items():
            # Check if field is missing or empty
            if isinstance(field_value, (list)) and not field_value:
                missing_fields.append(field_name)
            elif field_value in [None, ""]:
                missing_fields.append(field_name)
        # If any fields are missing, show an error
        if missing_fields:
            st.error(f"❌ The following required fields are missing or empty: {', '.join(missing_fields)}")
            return

        # Save config to the expected JSON location
        config_path = os.path.join(
            root_dir(), 'addmo', 's2_data_tuning', 'config', 'data_tuning_config.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            f.write(fixed_config_tuning_obj.model_dump_json(indent=4))

        st.success("✅ Configuration saved!")

        # Run the tuning process
        with st.spinner("Running data tuning.."):
            exe_data_tuning_fixed(overwrite_strategy)
            # Load default saving path for plot
            # Load default saving path for plot
            base_path = os.path.join(output_dir, "tuned_xy_fixed.pdf")
            pdf_viewer(base_path, width="80%", height=855)

            # Path to the optional 2-week plot
            two_weeks_path = os.path.join(output_dir, "tuned_xy_fixed_2weeks.pdf")
            if os.path.exists(two_weeks_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(two_weeks_path, width="80%", height=855)
            st.success("✅ Data tuning completed!")

    return output_dir

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


    if 'model_config_data' not in st.session_state:
        st.session_state.model_config_data = None
    if 'model_config_saved' not in st.session_state:
        st.session_state.model_config_saved = False
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'overwrite_strategy' not in st.session_state:
        st.session_state.overwrite_strategy = None

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
        type_of_tuning = st.selectbox("Which tuning would you like to use?", ["choose", "Auto", "Fixed"])

        if type_of_tuning == "Auto":
            st.text(
                "The default directory used for loading the data is: addmo_examples/results/test_raw_data/data_tuning_experiment_auto")
            path_type = st.selectbox(
                "Would you like to use the default saving path for loading the input data?",
                ["Select an option", "Yes", "No"],
                help="The path for loading input data depends on the experiment names defined above. "
                     "If the default saving paths and experiment folder names are not changed during data tuning and model config, choose Default")
            if path_type == "Yes":
                model_config_data["abs_path_to_data"] = os.path.join(
                    results_dir_data_tuning_auto(model_config_data['name_of_raw_data']),
                    "tuned_xy_auto.csv")
                st.success("✅ Tuned data path set in config.")
            elif path_type == "No":
                model_config_data["abs_path_to_data"] = st.text_input('path')
                st.success("✅ Tuned data path set in config.")

        elif type_of_tuning == "Fixed":
            st.text(
                "The default directory used for loading the data is: addmo_examples/results/test_raw_data/data_tuning_experiment_fixed")

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

    #Save Config
    if st.button("Save Model Config"):
        st.session_state.model_config_data = model_config_data
        st.session_state.model_config_saved = True

        model_config_obj = ModelTuningExperimentConfig(**model_config_data)
        output_dir = return_results_dir_model_tuning(
            model_config_obj.name_of_raw_data,
            model_config_obj.name_of_data_tuning_experiment,
            model_config_obj.name_of_model_tuning_experiment
        )
        st.session_state.output_dir = output_dir

        config_path = os.path.join(root_dir(), 'addmo', 's3_model_tuning', 'config', 'model_tuning_config.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(model_config_obj.model_dump_json(indent=4))

        st.success("✅ Model configuration saved!")

    #overwrite strategy
    if st.session_state.model_config_saved:
        st.subheader("Output Directory Strategy")
        st.code(st.session_state.output_dir)

        strategy = st.selectbox(
            "Choose how to handle existing results in the output directory:",
            ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
        )

        if strategy != "Select an option":
            st.session_state.overwrite_strategy = strategy.split()[-1].strip("()")

    # run model tuning
    if (
            st.session_state.model_config_saved
            and st.session_state.overwrite_strategy is not None
            and st.button("Run Model Tuning")
    ):
        model_config_data = st.session_state.model_config_data
        model_config_obj = ModelTuningExperimentConfig(**model_config_data)
        output_dir = st.session_state.output_dir
        overwrite_strategy = st.session_state.overwrite_strategy

        with st.spinner("Running model tuning..."):
            exe_model_tuning(overwrite_strategy, model_config_obj)
            plot_image_path = os.path.join(output_dir, "model_fit_scatter.pdf")

            st.markdown("### Model Fit Plot")
            pdf_viewer(plot_image_path, width="80%", height=855)
            st.success("✅ Model tuning completed!")
    return st.session_state.output_dir

def generate_addmo_insights():
    if "dir_submitted" not in st.session_state:
        st.session_state.dir_submitted = False
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = ""
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = None

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
        plot_dir = os.path.join(directory, 'plots')
        st.session_state.output_dir = plot_dir
        config_path = os.path.join(directory, "config.json")
        metadata_path = os.path.join(directory, "best_model_metadata.json")
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        with open(metadata_path, 'r') as f:
            model_metadata_config = json.load(f)

        # Initialize session state for plots if not exists
        if "plots_selections" not in st.session_state:
            st.session_state.plots_selections = []
        if "show_plots_form" not in st.session_state:
            st.session_state.show_plots_form = True
        if "requires_bounds_form" not in st.session_state:
            st.session_state.requires_bounds_form = ""
        if "plots_confirmed" not in st.session_state:
            st.session_state.plots_confirmed = False

        # Show plots selection form only if needed
        with st.form("Choose plots for generating insights"):
            plots_selections = st.multiselect(
                "Select the plots which you'd like to see",
                options=['Time Series plot',
                         'Predictions surface plot',
                         'Predictions parallel plot',
                         'Prediction surface with feature interaction scatter plot'],
                default=st.session_state.get("plots_selections", [])
            )
            submitted = st.form_submit_button("Confirm plots")

            if submitted:
                st.session_state.plots_selections = plots_selections
                st.session_state.plots_confirmed = True
                st.session_state.requires_bounds_form = any(
                    plot in st.session_state.plots_selections
                    for plot in ['Predictions carpet plot', 'Prediction surface with feature interaction scatter plot']
                )
                # st.rerun()

        if st.session_state.plots_confirmed:
            st.markdown(f"Using saved directory: `{directory}`")
            st.markdown(f"Plots will be saved in: `{st.session_state.output_dir}`")

            if st.session_state.requires_bounds_form:
                if "bounds_choice" not in st.session_state:
                    st.session_state.bounds_choice = "Select an option"

                bounds_choice = st.selectbox(
                    "Choose bounds for the columns of data",
                    ["Select an option", "Choose min and max of existing data as bounds", "Define custom bounds"],
                    index=["Select an option", "Choose min and max of existing data as bounds",
                           "Define custom bounds"].index(st.session_state.bounds_choice)
                )
                st.session_state.bounds_choice = bounds_choice

                if bounds_choice == "Define custom bounds":
                    feature_columns = model_metadata_config.get("features_ordered", [])

                    if "custom_bounds" not in st.session_state:
                        st.session_state.custom_bounds = {}

                    if "defaults" not in st.session_state:
                        st.session_state.defaults = {}

                    with st.form("Custom bounds and defaults input"):
                        for column in feature_columns:
                            default_min = st.session_state.custom_bounds.get(column, [0.0, 1.0])[0]
                            default_max = st.session_state.custom_bounds.get(column, [0.0, 1.0])[1]
                            default_val = st.session_state.defaults.get(column, 0.0)

                            st.markdown(f"**{column}**")
                            min_val = st.number_input(f"Min for {column}", key=f"{column}_min", value=default_min)
                            max_val = st.number_input(f"Max for {column}", key=f"{column}_max", value=default_max)
                            default_input = st.number_input(f"Default value for {column}", key=f"{column}_default",
                                                            value=default_val)

                            st.session_state.custom_bounds[column] = [min_val, max_val]
                            st.session_state.defaults[column] = default_input

                        if st.form_submit_button("Confirm bounds and defaults"):
                            st.session_state.bounds = st.session_state.custom_bounds
                            st.session_state.final_defaults = st.session_state.defaults
                            st.success("Custom bounds and default values have been saved.")

                elif bounds_choice == "Choose min and max of existing data as bounds":
                    st.session_state.bounds = None
                    st.session_state.defaults = None
                    if not model_config.get("abs_path_to_data"):
                        st.warning(
                            "Absolute path of data is required in order to create bounds and default values. "
                            "Please re-run the form again and specify the path to data for which the plots need to be created."
                        )
            if 'Predictions surface plot' in st.session_state.plots_selections:
                exe_carpet_plots(directory, "predictions_carpet",
                                 st.session_state.output_dir, save=True,
                                 bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
                st.markdown("### Carpet Plot")
                pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_carpet.pdf"), width="80%")

            if 'Prediction surface with feature interaction scatter plot' in st.session_state.plots_selections:

                exe_scatter_carpet_plots(directory, "predictions_scatter_surface",
                                         st.session_state.output_dir, save=True,
                                         bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
                st.markdown("### Surface with scatter Plot")
                pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_scatter_surface.pdf"), width="80%")

            if 'Time Series plot' in st.session_state.plots_selections:
                    exe_time_series_plot(model_config, "training_data_time_series", st.session_state.output_dir, save=True)
                    st.markdown("### Time Series Data Plot")

                    # Path to the main plot
                    base_path = os.path.join(st.session_state.output_dir, "training_data_time_series.pdf")
                    pdf_viewer(base_path, width="80%", height=855)

                    # Path to the optional 2-week plot
                    two_weeks_path = os.path.join(st.session_state.output_dir,
                                                  "training_data_time_series_2weeks.pdf")
                    if os.path.exists(two_weeks_path):
                        st.markdown("### Zoomed View: Time Series (2 Weeks)")
                        pdf_viewer(two_weeks_path, width="80%", height=855)

            if 'Predictions parallel plot' in st.session_state.plots_selections:
                fig = exe_interactive_parallel_plot(model_config, "parallel_plot", st.session_state.output_dir, save=True)
                scrollable_html = f"""
                <div style="overflow-x: auto; width: 100%;">
                    <div style="min-width: 1200px;"> <!-- Set to match or exceed your plot width -->
                        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                """

                st.markdown("### Interactive Parallel Plot")
                st.components.v1.html(scrollable_html, height=700)
                # Display the saved plot PDF
                # path = os.path.join(st.session_state.output_dir, "parallel_plot.pdf")
                # pdf_viewer(path, width="80%", height=1400)

    return st.session_state.output_dir

def input_custom_model_config():
    st.subheader("Enter Configuration for External Model Insights")
    st.markdown("Fill in the required details to generate insights from a model **not trained by this application**.")
    st.markdown("""
    **Note**:
    - You need to specify the data path for creating carpet plots if the bounds and default values for each feature are not known and need to be calculated automatically.
    - You need to provide the input data for the creation of other plots (e.g., Time Series Plot and Parallel Plot).
    """)
    with st.form("external_model_config_form"):
        abs_path_to_data = st.text_input("Absolute path to the data file (e.g., .xlsx) (Not required for carpet plots)", key="abs_path_to_data")
        name_of_target = st.text_input("Name of the target variable", key="name_of_target")
        start_train_val = st.text_input("Start date/time for train/validation (e.g., 2016-08-01 00:00) (Not required for carpet plots)", key="start_train_val")
        stop_train_val = st.text_input("Stop date/time for test (e.g., 2016-08-14 23:45) (Not required for carpet plots)", key="end_test")
        features_ordered_input = st.text_input("Enter the list of ordered columns in the data, except the target column", key="features_ordered")
        base_class = st.selectbox("Base class of regressor", options=["ScikitMLP", "Keras"],
                                  help="Choose the type of model class used to train the uploaded model")
        model_file = st.file_uploader("Upload your trained model (.keras or .joblib)", type=["keras", "joblib"])

        submitted = st.form_submit_button("Save Configuration")
        features_ordered = [feat.strip() for feat in features_ordered_input.split(",") if feat.strip()]
        saving_dir = create_or_clean_directory(os.path.join(results_dir(), "model_plots"))
        if base_class == "Keras":
            base_class = "SciKerasSequential"

    if submitted:
        data_config = {
            "saving_dir": saving_dir,
            "name_of_target": name_of_target,
        }
        if abs_path_to_data:
            data_config["abs_path_to_data"] = abs_path_to_data
        if start_train_val:
            data_config["start_train_val"] = start_train_val
        if stop_train_val:
            data_config["stop_train_val"] = stop_train_val

        model_metadata_config = {
            "addmo_class": base_class,
            "target_name": name_of_target,
            "features_ordered": features_ordered
        }
        st.session_state["external_data_config"] = data_config
        st.session_state["model_metadata_config"] = model_metadata_config

        metadata_config_path = os.path.join(saving_dir, "best_model_metadata.json")
        with open(metadata_config_path, "w") as f:
            json.dump(model_metadata_config, f, indent=4)

        data_config_path = os.path.join(saving_dir, "config.json")
        with open(data_config_path, "w") as f:
            json.dump(data_config, f, indent=4)

        model_path = os.path.join(saving_dir, model_file.name)
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())

        st.success("Configuration saved successfully!")


        return data_config, model_path, model_metadata_config
    return None

def generate_external_insights(config: dict, model_dir: str, model_metadata_config: dict ):
    """
    Generates plots for a model not trained by the current app, using the provided config and model directory.
    """
    if "external_config_submitted" not in st.session_state:
        st.session_state.external_config_submitted = True  # Set once config is passed

    if "output_dir" not in st.session_state:
        st.session_state.output_dir = config.get("saving_dir")

    if "plots_selections" not in st.session_state:
        st.session_state.plots_selections = []
    if "show_plots_form" not in st.session_state:
        st.session_state.show_plots_form = True
    if "show_bounds_form" not in st.session_state:
        st.session_state.show_bounds_form = False
    if "plots_confirmed" not in st.session_state:
        st.session_state.plots_confirmed = False

        # Show plots selection form only if needed
    with st.form("Choose plots for generating insights"):
        plots_selections = st.multiselect(
            "Select the plots which you'd like to see",
            options=['Time Series plot for training data',
                     'Predictions carpet plot',
                     'Predictions parallel plot'],
            default=st.session_state.get("plots_selections", [])
        )
        submitted = st.form_submit_button("Confirm plots")

        if submitted:
            st.session_state.plots_selections = plots_selections
            st.session_state.plots_confirmed = True
            st.session_state.show_bounds_form = 'Predictions carpet plot' in plots_selections
            st.rerun()

    if st.session_state.plots_confirmed:
        st.markdown(f"Using saved directory: `{model_dir}`")
        st.markdown(f"Plots will be saved in: `{st.session_state.output_dir}`")

        if 'Predictions carpet plot' in st.session_state.plots_selections:
            if "bounds_choice" not in st.session_state:
                st.session_state.bounds_choice = "Select an option"

            bounds_choice = st.selectbox(
                "Choose bounds for the columns of data",
                ["Select an option", "Choose min and max of existing data as bounds", "Define custom bounds"],
                index=["Select an option", "Choose min and max of existing data as bounds",
                       "Define custom bounds"].index(st.session_state.bounds_choice)
            )
            st.session_state.bounds_choice = bounds_choice

            if bounds_choice == "Define custom bounds":
                feature_columns = model_metadata_config.get("features_ordered", [])

                if "custom_bounds" not in st.session_state:
                    st.session_state.custom_bounds = {}

                if "defaults" not in st.session_state:
                    st.session_state.defaults = {}

                with st.form("Custom bounds and defaults input"):
                    for column in feature_columns:
                        # Defaults for bounds
                        default_min = st.session_state.custom_bounds.get(column, [0.0, 1.0])[0]
                        default_max = st.session_state.custom_bounds.get(column, [0.0, 1.0])[1]

                        # Default for default value
                        default_val = st.session_state.defaults.get(column, 0.0)

                        st.markdown(f"**{column}**")
                        min_val = st.number_input(f"Min for {column}", key=f"{column}_min", value=default_min)
                        max_val = st.number_input(f"Max for {column}", key=f"{column}_max", value=default_max)
                        default_input = st.number_input(f"Default value for {column}", key=f"{column}_default",
                                                        value=default_val)

                        # Save to session state
                        st.session_state.custom_bounds[column] = [min_val, max_val]
                        st.session_state.defaults[column] = default_input

                    if st.form_submit_button("Confirm bounds and defaults"):
                        st.session_state.bounds = st.session_state.custom_bounds
                        st.session_state.final_defaults = st.session_state.defaults
                        st.success("Custom bounds and default values have been saved.")

            elif bounds_choice == "Choose min and max of existing data as bounds":
                st.session_state.bounds = None
                st.session_state.defaults = None
                if not config.get("abs_path_to_data"):
                    st.write("Absolute path of data is required in order to create bounds and default values. Please re-run the form again and specify the path to data for which the plots need to be created.")

            if st.button("Generate Carpet Plot"):
                exe_carpet_plots(st.session_state.output_dir, "predictions_carpet_new",
                                 st.session_state.output_dir, save=True,
                                 bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
                st.markdown("### Carpet Plot")
                pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_carpet_new.pdf"),
                           width="80%")

        if 'Time Series plot for training data' in st.session_state.plots_selections:
            exe_time_series_plot(config, "training_data_time_series", st.session_state.output_dir,
                                 save=True)
            st.markdown("### Time Series Data Plot")
            base_path = os.path.join(st.session_state.output_dir, "training_data_time_series.pdf")
            pdf_viewer(base_path, width="80%", height=855)
            two_weeks_path = os.path.join(st.session_state.output_dir,
                                          "training_data_time_series_2weeks.pdf")
            if os.path.exists(two_weeks_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(two_weeks_path, width="80%", height=855)

        if 'Predictions parallel plot' in st.session_state.plots_selections:
            exe_parallel_plot(config, "parallel_plot", st.session_state.output_dir, True, st.session_state.model_dir)
            st.markdown("### Parallel Plot")
            path = os.path.join(st.session_state.output_dir, "parallel_plot.pdf")
            pdf_viewer(path, width="80%", height=1400)



    return st.session_state.output_dir

def exe_streamlit_data_insights():

    st.header('Generate Insights for the tuned saved models')
    st.markdown("""
    This setup allows you to **generate insightful visualizations** based on the results of previously trained and saved models. It's especially useful for **analyzing model performance** and **understanding the effect of features** through various plots.
    Once you load a previously saved model's config file and result directory, you can:

    - **Visualize Time Series Performance**  
      Plot predictions vs actual target values across time for the training data.
    
    - **Create Surface Plots**  
      These 3D plots help visualize the interaction between multiple features and the predicted values.
    
    - **Draw Parallel Plots**  
      Visualize the relationships between feature combinations and their contribution to the prediction.
    
    **Note**:
    - Ensure that your selected directory contains a valid trained model config and model results.
    - The plots depend on a successful model tuning run — you must have completed that step first.
    - If you want to generate plots for models not trained on this application, please ensure to upload the trained model and the corresponding model class.
     """)
    st.session_state.path = None
    if "choose_plotting_type" not in st.session_state:
        st.session_state.choose_plotting_type = "Select an option"

    choose_plotting_type= st.selectbox("Would you like to generate insights for a model trained by this application or generate insights for other models?",
                                       ["Select an option", "Generate insights for a model trained by this application","Generate insights for other models"], key="choose_plotting_type")

    if st.session_state.choose_plotting_type == "Generate insights for a model trained by this application":
         st.session_state.path = generate_addmo_insights()
    elif st.session_state.choose_plotting_type == "Generate insights for other models":
        if "external_config_submitted" not in st.session_state:
            result = input_custom_model_config()
            if result is not None:
                config, model_dir, model_metadata_config = result
                st.session_state.config = config
                st.session_state.model_dir = model_dir
                st.session_state.model_metadata_config = model_metadata_config
                st.session_state.output_dir = config.get("saving_dir")
                st.session_state.external_config_submitted = True
                st.rerun()
        else:
            st.session_state.path = generate_external_insights(st.session_state.config, st.session_state.model_dir, st.session_state.model_metadata_config )



    return st.session_state.path

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

    #### Select Tuning Type
    This is **critical** if the model was trained on **tuned data**:
    - `None`: Use raw data without any tuning (only if the model was trained that way).

    ⚠️ **Make sure the tuning type and input structure match the training phase!**  
    If the trained model used tuned data, you must choose the **same tuning procedure** here. The same tuning process will be recreated on input data.

   The default results directory is: `addmo-automated-ml-regression/addmo_examples/results/model_streamlit_test/model_testing`

   
    """)
    for key in ["dir_submitted", "input_submitted", "tuning_path_confirmed", "tuning_submitted"]:
        if key not in st.session_state:
            st.session_state[key] = False
    st.session_state.output_dir = results_model_streamlit_testing("model_testing")
    for key in ["model_dir", "input_data", "tuning_type", "output_dir", "custom_tuning_path", "tuning_path_type","tuning_type_selected", "saving_dir"]:
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

        if st.session_state.input_submitted and not st.session_state.tuning_type_selected:
            with st.form("Tuning Type"):
                tuning_type = st.radio("Select type of tuning for the raw input data:",
                                       ("Auto", "Fixed", "None"))
                submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.tuning_type = tuning_type
                if tuning_type in ("Auto", "Fixed"):
                    st.session_state.tuning_type_selected = True
                else:
                    st.session_state.tuning_path_confirmed = True  # No tuning needed

        if st.session_state.tuning_type_selected and not st.session_state.tuning_path_confirmed:
            with st.form("Tuning Path"):
                path_type = st.selectbox("Use default saving path for loading the input data?",
                                         ["Select an option", "Yes", "No"],
                                         index=0)
                st.session_state.tuning_path_type = path_type

                abs_path = None
                path_set = False

                model_config["abs_path_to_data"] = ""

                if path_type == "Yes":
                    if st.session_state.tuning_type == "Auto":
                        abs_path = os.path.join(
                            results_dir_data_tuning_auto(model_config["name_of_raw_data"]),
                            "tuned_xy_auto.csv")
                    elif st.session_state.tuning_type == "Fixed":
                        abs_path = os.path.join(
                            results_dir_data_tuning_fixed(model_config["name_of_raw_data"]),
                            "tuned_xy_fixed.csv")
                    path_set = True

                elif path_type == "No":
                    custom_path = st.text_input("Enter custom path for tuned data:")
                    st.session_state.custom_tuning_path = custom_path
                    if custom_path.strip():
                        abs_path = custom_path
                        path_set = True

                confirm = st.form_submit_button("Confirm Path")

                if confirm:
                    if path_set and abs_path:
                        model_config["abs_path_to_data"] = abs_path
                        st.session_state.model_config = model_config
                        st.session_state.tuning_submitted = True
                        st.session_state.tuning_path_confirmed = True
                        st.success("✅ Tuned data path confirmed!")
                    else:
                        st.error("❌ Please provide a valid path before confirming.")

        # Run model test when all inputs are gathered
        if st.session_state.tuning_submitted or st.session_state.tuning_path_confirmed:
            error, saving_dir = model_test(st.session_state.model_dir, model_config, st.session_state.input_data,
                                           "model_streamlit_test", st.session_state.tuning_type)

            st.session_state.saving_dir = saving_dir
            st.write(f"Results saved in: ", saving_dir)
            st.write(f"Error is: ", error)
            pdf_viewer(os.path.join(saving_dir, "model_fit_scatter.pdf"), width="80%")

    return st.session_state.saving_dir

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
        st.session_state.saving_dir = None

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
            st.session_state.saving_dir = result_dir

    return st.session_state.saving_dir



# Streamlit UI

st.set_page_config(
    page_title="ADDMO",
    page_icon=os.path.join(root_dir(), 'staticfiles', '230718 Logo ADDMo-01.png'),
    layout="wide"
)

st.markdown("""
    <style>
        /* Remove top whitespace */
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{base64.b64encode(open(os.path.join(root_dir(), 'staticfiles', 'logo.png'), 'rb').read()).decode()}" 
             style="width: 1000px;" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
        <h1 style='margin: 0; padding-left: 0; padding-top: 12px; line-height: 1;'>
            ADDMO - Automated Data & Model Optimization
        </h1>
        """, unsafe_allow_html=True)


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

if 'last_saved_path' not in st.session_state:
    st.session_state.last_saved_path = ""
st.write("📁 Last Saved Path")
st.code(st.session_state.get("last_saved_path", "No path saved yet."))

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
            last_saved_path = exe_streamlit_data_tuning_auto()
            st.session_state.last_saved_path = last_saved_path

        elif st.session_state.tuning_type == "Fixed":
            last_saved_path = exe_streamlit_data_tuning_fixed()
            st.session_state.last_saved_path = last_saved_path

        # Add a reset button to allow switching tuning types
        st.markdown("---")
        if st.button("Run another tuning type"):
            for key in list(st.session_state.keys()):
                if key != "last_saved_path":
                    del st.session_state[key]



if tab == "Model Tuning":
    last_saved_path = exe_streamlit_model_tuning()
    st.session_state.last_saved_path = last_saved_path

    st.markdown("---")
    if st.button("Run another model tuning"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if tab=="Insights":
    last_saved_path = exe_streamlit_data_insights()
    st.session_state.last_saved_path = last_saved_path

    st.markdown("---")
    if st.button("Generate other insights"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()

if tab=="Model Testing":
    last_saved_path = exe_streamlit_model_testing()
    st.session_state.last_saved_path = last_saved_path
    st.markdown("---")
    if st.button("Test another model"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()


if tab=="Data Tuning Recreate":
    last_saved_path = exe_streamlit_data_tuning_recreate()
    st.session_state.last_saved_path = last_saved_path
    st.markdown("---")
    if st.button("Generate another tuning type"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()









