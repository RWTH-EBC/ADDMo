import os
import json
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from streamlit_pydantic import pydantic_input

# Import the existing configuration class
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.util.load_save_utils import root_dir


def load_config_from_json(config_path):
    """Load configuration from the JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        st.warning(f"Could not load configuration from file: {e}")
        return None


def save_config_to_json(config_data, config_path):
    """Save configuration to JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False


def run_data_tuning():
    """Execute the data tuning process."""
    try:
        st.info("Running data tuning process...")

        # Import the execution module
        from exe_data_tuning_auto import exe_data_tuning_auto

        with st.spinner("Processing data tuning..."):
            # Run the data tuning process
            exe_data_tuning_auto()

        st.success("Data tuning completed successfully!")
    except ImportError:
        st.error("Could not import the data tuning module. Make sure the files are in the correct location.")
    except Exception as e:
        st.error(f"Error during data tuning: {e}")


def render_global_section(config_obj):
    """Render the global section of the configuration."""
    st.subheader("Global Settings")

    col1, col2 = st.columns(2)
    with col1:
        config_obj.name_of_raw_data = st.text_input(
            "Name of Raw Data",
            value=config_obj.name_of_raw_data,
            help="Set name of the folder where the experiments shall be saved"
        )

        config_obj.name_of_tuning = st.text_input(
            "Name of Tuning",
            value=config_obj.name_of_tuning,
            help="Set name of the experiments series"
        )

    with col2:
        config_obj.abs_path_to_data = st.text_input(
            "Path to Data",
            value=config_obj.abs_path_to_data,
            help="Path to the file that has the system_data"
        )

        config_obj.name_of_target = st.text_input(
            "Target Variable Name",
            value=config_obj.name_of_target,
            help="Name of the target variable"
        )

    return config_obj


def render_feature_construction(config_obj):
    """Render the feature construction section of the configuration."""
    st.subheader("Feature Construction")

    col1, col2 = st.columns(2)
    with col1:
        config_obj.create_differences = st.checkbox(
            "Create Differences",
            value=config_obj.create_differences,
            help="Feature difference creation (building the derivative of the features)"
        )

        config_obj.create_manual_target_lag = st.checkbox(
            "Create Manual Target Lag",
            value=config_obj.create_manual_target_lag,
            help="Manual construction of target lags"
        )

        target_lag_str = ", ".join(map(str, config_obj.target_lag))
        updated_target_lag = st.text_input(
            "Target Lag",
            value=target_lag_str,
            help="Array of lags for the target (comma-separated integers)"
        )
        try:
            config_obj.target_lag = [int(x.strip()) for x in updated_target_lag.split(",")]
        except:
            st.error("Target lag must be comma-separated integers")

    with col2:
        config_obj.create_automatic_timeseries_target_lag = st.checkbox(
            "Create Automatic Timeseries Target Lag",
            value=config_obj.create_automatic_timeseries_target_lag,
            help="Automatic construction of time series target lags"
        )

        config_obj.minimum_target_lag = st.number_input(
            "Minimum Target Lag",
            value=config_obj.minimum_target_lag,
            min_value=1,
            help="Minimal target lag which shall be considered"
        )

        config_obj.create_automatic_feature_lags = st.checkbox(
            "Create Automatic Feature Lags",
            value=config_obj.create_automatic_feature_lags,
            help="Automatic construction of feature lags via wrapper"
        )

    # Special handling for feature_lags (complex nested dict)
    config_obj.create_manual_feature_lags = st.checkbox(
        "Create Manual Feature Lags",
        value=config_obj.create_manual_feature_lags,
        help="Manual construction of feature lags"
    )

    if config_obj.create_manual_feature_lags:
        feature_lags_str = json.dumps(config_obj.feature_lags, indent=2)
        updated_feature_lags_str = st.text_area(
            "Feature Lags (JSON format)",
            value=feature_lags_str,
            height=150,
            help="Feature lags in format {var_name: [lags]}"
        )
        try:
            config_obj.feature_lags = json.loads(updated_feature_lags_str)
        except:
            st.error("Invalid JSON format for feature lags")

    # Additional feature lag parameters
    col1, col2 = st.columns(2)
    with col1:
        config_obj.minimum_feature_lag = st.number_input(
            "Minimum Feature Lag",
            value=config_obj.minimum_feature_lag,
            min_value=1,
            help="Minimum feature lag"
        )

    with col2:
        config_obj.maximum_feature_lag = st.number_input(
            "Maximum Feature Lag",
            value=config_obj.maximum_feature_lag,
            min_value=1,
            help="Maximum feature lag"
        )

    return config_obj


def render_feature_selection(config_obj):
    """Render the feature selection section of the configuration."""
    st.subheader("Feature Selection")

    # Manual feature selection
    config_obj.manual_feature_selection = st.checkbox(
        "Manual Feature Selection",
        value=config_obj.manual_feature_selection,
        help="Manual selection of Features by their Column number"
    )

    if config_obj.manual_feature_selection:
        selected_features_str = json.dumps(config_obj.selected_features, indent=2)
        updated_selected_features_str = st.text_area(
            "Selected Features (JSON array format)",
            value=selected_features_str,
            height=150,
            help="Variable names of the features to keep"
        )
        try:
            config_obj.selected_features = json.loads(updated_selected_features_str)
        except:
            st.error("Invalid JSON format for selected features")

    # Feature filtering options
    col1, col2 = st.columns(2)
    with col1:
        config_obj.filter_low_variance = st.checkbox(
            "Filter Low Variance",
            value=config_obj.filter_low_variance,
            help="Remove features with low variance"
        )

        if config_obj.filter_low_variance:
            config_obj.low_variance_threshold = st.number_input(
                "Low Variance Threshold",
                value=config_obj.low_variance_threshold,
                min_value=0.0,
                format="%.3f",
                help="Variance threshold for feature removal"
            )

        config_obj.filter_ICA = st.checkbox(
            "Filter ICA",
            value=config_obj.filter_ICA,
            help="Filter: Independent Component Analysis(ICA)"
        )

    with col2:
        config_obj.filter_univariate = st.checkbox(
            "Filter Univariate",
            value=config_obj.filter_univariate,
            help="Filter univariate by scikit-learn"
        )

        if config_obj.filter_univariate:
            config_obj.univariate_score_function = st.selectbox(
                "Univariate Score Function",
                options=["mutual_info_regression", "f_regression"],
                index=0 if config_obj.univariate_score_function == "mutual_info_regression" else 1,
                help="Score function for univariate feature selection"
            )

            config_obj.univariate_search_mode = st.selectbox(
                "Univariate Search Mode",
                options=["percentile", "k_best"],
                index=0 if config_obj.univariate_search_mode == "percentile" else 1,
                help="Search mode for univariate feature selection"
            )

            config_obj.univariate_filter_params = st.number_input(
                "Univariate Filter Params",
                value=config_obj.univariate_filter_params,
                min_value=1,
                help="Percent of features to keep or number of top features to keep"
            )

    # Embedded feature selection
    col1, col2 = st.columns(2)
    with col1:
        config_obj.embedded_model = st.selectbox(
            "Embedded Model",
            options=["RF", "MLP", "LR", "SVR"],
            index=["RF", "MLP", "LR", "SVR"].index(config_obj.embedded_model),
            help="Estimator for use in all embedded methods"
        )

        config_obj.filter_recursive_embedded = st.checkbox(
            "Filter Recursive Embedded",
            value=config_obj.filter_recursive_embedded,
            help="Enable recursive feature elimination"
        )

        if config_obj.filter_recursive_embedded:
            config_obj.recursive_embedded_number_features_to_select = st.number_input(
                "Number of Features to Select",
                value=config_obj.recursive_embedded_number_features_to_select,
                min_value=1,
                help="Number of features to select in recursive feature elimination"
            )

    with col2:
        config_obj.wrapper_sequential_feature_selection = st.checkbox(
            "Wrapper Sequential Feature Selection",
            value=config_obj.wrapper_sequential_feature_selection,
            help="Enable wrapper sequential feature selection"
        )

        if config_obj.wrapper_sequential_feature_selection:
            config_obj.sequential_direction = st.selectbox(
                "Sequential Direction",
                options=["forward", "backward"],
                index=0 if config_obj.sequential_direction == "forward" else 1,
                help="Direction for sequential feature selection"
            )

    config_obj.min_increase_4_wrapper = st.number_input(
        "Minimum Increase for Wrapper",
        value=config_obj.min_increase_4_wrapper,
        min_value=0.0,
        format="%.5f",
        help="Minimum score increase for a feature to be considered worthy in wrapper methods"
    )

    return config_obj


def render_model_config(config_obj):
    """Render the model configuration section."""
    st.subheader("Model Configuration")

    # Get a reference to the model tuning config
    model_config = config_obj.config_model_tuning

    col1, col2 = st.columns(2)
    with col1:
        model_config.hyperparameter_tuning_type = st.selectbox(
            "Hyperparameter Tuning Type",
            options=["OptunaTuner", "RandomizedSearchCV", "GridSearchCV"],
            index=["OptunaTuner", "RandomizedSearchCV", "GridSearchCV"].index(
                model_config.hyperparameter_tuning_type
            ),
            help="Type of hyperparameter tuning"
        )

        model_config.validation_score_mechanism = st.selectbox(
            "Validation Score Mechanism",
            options=["cv", "train_test_split"],
            index=0 if model_config.validation_score_mechanism == "cv" else 1,
            help="Validation score mechanism"
        )

        model_config.validation_score_splitting = st.selectbox(
            "Validation Score Splitting",
            options=["KFold", "TimeSeriesSplit"],
            index=0 if model_config.validation_score_splitting == "KFold" else 1,
            help="Validation score splitting method"
        )

    with col2:
        model_config.models = st.selectbox(
            "Model",
            options=["MLP", "RF", "SVR", "LR","ScikitMLP_TargetTransformed",'SciKerasSequential'],
            index=["MLP", "RF", "SVR", "LR", "ScikitMLP_TargetTransformed",'SciKerasSequential'].index(model_config.models),
            help="Model type"
        )

        model_config.validation_score_metric = st.selectbox(
            "Validation Score Metric",
            options=["r2", "mse", "rmse", "mae"],
            index=["r2", "mse", "rmse", "mae"].index(model_config.validation_score_metric),
            help="Validation score metric"
        )

        # For hyperparameter_tuning_kwargs, handling as a dictionary
        if model_config.hyperparameter_tuning_kwargs is None:
            model_config.hyperparameter_tuning_kwargs = {"n_trials": 5}

        n_trials = model_config.hyperparameter_tuning_kwargs.get("n_trials", 5)
        updated_n_trials = st.number_input(
            "Number of Trials",
            value=n_trials,
            min_value=1,
            help="Number of trials for hyperparameter tuning"
        )
        model_config.hyperparameter_tuning_kwargs = {"n_trials": updated_n_trials}

    # For validation_score_splitting_kwargs, handling as a dictionary
    if model_config.validation_score_splitting_kwargs is None:
        model_config.validation_score_splitting_kwargs = {"n_splits": 3}

    n_splits = model_config.validation_score_splitting_kwargs.get("n_splits", 3)
    model_config.validation_score_splitting_kwargs = {
        "n_splits": st.number_input(
            "Number of Splits",
            value=n_splits,
            min_value=2,
            help="Number of splits for cross-validation"
        )
    }

    return config_obj


def main():
    st.set_page_config(
        page_title="ADDMO Data Tuning Configuration",
        page_icon="🔧",
        layout="wide"
    )

    st.title("ADDMO Data Tuning Auto Configuration")
    st.markdown("""
    This application allows you to configure and run the automated data tuning process.

    1. Configure the parameters in the Configuration tab
    2. Review and edit the full JSON in the JSON View tab
    3. Run the data tuning process in the Execute tab
    """)

    # Define the path to the configuration file
    config_path = os.path.join('addmo', 's1_data_tuning_auto', 'config', 'data_tuning_auto_config.json')

    if not os.path.exists(config_path):
        config_path = 'data_tuning_auto_config.json'  # Fallback to local path

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Configuration", "JSON View", "Execute"])

    # Load configuration from JSON
    json_config = load_config_from_json(config_path)

    # Initialize config object using the DataTuningAutoSetup class
    if json_config:
        config_obj = DataTuningAutoSetup(**json_config)
    else:
        # If no JSON config found, create a default one
        config_obj = DataTuningAutoSetup()

    with tab1:
        st.header("Data Tuning Configuration")

        # Create a form for the configuration
        with st.form("config_form"):
            st.subheader("Data Tuning Configuration")
            config: DataTuningAutoSetup = pydantic_input("Config Setup", DataTuningAutoSetup)
            st.subheader("Model Configuration")

            model_input: ModelTunerConfig = pydantic_input("ModelTunerConfig", ModelTunerConfig)
            model_input["models"] = st.multiselect(
                "List of Models",
                options=["MLP", "RF", "SVR", "LR", "ScikitMLP_TargetTransformed", "SciKerasSequential"],

                help="Choose one or more models to use in tuning"
            )

            config["config_model_tuning"] = model_input
            submitted = st.form_submit_button("Run Data Tuning")

            if submitted:
                config = DataTuningAutoSetup(
                    **config,
                )
                # Save config to the expected JSON location
                config_path = os.path.join(
                    root_dir(), 'addmo', 's1_data_tuning_auto', 'config', 'data_tuning_auto_config.json'
                )
                os.makedirs(os.path.dirname(config_path), exist_ok=True)

                with open(config_path, 'w') as f:
                    f.write(config.model_dump_json(indent=4))

                st.success("✅ Configuration saved!")

    with tab2:
        st.header("JSON Configuration")

        # Display JSON with option to edit
        json_str = json.dumps(config_obj.dict(), indent=2)
        edited_json = st.text_area("Edit JSON directly", json_str, height=400)

        try:
            edited_config = json.loads(edited_json)
            if st.button("Update from JSON"):
                try:
                    # Validate the configuration using the Pydantic model
                    validated_config = DataTuningAutoSetup(**edited_config)

                    # Save the validated configuration
                    if save_config_to_json(validated_config.dict(), config_path):
                        st.success(f"Configuration updated from JSON and saved to {config_path}!")

                        # Update the config object
                        config_obj = validated_config
                except Exception as e:
                    st.error(f"Invalid configuration: {e}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

        # Download button
        st.download_button(
            label="Download Config JSON",
            data=json.dumps(config_obj.dict(), indent=2),
            file_name="data_tuning_auto_config.json",
            mime="application/json"
        )

    with tab3:
        st.header("Execute Data Tuning")
        st.write("Click the button below to run the data tuning process using the current configuration.")

        if st.button("Run Data Tuning"):
            run_data_tuning()


if __name__ == "__main__":
    main()