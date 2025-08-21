import base64
import streamlit as st
import json
import io, os, zipfile
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

def _zipdir_to_bytes(dir_path: str):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, start=dir_path)
                zf.write(abs_path, arcname=rel_path)
    buf.seek(0)
    return buf

def check_missing_fields(config_dict):
    """Return list of empty or missing config fields."""
    missing = []
    for key, value in config_dict.items():
        if isinstance(value, list) and not value:
            missing.append(key)
        elif value in [None, ""]:
            missing.append(key)
    return missing


def save_config_to_json(config_obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(config_obj.model_dump_json(indent=4))


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

    #### Feature Construction
    - **`create_differences`**: Create first-order differences (like feature derivatives).
    - **`create_manual_target_lag`**: Add lags for the target variable manually.
    - **`target_lag`**: Define which lags to add for the target.
    - **`create_manual_feature_lags`**: Add lags for specific input variables.
    - **`feature_lags`**: Define feature-specific lags.
    """)

    st.header("Guide for Data Tuning")
    st.markdown("""
    ### Manual Configuration
    - Enable `manual_feature_selection` and specify `selected_features`.
    - Enable `create_manual_feature_lags` and specify `feature_lags`.
    - Enable `create_manual_target_lag` and specify `target_lag`.

    ### Automated Feature Selection
    - **`recursive_embedded_number_features_to_select`**: For RFE selection.
    - **`wrapper_sequential_feature_selection`**: Forward/backward selection.
    - **`sequential_direction`**: `'forward'` or `'backward'`.
    - **`min_increase_for_wrapper`**: Threshold to accept feature.
    """)

    auto_tuning_config = pydantic_input("Auto", DataTuningAutoSetup)
    auto_tuning_config_obj = DataTuningAutoSetup(**auto_tuning_config)
    st.session_state.auto_tuning_config = auto_tuning_config_obj
    st.session_state.output_dir = None
    st.subheader("Automatic feature selection")
    st.markdown("""
        This step performs **recursive feature elimination (RFE)** using Random Forest.
    """)

    strategy = st.selectbox(
        "Select the feature selection strategy:",
        ["Select an option", "Minimum number of features", "Minimum score improvement"]
    )

    if strategy == "Minimum number of features":
        st.session_state.auto_tuning_config.filter_recursive_by_count = True
    elif strategy == "Minimum score improvement":
        st.session_state.auto_tuning_config.filter_recursive_by_score = True

    output_dir = results_dir_data_tuning(auto_tuning_config_obj)
    st.session_state.output_dir = output_dir
    st.subheader("Output Directory Strategy")
    st.write("The default directory for saving the tuned data is:")
    st.code(st.session_state.output_dir)

    dir_strategy = st.selectbox(
        "Choose how to handle existing results:",
        ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
    )
    if dir_strategy != "Select an option":
        overwrite_strategy = dir_strategy.split()[-1].strip("()")

    if st.button("Run Auto Data Tuning"):
        missing = check_missing_fields(auto_tuning_config)
        if missing:
            st.error(f"Missing fields: {', '.join(missing)}")
            return None

        config_path = os.path.join(
            root_dir(), 'addmo', 's1_data_tuning_auto', 'config', 'data_tuning_auto_config.json'
        )
        save_config_to_json(auto_tuning_config_obj, config_path)
        st.success("Configuration saved!")

        with st.spinner("Running data tuning..."):
            exe_data_tuning_auto(overwrite_strategy)
            base_path = os.path.join(st.session_state.output_dir, "tuned_xy_auto.pdf")
            pdf_viewer(base_path, width="80%", height=855)

            zoomed_path = os.path.join(st.session_state.output_dir, "tuned_xy_auto_2weeks.pdf")
            if os.path.exists(zoomed_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(zoomed_path, width="80%", height=855)
            st.success("Data tuning completed!")
            zip_bytes = _zipdir_to_bytes(st.session_state.output_dir)
            st.download_button(
                label="Download Data Tuning Auto results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(st.session_state.output_dir)}.zip",
                mime="application/zip"
            )

    return st.session_state.output_dir

def exe_streamlit_data_tuning_fixed():
    st.header("Data Tuning Fixed Configuration")
    st.markdown("""
        This setup allows you tune the system data in a fixed manner without randomness
    """)

    st.subheader("Data Tuning Configuration")
    st.markdown("""
        #### General Setup
        - Same fields as auto tuning.
        - Default folder: `addmo_examples\\results\\test_raw_data`
        - Default tuning name: `data_tuning_experiment_fixed`

        ### Feature Construction
        - `create_lag`, `create_diff`, `create_sqaured`

        ### Naming Convention
        - Example: `Temperature__lag3`, `Power__diff`
    """)

    fixed_tuning_config = pydantic_input("Config Setup", DataTuningFixedConfig)
    fixed_config_tuning_obj = DataTuningFixedConfig(**fixed_tuning_config)
    st.session_state.output_dir = None
    output_dir = results_dir_data_tuning(fixed_config_tuning_obj)
    st.session_state.output_dir = output_dir
    st.subheader("Output Directory Strategy")
    st.write("The default directory for saving the tuned data is:")
    st.code(st.session_state.output_dir)

    dir_strategy = st.selectbox(
        "Choose how to handle existing results:",
        ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
    )
    if dir_strategy != "Select an option":
        overwrite_strategy = dir_strategy.split()[-1].strip("()")

    if st.button("Run Fixed Data Tuning"):
        missing = check_missing_fields(fixed_tuning_config)
        if missing:
            st.error(f"Missing fields: {', '.join(missing)}")
            return

        config_path = os.path.join(
            root_dir(), 'addmo', 's2_data_tuning', 'config', 'data_tuning_config.json'
        )
        save_config_to_json(fixed_config_tuning_obj, config_path)
        st.success("Configuration saved!")

        with st.spinner("Running data tuning.."):
            exe_data_tuning_fixed(overwrite_strategy)
            base_path = os.path.join(st.session_state.output_dir, "tuned_xy_fixed.pdf")
            pdf_viewer(base_path, width="80%", height=855)

            zoomed_path = os.path.join(st.session_state.output_dir, "tuned_xy_fixed_2weeks.pdf")
            if os.path.exists(zoomed_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(zoomed_path, width="80%", height=855)
            st.success("Data tuning completed!")
            zip_bytes = _zipdir_to_bytes(st.session_state.output_dir)
            st.download_button(
                label="Download Data Tuning Fixed results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(st.session_state.output_dir)}.zip",
                mime="application/zip"
            )
    return st.session_state.output_dir

def exe_streamlit_model_tuning():
    st.header("Model Tuning")
    st.markdown("""
        This setup allows you to:
        - Select and train one or multiple machine learning models.
        - Tune their hyperparameters using Optuna or Grid Search.
        - Use cross-validation or holdout splits to validate performance.
    """)

    # Session initialization
    for key in ["model_config_data", "model_config_saved", "output_dir", "overwrite_strategy","model_tuner"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "model_config_saved" else False

    # Inputs
    st.subheader("Model Configuration")
    model_config_data = pydantic_input("ModelConfig", ModelTuningExperimentConfig)
    st.subheader("Model Tuning Configuration")
    model_tuner = pydantic_input("ModelTunerConfig", ModelTunerConfig)

    if not model_tuner.get("hyperparameter_tuning_kwargs"):
        model_tuner["hyperparameter_tuning_kwargs"] = {"n_trials": 2}

    model_config_data["_config_model_tuner"] = model_tuner

    st.subheader("Input data tuning type")
    type_of_data = st.selectbox("Would you like to use tuned data for model tuning?", ["choose", "Yes", "No"])
    st.session_state.use_tuned_data = type_of_data

    if type_of_data == "Yes":
        type_of_tuning = st.selectbox("Which tuning would you like to use?", ["choose", "Auto", "Fixed"])

        if type_of_tuning in ["Auto", "Fixed"]:
            default_text = f"The default directory used for loading the data is: addmo_examples/results/test_raw_data/data_tuning_experiment_{type_of_tuning.lower()}"
            st.text(default_text)
            path_type = st.selectbox(
                "Use default saving path for input data?",
                ["Select an option", "Yes", "No"]
            )

            if path_type == "Yes":
                func = results_dir_data_tuning_auto if type_of_tuning == "Auto" else results_dir_data_tuning_fixed
                model_config_data["abs_path_to_data"] = os.path.join(
                    func(),f"tuned_xy_{type_of_tuning.lower()}.csv")
                st.success("Tuned data path set in config.")

            elif path_type == "No":
                model_config_data["abs_path_to_data"] = st.text_input('path')
                st.success("Tuned data path set in config.")

    # Save config
    if st.button("Save Model Config"):
        st.session_state.model_config_data = model_config_data
        st.session_state.model_tuner = model_tuner
        st.session_state.model_config_saved = True
        model_config_obj = ModelTuningExperimentConfig(**model_config_data)
        model_tuner_obj = ModelTunerConfig(**model_tuner)

        output_dir = return_results_dir_model_tuning(
            model_config_obj.name_of_raw_data,
            model_config_obj.name_of_data_tuning_experiment,
            model_config_obj.name_of_model_tuning_experiment,
        )
        st.session_state.output_dir = output_dir

        config_path_exp = os.path.join(root_dir(), 'addmo', 's3_model_tuning', 'config', 'model_tuner_experiment_config.json')
        config_path_tuner = os.path.join(root_dir(), 'addmo', 's3_model_tuning', 'config', 'model_tuner_config.json')
        save_config_to_json(model_config_obj, config_path_exp)
        save_config_to_json(model_tuner_obj, config_path_tuner)
        st.success("Model configuration saved!")

    if st.session_state.model_config_saved:
        st.subheader("Output Directory Strategy")
        st.code(st.session_state.output_dir)
        strategy = st.selectbox(
            "Handle existing results in output directory:",
            ["Select an option", "Overwrite (y)", "Delete and recreate (d)"]
        )
        if strategy != "Select an option":
            st.session_state.overwrite_strategy = strategy.split()[-1].strip("()")

    if (st.session_state.model_config_saved and st.session_state.overwrite_strategy is not None
            and st.button("Run Model Tuning")):
        model_config_obj = ModelTuningExperimentConfig(**st.session_state.model_config_data)
        model_tuner_obj= ModelTunerConfig(**st.session_state.model_tuner)
        with st.spinner("Running model tuning..."):
            exe_model_tuning(st.session_state.overwrite_strategy, model_config_obj, model_tuner_obj)
            plot_image_path = os.path.join(st.session_state.output_dir, "model_fit_scatter.pdf")
            st.markdown("### Model Fit Plot")
            pdf_viewer(plot_image_path, width="80%", height=855)
            st.success("Model tuning completed!")
            zip_bytes = _zipdir_to_bytes(st.session_state.output_dir)
            st.download_button(
                label="Download Model Tuning results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(st.session_state.output_dir)}.zip",
                mime="application/zip"
            )
    return st.session_state.output_dir

def generate_external_insights(model_config, model_metadata_config):
    """
    Generates plots for a model not trained by the current app,
    using the provided config and model directory.
    """

    # Initialize Streamlit session state variables if not already set
    if "external_config_submitted" not in st.session_state:
        st.session_state.external_config_submitted = True
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = model_config.get("saving_dir")
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = os.path.join(st.session_state.model_dir, 'plots')
    if "plots_selections" not in st.session_state:
        st.session_state.plots_selections = []
    if "show_plots_form" not in st.session_state:
        st.session_state.show_plots_form = True
    if "requires_bounds_form" not in st.session_state:
        st.session_state.requires_bounds_form = ""
    if "plots_confirmed" not in st.session_state:
        st.session_state.plots_confirmed = False
    if "bounds_confirmed" not in st.session_state:
        st.session_state.bounds_confirmed = False

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
                for plot in ['Predictions surface plot', 'Prediction surface with feature interaction scatter plot']
            )
            st.session_state.bounds_confirmed = False
            st.session_state.bounds_choice = "Select an option"
            st.session_state.bounds_choice_submitted = False

    if st.session_state.plots_confirmed:
        st.markdown(f"Using saved directory: {st.session_state.model_dir}")
        st.markdown(f"Plots will be saved in: {st.session_state.output_dir}")

        if st.session_state.requires_bounds_form:
            with st.form("Choose bounds form"):
                bounds_choice = st.selectbox(
                    "Choose bounds for the columns of data",
                    ["Select an option", "Choose min and max of existing data as bounds", "Define custom bounds"],
                    index=["Select an option", "Choose min and max of existing data as bounds",
                           "Define custom bounds"].index(
                        st.session_state.get("bounds_choice", "Select an option")
                    ),
                    key="bounds_choice_selectbox"
                )
                bounds_submitted = st.form_submit_button("Confirm bounds choice")
                if bounds_submitted:
                    st.session_state.bounds_choice = bounds_choice
                    st.session_state.bounds_choice_submitted = True
                    st.rerun()

        if st.session_state.get("bounds_choice_submitted", False):
            if st.session_state.bounds_choice == "Define custom bounds":
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
                        st.markdown(f"*{column}*")
                        min_val = st.number_input(f"Min for {column}", key=f"{column}_min", value=default_min)
                        max_val = st.number_input(f"Max for {column}", key=f"{column}_max", value=default_max)
                        default_input = st.number_input(f"Default value for {column}", key=f"{column}_default",
                                                        value=default_val)
                        st.session_state.custom_bounds[column] = [min_val, max_val]
                        st.session_state.defaults[column] = default_input
                    if st.form_submit_button("Confirm bounds and defaults"):
                        st.session_state.bounds = st.session_state.custom_bounds
                        st.session_state.final_defaults = st.session_state.defaults
                        st.session_state.bounds_confirmed = True
                        st.success("Custom bounds and default values have been saved.")
            elif st.session_state.requires_bounds_form and st.session_state.bounds_choice == "Choose min and max of existing data as bounds":
                st.session_state.bounds = None
                st.session_state.defaults = None
                st.session_state.bounds_confirmed = True
                if not model_config.get("abs_path_to_data"):
                    st.warning("Absolute path of data is required in order to create bounds and default values.")

        if 'Predictions surface plot' in st.session_state.plots_selections and st.session_state.bounds_confirmed:
            exe_carpet_plots(st.session_state.model_dir, "predictions_surface", st.session_state.output_dir, save=True,
                             bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
            st.markdown("### Carpet Plot")
            pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_surface.pdf"), width="80%")

        if 'Prediction surface with feature interaction scatter plot' in st.session_state.plots_selections and st.session_state.bounds_confirmed:
            exe_scatter_carpet_plots(st.session_state.model_dir, "predictions_scatter_surface",
                                     st.session_state.output_dir, save=True, bounds=st.session_state.bounds,
                                     defaults_dict=st.session_state.defaults)
            st.markdown("### Surface with scatter Plot")
            pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_scatter_surface.pdf"), width="80%")

        if 'Time Series plot' in st.session_state.plots_selections:
            exe_time_series_plot(st.session_state.model_dir, "training_data_time_series", st.session_state.output_dir,
                                 save=True)
            st.markdown("### Time Series Data Plot")
            base_path = os.path.join(st.session_state.output_dir, "training_data_time_series.pdf")
            pdf_viewer(base_path, width="80%", height=855)
            two_weeks_path = os.path.join(st.session_state.output_dir, "training_data_time_series_2weeks.pdf")
            if os.path.exists(two_weeks_path):
                st.markdown("### Zoomed View: Time Series (2 Weeks)")
                pdf_viewer(two_weeks_path, width="80%", height=855)

        if 'Predictions parallel plot' in st.session_state.plots_selections:
            fig = exe_interactive_parallel_plot(st.session_state.model_dir, "parallel_plot",
                                                st.session_state.output_dir, save=True)
            scrollable_html = f"""
            <div style="overflow-x: auto; width: 100%;">
                <div style="min-width: 1200px;">
                    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
            </div>
            """
            st.markdown("### Interactive Parallel Plot")
            st.components.v1.html(scrollable_html, height=700)

    return st.session_state.output_dir


def generate_addmo_insights():
    if "dir_submitted" not in st.session_state:
        st.session_state.dir_submitted = False
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = None
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
            if submitted and directory:
                st.session_state.model_dir = directory
                st.session_state.dir_submitted = True

    if st.session_state.dir_submitted and st.session_state.model_dir is not None and st.session_state.model_dir!="":
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

        if "plots_selections" not in st.session_state:
            st.session_state.plots_selections = []
        if "show_plots_form" not in st.session_state:
            st.session_state.show_plots_form = True
        if "requires_bounds_form" not in st.session_state:
            st.session_state.requires_bounds_form = ""
        if "plots_confirmed" not in st.session_state:
            st.session_state.plots_confirmed = False
        if "bounds_confirmed" not in st.session_state:
            st.session_state.bounds_confirmed = False

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
                    for plot in ['Predictions surface plot', 'Prediction surface with feature interaction scatter plot']
                )
                st.session_state.bounds_confirmed = False
                st.session_state.bounds_choice = "Select an option"
                st.session_state.bounds_choice_submitted = False

        if st.session_state.plots_confirmed:
            st.markdown(f"Using saved directory: {directory}")
            st.markdown(f"Plots will be saved in: {st.session_state.output_dir}")

            if st.session_state.requires_bounds_form:
                with st.form("Choose bounds form"):
                    bounds_choice = st.selectbox(
                        "Choose bounds for the columns of data",
                        ["Select an option", "Choose min and max of existing data as bounds", "Define custom bounds"],
                        index=["Select an option", "Choose min and max of existing data as bounds",
                               "Define custom bounds"].index(
                            st.session_state.get("bounds_choice", "Select an option")
                        ),
                        key="bounds_choice_selectbox"
                    )
                    bounds_submitted = st.form_submit_button("Confirm bounds choice")
                    if bounds_submitted:
                        st.session_state.bounds_choice = bounds_choice
                        st.session_state.bounds_choice_submitted = True
                        st.rerun()

            if st.session_state.get("bounds_choice_submitted", False):
                if st.session_state.bounds_choice == "Define custom bounds":
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
                            st.markdown(f"*{column}*")
                            min_val = st.number_input(f"Min for {column}", key=f"{column}_min", value=default_min)
                            max_val = st.number_input(f"Max for {column}", key=f"{column}_max", value=default_max)
                            default_input = st.number_input(f"Default value for {column}", key=f"{column}_default", value=default_val)
                            st.session_state.custom_bounds[column] = [min_val, max_val]
                            st.session_state.defaults[column] = default_input
                        if st.form_submit_button("Confirm bounds and defaults"):
                            st.session_state.bounds = st.session_state.custom_bounds
                            st.session_state.final_defaults = st.session_state.defaults
                            st.session_state.bounds_confirmed = True
                            st.success("Custom bounds and default values have been saved.")
                elif st.session_state.requires_bounds_form and st.session_state.bounds_choice == "Choose min and max of existing data as bounds":
                    st.session_state.bounds = None
                    st.session_state.defaults = None
                    st.session_state.bounds_confirmed= True
                    if not model_config.get("abs_path_to_data"):
                        st.warning("Absolute path of data is required in order to create bounds and default values.")

            if 'Predictions surface plot' in st.session_state.plots_selections and st.session_state.bounds_confirmed:
                exe_carpet_plots(st.session_state.model_dir, "predictions_surface", st.session_state.output_dir, save=True, bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
                st.markdown("### Carpet Plot")
                pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_surface.pdf"), width="80%")

            if 'Prediction surface with feature interaction scatter plot' in st.session_state.plots_selections and st.session_state.bounds_confirmed:
                exe_scatter_carpet_plots(st.session_state.model_dir, "predictions_scatter_surface", st.session_state.output_dir, save=True, bounds=st.session_state.bounds, defaults_dict=st.session_state.defaults)
                st.markdown("### Surface with scatter Plot")
                pdf_viewer(os.path.join(st.session_state.output_dir, "predictions_scatter_surface.pdf"), width="80%")

            if 'Time Series plot' in st.session_state.plots_selections:
                exe_time_series_plot(st.session_state.model_dir, "training_data_time_series", st.session_state.output_dir, save=True)
                st.markdown("### Time Series Data Plot")
                base_path = os.path.join(st.session_state.output_dir, "training_data_time_series.pdf")
                pdf_viewer(base_path, width="80%", height=855)
                two_weeks_path = os.path.join(st.session_state.output_dir, "training_data_time_series_2weeks.pdf")
                if os.path.exists(two_weeks_path):
                    st.markdown("### Zoomed View: Time Series (2 Weeks)")
                    pdf_viewer(two_weeks_path, width="80%", height=855)

            if 'Predictions parallel plot' in st.session_state.plots_selections:
                fig = exe_interactive_parallel_plot(st.session_state.model_dir, "parallel_plot", st.session_state.output_dir, save=True)
                scrollable_html = f"""
                <div style="overflow-x: auto; width: 100%;">
                    <div style="min-width: 1200px;">
                        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                """
                st.markdown("### Interactive Parallel Plot")
                st.components.v1.html(scrollable_html, height=700)

    return st.session_state.output_dir

def input_custom_model_config():
    st.subheader("Enter Configuration for External Model Insights")
    st.markdown("Fill in the required details to generate insights from a model *not trained by this application*.")
    st.markdown("""
    *Note*:
    - You need to specify the data path for creating carpet plots if the bounds and default values for each feature are not known and need to be calculated automatically.
    - You need to provide the input data for the creation of other plots (e.g., Time Series Plot and Parallel Plot).
    """)
    with st.form("external_model_config_form"):
        abs_path_to_data = st.text_input("Absolute path to the data file (e.g., .xlsx) (Not required for carpet plots)", key="abs_path_to_data")
        name_of_target = st.text_input("Name of the target variable", key="name_of_target")
        start_train_val = st.text_input("Start date/time for train/validation (e.g., 2016-08-01 00:00) (Not required for carpet plots)", key="start_train_val")
        stop_train_val = st.text_input("Stop date/time for test (e.g., 2016-08-14 23:45) (Not required for carpet plots)", key="end_test")
        features_ordered_input = st.text_input("Enter the list of ordered columns in the data, except the target column", key="features_ordered")
        base_class = st.selectbox("Base class of regressor", options=["ScikitMLP", "Keras"])
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

        base_name = os.path.splitext(model_file.name)[0]
        metadata_config_path = os.path.join(saving_dir, base_name + "_metadata.json")
        with open(metadata_config_path, "w") as f:
            json.dump(model_metadata_config, f, indent=4)
        data_config_path = os.path.join(saving_dir, "config.json")
        with open(data_config_path, "w") as f:
            json.dump(data_config, f, indent=4)
        model_path = os.path.join(saving_dir, model_file.name)
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        st.success("Configuration saved successfully!")
        return data_config, model_metadata_config
    return None

def exe_streamlit_data_insights():
    st.header('Generate Insights for the tuned saved models')
    st.markdown("""
    This setup allows you to *generate insightful visualizations* based on the results of previously trained and saved models.
    """)
    st.session_state.path = None
    if "choose_plotting_type" not in st.session_state:
        st.session_state.choose_plotting_type = "Select an option"
    choose_plotting_type = st.selectbox(
        "Would you like to generate insights for a model trained by this application or generate insights for other models?",
        ["Select an option", "Generate insights for a model trained by this application", "Generate insights for other models"],
        key="choose_plotting_type"
    )
    if st.session_state.choose_plotting_type == "Generate insights for a model trained by this application":
        st.session_state.path = generate_addmo_insights()
        zip_bytes = _zipdir_to_bytes(st.session_state.output_dir)
        st.download_button(
            label="Download Insights results folder",
            data=zip_bytes,
            file_name=f"{os.path.basename(st.session_state.output_dir)}.zip",
            mime="application/zip"
        )
    elif st.session_state.choose_plotting_type == "Generate insights for other models":
        if "external_config_submitted" not in st.session_state:
            result = input_custom_model_config()
            if result is not None:
                config, model_metadata_config = result
                st.session_state.config = config
                st.session_state.model_metadata_config = model_metadata_config
                # st.session_state.output_dir = config.get("saving_dir")
                st.session_state.external_config_submitted = True
                st.rerun()
        else:
            st.session_state.path = generate_external_insights(
                st.session_state.config,
                st.session_state.model_metadata_config
            )

            zip_bytes = _zipdir_to_bytes(st.session_state.output_dir)
            st.download_button(
                label="Download Insights results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(st.session_state.output_dir)}.zip",
                mime="application/zip"
            )
    return st.session_state.path

def exe_streamlit_model_testing():
    st.header("Model Testing")
    st.markdown("""
    This setup allows you to *test a previously trained and saved model* using new or unseen input data. This step helps you *evaluate model performance* beyond the training phase and validate generalization.

    Once a trained model is selected, this tab lets you:

    - Load the trained model's saved config
    - Provide new input data for testing
    - Reapply the correct *data tuning procedure* if required
    - Run model predictions and display a scatter plot for evaluation
    - Save and display the results

    ---

    #### Select Tuning Type
    This is *critical* if the model was trained on *tuned data*:
    - Note: Use raw data without any tuning (only if the model was trained that way).

    ⚠️ *Make sure the tuning type and input structure match the training phase!*

    The default results directory is:  
    addmo-automated-ml-regression/addmo_examples/results/model_streamlit_test/model_testing
    """)

    for key in ["dir_submitted", "input_submitted", "tuning_path_confirmed", "tuning_submitted"]:
        if key not in st.session_state:
            st.session_state[key] = False

    st.session_state.output_dir = results_model_streamlit_testing("model_testing")

    for key in [
        "model_dir", "input_data", "tuning_type", "output_dir",
        "custom_tuning_path", "tuning_path_type", "tuning_type_selected", "saving_dir"
    ]:
        if key not in st.session_state:
            st.session_state[key] = ""

    if not st.session_state.dir_submitted:
        with st.form("Model Directory"):
            option = st.radio(
                "Select directory option for loading previously saved model for testing:",
                ("Default", "Custom")
            )
            if option == "Custom":
                directory = st.text_input("Enter custom results directory (including experiment folder).")
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
                option = st.radio("Select raw input data path for testing the saved model:", ("Default", "Custom"))
                st.text("Default raw data path: addmo_examples/raw_input_data/InputData.xlsx")

                if option == "Default":
                    input_data_path = os.path.join(root_dir(), 'addmo_examples', 'raw_input_data', 'InputData.xlsx')
                else:
                    input_data_path = st.text_input("Enter custom input data path:")

                submitted = st.form_submit_button("Submit")

                if submitted:
                    st.session_state.input_data = input_data_path
                    st.session_state.input_submitted = True

        if st.session_state.input_submitted and not st.session_state.tuning_type_selected:
            with st.form("Tuning Type"):
                tuning_type = st.radio("Select type of tuning for raw input data:", ("Auto", "Fixed", "None"))
                submitted = st.form_submit_button("Submit")

            if submitted:
                st.session_state.tuning_type = tuning_type
                if tuning_type in ("Auto", "Fixed"):
                    st.session_state.tuning_type_selected = True
                else:
                    st.session_state.tuning_path_confirmed = True  # No tuning required

        if st.session_state.tuning_type_selected and not st.session_state.tuning_path_confirmed:
            with st.form("Tuning Path"):
                path_type = st.selectbox("Use default saving path for loading tuned data?",
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
                            "tuned_xy_auto.csv"
                        )
                        path_set = True
                    elif st.session_state.tuning_type == "Fixed":
                        abs_path = os.path.join(
                            results_dir_data_tuning_fixed(model_config["name_of_raw_data"]),
                            "tuned_xy_fixed.csv"
                        )
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
                        st.success("Tuned data path confirmed!")
                    else:
                        st.error("Please provide a valid path before confirming.")

        # Run model test when all inputs are gathered
        if st.session_state.tuning_submitted or st.session_state.tuning_path_confirmed:
            # cfg = st.session_state.get("model_config", model_config)
            # st.write('path of tuned data is: ', model_config["abs_path_to_data"])
            error, saving_dir = model_test(
                st.session_state.model_dir,
                model_config,
                st.session_state.input_data,
                "model_streamlit_test",
                st.session_state.tuning_type
            )

            st.session_state.saving_dir = saving_dir
            st.write("Results saved in:", saving_dir)
            st.write("Error is:", error)
            pdf_viewer(os.path.join(saving_dir, "model_fit_scatter.pdf"), width="80%")
            st.success("Model testing completed!")
            zip_bytes = _zipdir_to_bytes(saving_dir)
            st.download_button(
                label="Download Model Testing results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(saving_dir)}.zip",
                mime="application/zip"
            )

    return st.session_state.saving_dir

def exe_streamlit_data_tuning_recreate():
    st.header("Recreate data tuning for new data")
    st.markdown("""
    This setup allows you to *recreate the exact data tuning process* applied during a previous experiment, using the saved tuning configuration.  

    Although data tuning is *automatically handled* in the*Model Testing tab, this tab gives you *manual control* over the recreation of tuned datasets.  
    It's useful when:
    - You need a standalone tuned dataset for further analysis or visualization.
    - You want to verify how tuning was applied.
    - You want to reproduce preprocessing outside the model testing flow.

    > ⚠️ This tab *only supports tuning configurations saved by this app*.  
    > It cannot recreate tuning from externally trained models or configurations.
    ---""")

    if "tuning_type" not in st.session_state:
        st.session_state.tuning_type = None
        st.session_state.saving_dir = None

    if "tuning_submitted" not in st.session_state:
        st.session_state.tuning_submitted = False
        st.session_state.dir_submitted = False
        st.session_state.model_dir = None
    if "tuning_submitted" not in st.session_state:
        st.session_state.tuning_submitted = False
    if "dir_submitted" not in st.session_state:
        st.session_state.dir_submitted = False
    if "model_dir" not in st.session_state:
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
        with st.form("Data Directory"):
            option = st.radio(
                "Select directory option for loading previously saved config for tuned data:",
                ("Default", "Custom"))
            directory = st.text_input("Enter the custom results directory path. The path should contain the experiment folder as well.") if option == "Custom" else None
            submitted = st.form_submit_button("Submit")

        if submitted:
            if option == "Default":
                st.session_state.model_dir = results_dir_data_tuning_auto() if st.session_state.tuning_type == "Auto" else results_dir_data_tuning_fixed()
            else:
                st.session_state.model_dir = directory
            st.session_state.dir_submitted = True

    if st.session_state.dir_submitted:
        with st.form("Raw Input data"):
            input_data_path = st.text_input("Enter the raw input data path:")
            submitted = st.form_submit_button("Submit")
        if submitted:
            config_path = os.path.join(st.session_state.model_dir, "config.json")
            with open(config_path, 'r') as f:
                data_config = json.load(f)
                input_data_exp_name = data_config.get("name_of_raw_data")

            if st.session_state.tuning_type == "Auto":
                tuned_x_new, y_new, new_config, tuned_xy_new = data_tuning_recreate_auto(data_config, input_data_path, input_data_exp_name)
            else:
                tuned_x_new, tuned_y_new, new_config, tuned_xy_new = data_tuning_recreate_fixed(data_config, input_data_path, input_data_exp_name)

            st.write(tuned_x_new)
            result_dir = results_model_streamlit_testing(input_data_exp_name)
            tuned_xy_new.to_csv(os.path.join(result_dir, "tuned_data_recreated.csv"))
            st.write('The tuned data is saved at:', result_dir)
            st.session_state.saving_dir = result_dir
            zip_bytes = _zipdir_to_bytes(result_dir)
            st.download_button(
                label="Download Data Tuning results folder",
                data=zip_bytes,
                file_name=f"{os.path.basename(result_dir)}.zip",
                mime="application/zip"
            )

    return st.session_state.saving_dir


# Streamlit UI Setup

st.set_page_config(
    page_title="ADDMO",
    page_icon=os.path.join(root_dir(), 'staticfiles', '230718 Logo ADDMo-01.png'),
    layout="wide"
)

st.markdown("""
    <style>.block-container { padding-top: 1rem; }</style>
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
Welcome to *ADDMO, a AutoML toolkit* designed for *time series regression tasks*.

This application helps you configure, run, and analyze the full machine learning pipeline — from *data preprocessing* to *model tuning*, with full documentation and visualization at each step.

---

### Workflow Overview

1. *Data Tuning (Auto/Fixed)*  
2. *Model Tuning*  
3. *Data Insights*  
4. *Model Testing*  
5. *Data Tuning Recreate*  

Each module is modular and optional, enabling flexible experimentation and analysis.
""")

if 'last_saved_path' not in st.session_state:
    st.session_state.last_saved_path = ""

st.write("📁 Last Saved Path")
st.code(st.session_state.get("last_saved_path", "No path saved yet."))

tab = st.radio("Choose Tab", ["Data Tuning", "Model Tuning", "Insights", "Model Testing", "Data Tuning Recreate"], horizontal=True)

if tab == "Data Tuning":
    st.header("Choose Data Tuning type")

    if "tuning_type" not in st.session_state:
        st.session_state.tuning_type = None
    if "tuning_submitted" not in st.session_state:
        st.session_state.tuning_submitted = False

    if not st.session_state.tuning_submitted:
        with st.form("Type of tuning"):
            tuning_type = st.radio("Choose tuning type for dataset", ["Auto", "Fixed"], index=0, key="tuning_type_radio")
            submitted = st.form_submit_button("Confirm")
        if submitted:
            st.session_state.tuning_type = tuning_type
            st.session_state.tuning_submitted = True
            st.rerun()
    else:
        if st.session_state.tuning_type == "Auto":
            st.session_state.last_saved_path = exe_streamlit_data_tuning_auto()
        elif st.session_state.tuning_type == "Fixed":
            st.session_state.last_saved_path = exe_streamlit_data_tuning_fixed()

        st.markdown("---")
        if st.button("Run another tuning type"):
            for key in list(st.session_state.keys()):
                if key != "last_saved_path":
                    del st.session_state[key]
            st.rerun()

elif tab == "Model Tuning":
    st.session_state.last_saved_path = exe_streamlit_model_tuning()
    st.markdown("---")
    if st.button("Run another model tuning"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

elif tab == "Insights":
    st.session_state.last_saved_path = exe_streamlit_data_insights()
    st.markdown("---")
    if st.button("Generate other insights"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()

elif tab == "Model Testing":
    st.session_state.last_saved_path = exe_streamlit_model_testing()
    st.markdown("---")
    if st.button("Test another model"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()

elif tab == "Data Tuning Recreate":
    st.session_state.last_saved_path = exe_streamlit_data_tuning_recreate()
    st.markdown("---")
    if st.button("Generate another tuning type"):
        for key in list(st.session_state.keys()):
            if key != "last_saved_path":
                del st.session_state[key]
        st.rerun()