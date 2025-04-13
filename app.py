import streamlit as st
from streamlit_pydantic import pydantic_input
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo_examples.executables.exe_data_tuning_auto import exe_data_tuning_auto
from addmo_examples.executables.exe_data_tuning_fixed import exe_data_tuning_fixed
import json
import os
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo_examples.executables.exe_model_tuning import exe_model_tuning
from addmo.util.load_save_utils import root_dir
from s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from streamlit_pdf_viewer import pdf_viewer
from addmo.util.definitions import results_dir_data_tuning_auto,results_dir_model_tuning_fixed,return_results_dir_model_tuning
from addmo_examples.executables.exe_data_insights import exe_time_series_plot,exe_parallel_plot,exe_carpet_plots
from addmo.s4_model_testing.model_testing import model_test

def exe_streamlit_data_tuning_auto():
    st.header("Data Tuning Auto Configuration")

    # Show form with all config fields
    with st.form("config_form_auto"):
        st.subheader("Data Tuning Configuration")
        auto_tuning_config: DataTuningAutoSetup = pydantic_input( "Auto",DataTuningAutoSetup)
        st.subheader("Model Configuration")
        model_input: ModelTunerConfig = pydantic_input("model", ModelTunerConfig, group_optional_fields= "sidebar")
        auto_tuning_config["config_model_tuning"]=model_input
        overwrite_strategy = st.radio(
            "To overwrite the existing content type in 'addmo_examples/results/test_raw_data/data_tuning_experiment_auto' results directory, choose 'y', for deleting the current contents type choose 'd' ",
            ["y","d"],
            index=0,
            help="Select how to handle existing results directory."
        )
        submitted = st.form_submit_button("Run Auto Data Tuning")

    # Run when user submits the config
    if submitted:
        if auto_tuning_config is None :
            st.error("❌ Form is incomplete or invalid. Please fill out all required fields.")
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
            plot_image_path = os.path.join(results_dir_data_tuning_auto(auto_tuning_config), "tuned_xy_auto.pdf")

            st.markdown("### Auto-Tuned Data Plot")

            # Display the saved plot PDF
            pdf_viewer(plot_image_path, width= "80%", height= 855)

            st.success("✅ Data tuning completed!")

def exe_streamlit_data_tuning_fixed():
    st.header("Data Tuning Fixed Configuration")

    # Show form with all config fields
    with st.form("config_form_fixed"):
        st.subheader("Data Tuning Configuration")
        config_data = pydantic_input(key="Config Setup",model=DataTuningFixedConfig)
        default_features = [
            "Schedule",
            "Total active power",
            "FreshAir Temperature___diff",
            "FreshAir Temperature___lag1",
            "FreshAir Temperature___squared",
            "Total active power___lag1"
        ]
        features_input = st.text_area(
            "(comma separated)",
            value=", ".join(default_features),
            help="List of features which the tuning shall result in."
        )
        config_data['features'] = [f.strip() for f in features_input.split(",") if f.strip()]
        overwrite_strategy = st.radio(
            "To overwrite the existing content type in 'addmo_examples/results/test_raw_data/data_tuning_experiment_fixed' results directory, choose 'y', for deleting the current contents type choose 'd' ",
            ["y", "d"],
            index=0,
            help="Select how to handle existing results directory."
        )
        submitted = st.form_submit_button("Run Fixed Data Tuning")

    # Run when user submits the config
    if submitted:
        if config_data is None:
            st.error("❌ Form is incomplete or invalid. Please fill out all required fields.")
        config_data  = DataTuningFixedConfig(**config_data )
        # Save config to the expected JSON location
        config_path = os.path.join(
            root_dir(), 'addmo', 's2_data_tuning', 'config', 'data_tuning_config.json'
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            f.write(config_data.model_dump_json(indent=4))

        st.success("✅ Configuration saved!")

        # Run the tuning process
        with st.spinner("Running data tuning..."):
            exe_data_tuning_fixed(overwrite_strategy)
            plot_image_path = os.path.join(results_dir_model_tuning_fixed(config_data), "xy_tuned_fixed.pdf")

            st.markdown("### Tuned Fixed Data Plot")

            # Display the saved plot PDF
            pdf_viewer(plot_image_path, width="80%", height=855)

            st.success("✅ Data tuning completed!")

def exe_streamlit_model_tuning():
    st.header("Model Tuning")

    # Form 1: Model configuration
    with st.form("config_form_model"):
        st.subheader("Model Configuration")
        model_config_data = pydantic_input("ModelConfig", ModelTuningExperimentConfig)

        with st.sidebar:
            st.subheader("Model Tuning Configuration")
            model_tuner = pydantic_input("ModelTunerConfig", ModelTunerConfig)

        submitted_config = st.form_submit_button("Save Model Config")

        if submitted_config:
            model_config_data["config_model_tuner"] = model_tuner
            st.session_state.model_config_data = model_config_data
            st.success("Model configuration saved!")


    # Form 2: Ask about tuned data
    with st.form("form_tuned_data"):
        type_of_data = st.radio("Would you like to use tuned data for model tuning?", ["Yes", "No"])
        submitted_tuning_pref = st.form_submit_button("Submit")

        if submitted_tuning_pref:
            st.session_state.use_tuned_data = type_of_data

    # Form 3: If tuned data is selected, ask which tuning
    if st.session_state.get("use_tuned_data") == "Yes":
        with st.form("form_choose_tuning_type"):
            type_of_tuning = st.radio("Which tuning would you like to use?", ["Auto", "Fixed"])
            submitted_tuning_type = st.form_submit_button("Submit")
            if submitted_tuning_type:
                if "model_config_data" in st.session_state:
                    # model_config_data = st.session_state.model_config_data
                    if type_of_tuning == "Auto":
                        st.session_state.model_config_data["abs_path_to_data"] = os.path.join(results_dir_data_tuning_auto(),
                                                                             "tuned_xy_auto.csv")
                    else:
                        st.session_state.model_config_data["abs_path_to_data"] = os.path.join(results_dir_model_tuning_fixed(),
                                                                             "tuned_xy_fixed.csv")
                    st.success("✅ Tuned data path set in config.")

    # Form 4: Final submission
    with st.form("form_run_model_tuning"):
        overwrite_strategy = st.radio(
            "To overwrite the existing content type in 'addmo_examples/results/test_raw_data/test_data_tuning/test_model_tuning' results directory, choose 'y', for deleting the current contents type choose 'd' ",
            ["y", "d"],
            index=0,
            help="Select how to handle existing results directory."
        )
        final_submit = st.form_submit_button("Run Model Tuning")

        if final_submit:
            if "model_config_data" not in st.session_state:
                st.error("🚫 Model configuration is missing. Please fill out the model config first.")
            else:
                model_config_data = st.session_state.model_config_data
                # Save and execute
                config_path = os.path.join(
                    root_dir(), 'addmo', 's3_model_tuning', 'config', 'model_tuning_config.json'
                )
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                model_config_data_obj  = ModelTuningExperimentConfig(**model_config_data)
                with open(config_path, 'w') as f:
                    f.write(model_config_data_obj .model_dump_json(indent=4))

                st.success("✅ Configuration saved!")

                with st.spinner("Running model tuning..."):
                    exe_model_tuning(overwrite_strategy, model_config_data_obj)
                    plot_image_path = os.path.join(return_results_dir_model_tuning(), "model_fit_scatter.pdf")

                    st.markdown("### Model fit Plot")

                    # Display the saved plot PDF
                    pdf_viewer(plot_image_path, width="80%", height=855)
                    st.success("✅ Model tuning completed!")

def exe_streamlit_data_insights():
    if "dir_submitted" not in st.session_state:
        st.session_state.dir_submitted = False
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = ""

    if st.session_state.dir_submitted is False:
        with st.form("Model Directory"):
            option = st.radio(
                "Select results directory option for loading previously saved model:",
                ("Default", "Custom")
            )

            if option == "Custom":
                directory = st.text_input("Enter the custom results directory path")
            else:
                directory = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning',
                                                            'test_model_tuning_raw')

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

        with st.form("Choose plots for execution"):
            plots_selections = st.multiselect("Select the plots which you'd like to see",
                                              options=['Time Series plot for training data',
                                                       'Predictions carpet plot',
                                                       'Predictions parallel plot'])
            submitted = st.form_submit_button("Run")

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
                    pdf_viewer(path, width="100%")

                if 'Predictions parallel plot' in plots_selections:
                    exe_parallel_plot(model_config, "parallel_plot", plot_dir, save=True)
                    st.markdown("### Parallel Plot")

                    # Display the saved plot PDF
                    path = os.path.join(plot_dir, "parallel_plot.pdf")
                    pdf_viewer(path, width="100%", height=1400 )

# Streamlit UI

st.set_page_config(
    page_title="ADDMO",
    page_icon="🔧",
    layout="wide"
)

st.title("ADDMO")
st.markdown("""
This application allows you to configure and run the automated data and model tuning process.

1. Data Tuning (Auto/Fixed)
2. Model Tuning
3. Data Insights
4. Model Testing
""")
tab = st.radio("Choose Tab", ["Data Tuning", "Model Tuning", "Insights", "Model Testing"], horizontal=True)
if tab == "Data Tuning":
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
                directory = st.text_input("Enter the custom results directory path")
            else:
                directory = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning',
                                                            'test_model_tuning_raw')

            submitted = st.form_submit_button("Submit")

            if submitted:
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
            error, saving_dir = model_test(st.session_state.model_dir, model_config, st.session_state.input_data, "model_streamlit_test", st.session_state.tuning_type)
            st.write(f"Results saved in: ", saving_dir)
            st.write(f"Error is: ", error)
            pdf_viewer(os.path.join(saving_dir,"model_fit_scatter.pdf"), width="80%")



            # Button to test another model
            if st.button("Test another model"):
                for key in ["dir_submitted", "input_submitted", "tuning_submitted"]:
                    st.session_state[key] = False
                for key in ["model_dir", "input_data", "tuning_type"]:
                    st.session_state[key] = ""
                st.rerun()






