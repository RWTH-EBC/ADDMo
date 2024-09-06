import os
import shutil


def collect_plots(base_dir, plot_name):
    # Infer the output directory
    output_dir = os.path.join(os.path.dirname(base_dir), f"{os.path.basename(base_dir)}_plots", plot_name)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all subdirectories in the base directory
    for run_folder in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_folder)

        # Check if it's a directory
        if os.path.isdir(run_path):
            plot_path = os.path.join(run_path, 'plots', plot_name)

            # Check if the plot exists
            if os.path.exists(plot_path):
                # Create the new filename
                new_filename = f"{run_folder}{os.path.splitext(plot_name)[1]}"

                # Copy the file to the output directory with the new name
                shutil.copy2(plot_path, os.path.join(output_dir, new_filename))
                print(f"Copied {plot_name} from {run_folder} to {output_dir}")
            else:
                print(f"Plot {plot_name} not found in {run_folder}")

    return output_dir


# Example usage
base_directory = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDezSelect_SVR\poly_sweep"
plot_to_collect = "carpets_system.png"

output_directory = collect_plots(base_directory, plot_to_collect)
print(f"Plots collected in: {output_directory}")