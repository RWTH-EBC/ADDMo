import os

import pandas as pd
import numpy as np
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from PIL import Image

from aixtra.util import loading_saving_aixtra

def select_train(xy: pd.DataFrame, plot_dir: str = None) -> pd.DataFrame:
    # Function to scale the data
    def scale_data(data):
        return (data - data.min()) / (data.max() - data.min())

    # Scale the entire dataset
    xy_scaled = scale_data(xy)

    # Start with the first day of data
    initial_day_scaled = xy_scaled.iloc[:96]  # Assuming 15-minute intervals, 96 points = 1 day
    dataset_scaled = initial_day_scaled.copy()
    remaining_data_scaled = xy_scaled.iloc[96:].copy()

    # Keep track of unscaled data indices
    dataset_indices = xy.index[:96]
    remaining_indices = xy.index[96:]

    columns = xy.columns
    column_combinations = [(columns[i], columns[j]) for i in range(len(columns)) for j in
                           range(i + 1, len(columns))]

    # Initialize KNN detector
    clf = KNN(contamination=0.1, n_neighbors=5)
    clf.fit(dataset_scaled)

    block_size = 4 * 2
    nr_blocks = 14*24*4 // block_size

    iteration = 1
    # Choose data for 63 more days
    while iteration <= nr_blocks:
        # Calculate novelty scores for remaining data
        novelty_scores = clf.decision_function(remaining_data_scaled)

        # Combine scores with index
        scored_data = list(zip(remaining_indices, novelty_scores))

        # Sort by novelty score in descending order
        scored_data.sort(key=lambda x: x[1], reverse=True)

        # Select the top block_size most novel data points
        top_novel_points = scored_data[:block_size]

        # Add the most novel points to the dataset
        new_indices = [idx for idx, _ in top_novel_points]
        dataset_indices = dataset_indices.append(pd.Index(new_indices))
        dataset_scaled = pd.concat([dataset_scaled, remaining_data_scaled.loc[new_indices]])

        # Remove the selected points from remaining_data
        remaining_indices = remaining_indices.drop(new_indices)
        remaining_data_scaled = remaining_data_scaled.drop(index=new_indices)

        # Train KNN detector on updated dataset
        clf = KNN(contamination=0.1, n_neighbors=5)
        clf.fit(dataset_scaled)

        print(f"Iteration {iteration}: Added {len(new_indices)} new data points")

        # Plot the data selection (using unscaled data for visualization)
        if plot_dir:
            for var1, var2 in column_combinations:
                combo_dir = os.path.join(plot_dir, f'{var1}_{var2}')
                os.makedirs(combo_dir, exist_ok=True)
                plot_data_selection(var1, var2, xy, xy.loc[dataset_indices], xy.loc[new_indices],
                                    iteration,
                                    save_path=os.path.join(combo_dir, f'iteration_{iteration}.png'))
        iteration += 1

    # Sort dataset by index and return unscaled data
    return dataset_indices


def plot_data_selection(var1, var2, dataset_all, dataset_selected=None, new_data=None,
                        iteration=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(dataset_all[var1], dataset_all[var2], color='#646567', alpha=0.5, label='All Data')

    if dataset_selected is not None:
        ax.scatter(dataset_selected[var1], dataset_selected[var2], color='#006165',
                   label='Selected Data')

    if new_data is not None:
        ax.scatter(new_data[var1], new_data[var2], color='#A11035', label='New Data')

    ax.set_ylabel(var2)
    ax.set_xlabel(var1)
    ax.legend()

    if iteration is not None:
        ax.set_title(f'Data Selection - Iteration {iteration}')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def create_gifs_for_all_combinations(base_folder):
    # Get all subdirectories (each representing a variable combination)
    combo_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    for combo_folder in combo_folders:
        combo_path = os.path.join(base_folder, combo_folder)
        create_gif_selection(combo_path, base_folder)

def create_gif_selection(image_folder, output_folder):
    # Define the output video name
    gif_name = f'{os.path.basename(image_folder)}_Knn_Auswahl.gif'
    output_gif_filename = os.path.join(output_folder, gif_name)

    # Get a list of all PNG images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sort images by iteration number
    images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Create the frames
    frames = []
    for i in images:
        img = Image.open(os.path.join(image_folder, i))

        # Convert to RGB mode (in case some images are in RGBA mode)
        img = img.convert('RGB')
        # Resize all images to the size of the first image
        if not frames:
            size = img.size
        else:
            img = img.resize(size, Image.LANCZOS)

        frames.append(img)

    # Save the png images into a GIF file that loops forever
    frames[0].save(output_gif_filename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100, loop=0)  # Changed loop to 0 for infinite looping

    print(f"GIF created: {output_gif_filename}")

# Example usage
if __name__ == "__main__":
    path_csvs = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\layer\8_bes_VLCOPcorr_random_absurd-sweep-5"

    xy_training = loading_saving_aixtra.read_csv(
        "xy_train", directory=path_csvs
    )
    xy_validation = loading_saving_aixtra.read_csv(
        "xy_val", directory=path_csvs
    )
    xy_test = loading_saving_aixtra.read_csv(
        "xy_test", directory=path_csvs
    )

    # append all of them
    xy = pd.concat([xy_training, xy_validation, xy_test])
    xy.index = xy.index.astype(int)

    # Call the select_train function
    plot_dir = r'D:\04_GitRepos\addmo-extra\aixtra_use_case\results'
    indices = select_train(xy, plot_dir=plot_dir)
    create_gifs_for_all_combinations(plot_dir)

    # save indices to a file
    indices_list = list(indices)
    indices_path = os.path.join(plot_dir, "selected_indices.txt")
    with open(indices_path, 'w') as f:
        for item in indices_list:
            f.write("%s, " % item)

    selected_dataset = xy.loc[indices].sort_index()

    # save the selected dataset
    data_path = os.path.join(plot_dir, "selected_dataset.csv")
    selected_dataset.to_csv(data_path, sep=";", index=True, header=True, encoding="utf-8")
