import os
import shutil

import pandas as pd
import yaml

from data_manipulation import DataManipulation
from data_preprocessing import DataPreprocessing
from data_splitter import DataSplitter
from utils import copy_images_to_client_folders, transfer_to_benchmark


# Clean folders function
def clean_folders(client_dir: str, benchmark_dir: str) -> None:
    """
    Cleans the specified directories by deleting their contents and recreating them.

    Args:
        client_dir (str): Path to the client data directory.
        benchmark_dir (str): Path to the benchmark data directory.
    """
    # Clean client_data directory
    if os.path.exists(client_dir):
        shutil.rmtree(client_dir)
        print(f"Cleaned {client_dir}")
    os.makedirs(client_dir, exist_ok=True)

    # Clean benchmark_data directory
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
        print(f"Cleaned {benchmark_dir}")
    os.makedirs(benchmark_dir, exist_ok=True)


def move_data_to_final_folders(intermediate_dir: str, final_dir: str) -> None:
    """
    Moves data from intermediate folders to the final client folders.

    Args:
        intermediate_dir (str): Path to the intermediate client data directory.
        final_dir (str): Path to the final client data directory.
    """
    if os.path.exists(intermediate_dir):
        # Move contents from intermediate folder to final folder
        for item in os.listdir(intermediate_dir):
            s = os.path.join(intermediate_dir, item)
            d = os.path.join(final_dir, item)
            shutil.move(s, d)
        print(f"Moved data from {intermediate_dir} to {final_dir}")

    # Optionally, clean up the intermediate folder after moving
    shutil.rmtree(intermediate_dir)
    print(f"Cleaned up {intermediate_dir}")


def main() -> None:
    # Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract top-level choices
    split_strategy = config['use_split_strategy']
    manipulation_technique = config['use_manipulation_technique']
    preprocessing_technique = config['use_preprocessing_technique']
    num_clients = config['num_clients']

    # Data settings
    save_split = config['data']['save_split']
    preprocessed_data_dir = config['data']['preprocessed_data_dir']
    clean_data_before_run = config['data']['clean_data_before_run']  # Extract the cleaning option

    # Benchmark settings
    benchmark_csv_file = config['benchmark']['csv_file']  # Path to benchmark CSV file
    benchmark_img_dir = config['benchmark']['img_dir']  # Path to benchmark image directory
    benchmark_percentage = config['benchmark']['benchmark_percentage']  # Percentage for benchmark
    save_benchmark = config['benchmark']['save_benchmark']  # Whether to save benchmark data

    # Clean the client and benchmark folders if required
    benchmark_dir = 'benchmark_data'
    if clean_data_before_run:
        clean_folders(preprocessed_data_dir, benchmark_img_dir)  # Clean the preprocessed data directory

    # Step 2: Load the data
    data_frame = pd.read_csv(config['data']['csv_file'])
    img_dir = config['data']['img_dir']

    # Step 3: Split into clients and save to intermediate folders
    intermediate_dir = f'{preprocessed_data_dir}/intermediate'  # Intermediate folders inside preprocessed_data_dir
    split_config = config['splitting']['strategies'][split_strategy]
    splitter = DataSplitter(data_frame, split_config, num_clients=num_clients)
    client_data = splitter.split_data()

    # Save split data to intermediate folders
    if save_split:
        for client_id, client_df in client_data.items():
            intermediate_folder = f'{intermediate_dir}/client_{client_id}'
            os.makedirs(intermediate_folder, exist_ok=True)
            client_df.to_csv(os.path.join(intermediate_folder, 'data.csv'), index=False)

    # Step 4: Manipulation and Augmentation in Intermediate Folders
    if manipulation_technique != 'none' or preprocessing_technique != 'none':
        for client_id, client_df in client_data.items():
            intermediate_folder = f'{intermediate_dir}/client_{client_id}'

            # Apply manipulation if needed
            if manipulation_technique != 'none':
                manipulation_config = config['manipulation']
                # if manipulation_config['manipulation_mode'] == "random":
                #     random_mode = True
                # else:
                #     random_mode = False

                manipulator = DataManipulation(manipulation_technique, manipulation_config)
                manipulator.bulk_process(client_id, img_dir, intermediate_folder)

            # Apply preprocessing if needed
            if preprocessing_technique != 'none':
                preprocess = DataPreprocessing()
                preprocess_config = config['preprocessing']['techniques'][preprocessing_technique]
                preprocess.bulk_process(client_df, img_dir, intermediate_folder, preprocessing_technique,
                                        preprocess_config)

    # Step 5: Move final data from intermediate folders to the final folders in preprocessed_data_dir
    if manipulation_technique == 'none' and preprocessing_technique == 'none':
        print("No manipulation or preprocessing, copying original images to final client folders...")
        copy_images_to_client_folders(client_data, img_dir, preprocessed_data_dir)

    for client_id in client_data.keys():
        final_folder = f'{preprocessed_data_dir}/client_{client_id}'
        os.makedirs(final_folder, exist_ok=True)
        move_data_to_final_folders(f'{intermediate_dir}/client_{client_id}', final_folder)

    # Step 6: Transfer percentage of data to benchmark folder (consolidated)
    benchmark_data = transfer_to_benchmark(client_data, benchmark_percentage, preprocessed_data_dir, benchmark_img_dir,
                                           benchmark_csv_file, save_benchmark)

    if not save_benchmark:
        # Benchmark data is in memory, and you can use it directly here
        print("Benchmark data is kept in memory and ready for use.")

    # Step 7: Make a list of all the client data directories and the benchmark data directory
    if save_split:
        client_dirs = [f'{preprocessed_data_dir}/client_{client_id}' for client_id in client_data.keys()]
    else:
        client_dirs = list(client_data.keys())

    print("Client Data Directories:", client_dirs)
    print(f"Benchmark Data CSV: {benchmark_csv_file}")
    print(f"Benchmark Data Image Directory: {benchmark_img_dir}")


if __name__ == '__main__':
    print("Data Loader start")
    current_working_dir = os.getcwd()
    print("Current working directory:", current_working_dir)
    main()
