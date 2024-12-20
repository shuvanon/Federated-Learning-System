import os
import shutil
from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def copy_images_to_client_folders(client_data, img_dir, destination_dir):
    """
    Copies images from the original image directory to the destination client folder based on the client data.

    Args:
        client_data (Dict[int, pd.DataFrame]): Dictionary containing the client-specific data subsets.
        img_dir (str): Directory where the original images are stored.
        destination_dir (str): Directory where the images should be copied to.
    """
    for client_id, data in client_data.items():
        client_folder = os.path.join(destination_dir, f'client_{client_id}')
        os.makedirs(client_folder, exist_ok=True)

        for _, row in data.iterrows():
            img_name = str(row[0]) + ".png"
            img_source_path = os.path.join(img_dir, img_name)
            img_dest_path = os.path.join(client_folder, img_name)

            if os.path.exists(img_source_path):
                shutil.copy(img_source_path, img_dest_path)
            else:
                print(f"Image {img_source_path} not found!")


def transfer_to_benchmark(client_data: Dict[int, pd.DataFrame], benchmark_percentage: float, preprocessed_data_dir: str,
                          img_dir: str, csv_file: str, save_benchmark: bool) -> pd.DataFrame:
    """
    Transfers a percentage of each client's data to a single benchmark CSV and copies images to the specified directory.

    Args:
        client_data (Dict[int, pd.DataFrame]): Dictionary of client-specific data subsets.
        benchmark_percentage (float): Percentage of each client's data to transfer.
        preprocessed_data_dir (str): Directory where the processed client data is stored.
        img_dir (str): Directory where benchmark images should be saved.
        csv_file (str): Path to save the benchmark CSV file.
        save_benchmark (bool): Whether to save benchmark data to file or keep in memory.

    Returns:
        pd.DataFrame: Benchmark data (labels) if kept in memory.
    """
    # Initialize an empty DataFrame to store the consolidated benchmark data
    consolidated_benchmark_data = pd.DataFrame()

    # Ensure the img_dir exists
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for client_id, data in client_data.items():
        # Sample the benchmark data for the client
        benchmark_data = data.sample(frac=benchmark_percentage)

        # Append the benchmark data to the consolidated DataFrame
        consolidated_benchmark_data = pd.concat([consolidated_benchmark_data, benchmark_data], ignore_index=True)

        # Correct the image name column reference here (assume the column is 'filename')
        for _, row in benchmark_data.iterrows():
            img_name = str(row.iloc[0]) + ".png"
            # print(f"Processing image: {img_name}")

            img_source_path = os.path.join(f'{preprocessed_data_dir}/client_{client_id}', img_name)
            if os.path.exists(img_source_path):
                img_dest_path = os.path.join(img_dir, img_name)
                # print(f"Copying {img_source_path} to {img_dest_path}")
                shutil.copy(img_source_path, img_dest_path)
                if os.path.exists(img_dest_path):
                    # print(f"Successfully copied to {img_dest_path}")
                    continue
                else:
                    print(f"Failed to copy to {img_dest_path}")
            else:
                print(f"Image not found: {img_source_path}")

    # Save to file if save_benchmark is True
    if save_benchmark:
        consolidated_benchmark_data.to_csv(csv_file, index=False)
        print(f"Benchmark data (CSV) saved to {csv_file} and images saved to {img_dir}")
    else:
        print("Benchmark data is kept in memory.")

    return consolidated_benchmark_data


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


class PreprocessedDataset(Dataset):
    """
    Custom Dataset class to load preprocessed images and labels for a client.

    Args:
        data_dir (str): Directory where the preprocessed images and labels are stored.
        transform (torchvision.transforms.Compose): Transformations to apply to the images.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.labels_file = os.path.join(data_dir, "labels.csv")
        self.data_frame = pd.read_csv(self.labels_file)  # Assuming labels.csv has columns: 'filename', 'label'
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Fetch a sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            image (Tensor): The image at the given index.
            label (int): The label corresponding to the image.
        """
        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx, 0])  # Filename is in the first column
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1])  # Label is in the second column

        if self.transform:
            image = self.transform(image)

        return image, label
