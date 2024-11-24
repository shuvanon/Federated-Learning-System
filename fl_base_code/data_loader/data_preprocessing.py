import os
import pandas as pd
from PIL import Image
from typing import Any

class DataPreprocessing:
    """
    DataPreprocessing class for applying preprocessing techniques such as flipping, rotation, 
    and other augmentations to entire datasets.

    Args:
        None

    Returns:
        None
    """
    
    def __init__(self) -> None:
        pass

    def fixed_flipping(self, img: Image.Image) -> Image.Image:
        """
        Applies fixed horizontal flipping to the given image.

        Args:
            img (Image.Image): The image to be flipped.

        Returns:
            Image.Image: The horizontally flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def rotation(self, img: Image.Image, degrees: float) -> Image.Image:
        """
        Applies rotation to the given image.

        Args:
            img (Image.Image): The image to be rotated.
            degrees (float): Degrees to rotate the image.

        Returns:
            Image.Image: The rotated image.
        """
        return img.rotate(degrees)

    def bulk_process(self, data_frame: pd.DataFrame, img_dir: str, output_dir: str, 
                     process_type: str, process_params: Any = None) -> None:
        """
        Processes the entire dataset by applying a specified transformation and saves the results.

        Args:
            data_frame (pd.DataFrame): DataFrame containing image filenames and labels.
            img_dir (str): Directory containing the original images.
            output_dir (str): Directory to save the processed images.
            process_type (str): The type of processing to apply ('fixed_flip', 'rotation').
            process_params (Any): Additional parameters for the processing function (e.g., degrees for rotation).

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, str(row.iloc[0])+".png")
            img = Image.open(img_path)

            if process_type == 'fixed_flip':
                img = self.fixed_flipping(img)
            elif process_type == 'rotation':
                degrees = process_params.get('degrees', 0)
                img = self.rotation(img, degrees)

            save_path = os.path.join(output_dir, str(row.iloc[0])+".png")
            img.save(save_path)

    def process_in_memory(self, data_frame: pd.DataFrame, img_dir: str, process_type: str, process_params: Any = None) -> pd.DataFrame:
        """
        Processes and returns the augmented images in memory for each client's data.

        Args:
            data_frame (pd.DataFrame): DataFrame containing image filenames and labels.
            img_dir (str): Directory containing the original images.
            process_type (str): The type of processing to apply ('fixed_flip', 'rotation').
            process_params (Any): Additional parameters for the processing function (e.g., degrees for rotation).

        Returns:
            pd.DataFrame: DataFrame with augmented images in memory.
        """
        augmented_data = []

        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, row[0])
            img = Image.open(img_path)

            if process_type == 'fixed_flip':
                img = self.fixed_flipping(img)
            elif process_type == 'rotation':
                degrees = process_params.get('degrees', 0)
                img = self.rotation(img, degrees)

            augmented_data.append((img, row[1]))  # Append image and label

        return pd.DataFrame(augmented_data, columns=['image', 'label'])
