import os
import pandas as pd
from PIL import Image, ImageEnhance
from typing import Dict, Any, Tuple

class DataManipulation:
    """
    DataManipulation class for applying various image manipulation techniques.

    Args:
        manipulation_config (dict): Configuration for the image manipulation techniques.

    Returns:
        None
    """
    
    def __init__(self, manipulation_config: dict) -> None:
        self.manipulation_config = manipulation_config

    def bulk_process(self, client_id: int, img_dir: str, manipulation_config: dict, output_dir: str) -> None:
        """
        Processes and saves the manipulated images for each client's data.

        Args:
            client_id (int): ID of the client whose data is being processed.
            img_dir (str): Directory containing the original images.
            manipulation_config (dict): Configuration for the manipulation technique.
            output_dir (str): Directory to save the manipulated images.

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_frame = pd.read_csv(f'client_data/client_{client_id}/data.csv')
        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, str(row[0])+".png")
            # print("IMAGE PATH:  "+img_path)
            img = Image.open(img_path)

            if manipulation_config['type'] == 'enhanced_contrast':
                img = self.enhanced_contrast(img, manipulation_config['factor'])

            save_path = os.path.join(output_dir, str(row[0])+".png")
            # print("SAVE PATH:  "+save_path)
            img.save(save_path)

        # Save manipulated data
        manipulated_data_csv = os.path.join(output_dir, 'data.csv')
        data_frame.to_csv(manipulated_data_csv, index=False)  # Ensure the CSV is saved here
        print(f"Manipulated data saved at {manipulated_data_csv}")

    def process_in_memory(self, data_frame: pd.DataFrame, img_dir: str) -> pd.DataFrame:
        """
        Processes and returns the manipulated images in memory for each client's data.

        Args:
            data_frame (pd.DataFrame): DataFrame containing image filenames and labels.
            img_dir (str): Directory containing the original images.

        Returns:
            pd.DataFrame: DataFrame with manipulated images in memory.
        """
        manipulated_data = []

        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, row[0])
            img = Image.open(img_path)

            if self.manipulation_config['type'] == 'enhanced_contrast':
                img = self.enhanced_contrast(img, self.manipulation_config['factor'])

            manipulated_data.append((img, row[1]))  # Append image and label

        return pd.DataFrame(manipulated_data, columns=['image', 'label'])

    def enhanced_contrast(self, img: Image.Image, factor: float) -> Image.Image:
        """
        Enhances the contrast of the given image.

        Args:
            img (Image.Image): The image to be processed.
            factor (float): Factor by which to enhance the contrast.

        Returns:
            Image.Image: The image with enhanced contrast.
        """
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
