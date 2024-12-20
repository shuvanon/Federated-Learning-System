import os
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom Dataset class for loading images and their labels from a CSV file.

    Args:
        data_frame (pd.DataFrame): DataFrame containing image file names and corresponding labels.
        img_dir (str): Directory containing the image files.
        transform (Optional[transforms.Compose]): Transformations to apply to the images.

    Returns:
        Tuple[torch.Tensor, int]: Transformed image tensor and its label.
    """

    def __init__(self, data_frame: pd.DataFrame, img_dir: str, transform: Optional[any] = None) -> None:
        self.data_frame = data_frame
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Fetches the image and label at the specified index."""
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
