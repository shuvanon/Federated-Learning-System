# Server utils.py

import os
from typing import Optional, Tuple

import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test(model, test_loader, device):
    """
    Evaluates the model on the given test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader containing the test data.
        device (torch.device): The device (CPU or GPU) to use for evaluation.

    Returns:
        Tuple[float, float]: Returns the average test loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data and labels to the device
            output = model(data)  # Forward pass
            test_loss += torch.nn.functional.cross_entropy(output, target).item()  # Compute the loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


class PreprocessedDataset(Dataset):
    """
    Custom dataset that loads images and their corresponding labels from a CSV file.
    The labels are string-based and are encoded into integers using LabelEncoder.

    Args:
        data_dir (str): Directory where the images are stored.
        csv_file (str): Path to the CSV file containing image file names and labels.
        label_column_index (int): The index of the label column in the CSV file.
        transform (callable, optional): Optional transformations to apply to the images.

    Attributes:
        data_dir (str): The directory where the images are located.
        data_frame (pd.DataFrame): DataFrame containing image names and labels.
        transform (callable): Image transformation functions.
        label_encoder (LabelEncoder): Encodes string labels to integers.
        label_column_index (int): Index of the label column in the CSV file.
    """

    def __init__(self, data_dir: str, csv_file: str, label_column_index: int,
                 transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.data_frame = pd.read_csv(csv_file)
        self.label_column_index = label_column_index

        # Set default transformations if none are provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to a standard size
            transforms.ToTensor(),  # Convert image to a tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize for better convergence
        ])

        # Initialize and fit the label encoder to convert string labels into integers
        self.label_encoder = LabelEncoder()
        self.data_frame.iloc[:, self.label_column_index] = self.label_encoder.fit_transform(
            self.data_frame.iloc[:, self.label_column_index]
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Fetches the image and the corresponding label at the given index.

        Args:
            idx (int): Index of the data to fetch.

        Returns:
            Tuple: A tuple (image, label) where image is the transformed image
                   and label is the integer-encoded label.
        """
        # Construct the full path for the image file
        img_name = os.path.join(self.data_dir, str(self.data_frame.iloc[idx, 0]) + ".png")
        image = Image.open(img_name).convert("RGB")  # Ensures 3 color channels (RGB)

        # Apply the image transformation
        if self.transform:
            image = self.transform(image)

        # Retrieve the encoded integer label
        label = self.data_frame.iloc[idx, self.label_column_index]  # Encoded integer label
        return image, label
