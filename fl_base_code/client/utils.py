import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder

def load_config(config_file):
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, dtype={'filename_column': str})
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = self.create_label_mapping()

    def create_label_mapping(self):
        labels = self.data_frame.iloc[:, 3].unique()  
        label_mapping = {label: idx for idx, label in enumerate(labels)}
        return label_mapping

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx, 0]))  # Ensure the filename is a string
        img_name = img_name+'.png'
        image = Image.open(img_name).convert('RGB')  # Convert image to grayscale
        tag_str = self.data_frame.iloc[idx, 3]
        tag = torch.tensor(self.label_mapping[tag_str], dtype=torch.long)  # Convert string label to integer and then to a tensor
        if self.transform:
            image = self.transform(image)
        return image, tag

def load_data(csv_file, img_dir, batch_size=32, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(model, train_loader, device, epochs=1, learning_rate=0.001, client_id=100):
    """
    Trains the model on the given training data.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader containing the training data.
        device (torch.device): The device (CPU or GPU) to use for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    model.to(device)  # Ensure model is on the correct device
    model.train()  # Set the model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification tasks

    # Loop over the number of epochs
    for epoch in range(epochs):
        running_loss = 0.0  # Track loss per epoch
        for batch in train_loader:
            # Unpack and move data and labels to the device
            data, target = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()  # Zero out gradients before the backward pass
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss

        # Optional: Log the average loss per epoch
        # print(client_id)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


def test(model, test_loader, device):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data and target to GPU
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
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

    def __init__(self, data_dir: str, csv_file: str, label_column_index: int, transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.data_frame = pd.read_csv(csv_file)
        self.label_column_index = label_column_index

        # Set default transformations if none are provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to a standard size
            transforms.ToTensor(),       # Convert image to a tensor
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