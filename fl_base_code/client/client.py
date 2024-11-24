import os
import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from model import Net
from utils import load_config, train, test, PreprocessedDataset
# from ..data_loader.utils import PreprocessedDataset
from typing import Tuple, Dict, Any

class FederatedClient(fl.client.NumPyClient):
    """
    Federated client class that handles local training and evaluation.

    Attributes:
        model (Net): The model to be trained and evaluated.
        device (torch.device): The device (CPU or GPU) used for training and evaluation.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(self, config: Dict[str, Any], client_id: int, data_loader: Tuple[DataLoader, DataLoader]):
        self.model = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.test_loader = data_loader
        self.epochs = config["training"]["epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.client_id = client_id

    def get_parameters(self, config: Dict[str, Any] = None) -> list:
        """Get the current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters: fl.common.NDArrays, config: Dict[str, Any]) -> Tuple[
        fl.common.NDArrays, int, Dict[str, float]]:
        """
        Train the model for this client.

        Args:
            parameters (fl.common.NDArrays): Model parameters from the server.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple[fl.common.NDArrays, int, Dict[str, float]]: Updated model weights, number of samples trained on, and training metrics.
        """
        #client_id = config.get("client_id", "unknown_client")
        print(f"Client {self.client_id}: Training for {self.epochs} epochs")

        try:
            # Load model parameters
            self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

            # Train the model
            train(self.model, self.train_loader, epochs=self.epochs, learning_rate=self.learning_rate, client_id=self.client_id,
                  device=self.device)

            # Extract updated model weights
            updated_weights = [v.cpu().numpy() for v in self.model.state_dict().values()]
            num_samples = len(self.train_loader.dataset)  # Ensure this is a positive integer

            # Dummy metric for illustration, replace with actual computation
            metrics = {"train_loss": 0.0}

            # Check returned values
            assert isinstance(updated_weights, list), "updated_weights should be a list of ndarrays."
            assert num_samples > 0, "num_samples should be a positive integer."

            return updated_weights, num_samples, metrics

        except Exception as e:
            print(f"An error occurred in fit function of client {self.client_id}: {e}")
            return [np.zeros_like(v) for v in parameters], 1, {"error": 1.0}  # Return safe defaults on error

        except Exception as e:
            print(f"An error occurred in fit function of client {self.client_id}: {e}")
            # Optionally, return defaults if an error occurs
            return [], 0, {}

    def evaluate(self, parameters: list, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on the local test data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        print(f"Client {client_id}: Test Loss: {loss}, Test Accuracy: {accuracy}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def set_parameters(self, parameters: list):
        """Set the model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)


def load_preprocessed_data(client_id: int, config: Dict[str, Any]) -> PreprocessedDataset:
    """
    Load preprocessed data for a given client.

    Args:
        client_id (int): The client ID.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        PreprocessedDataset: The preprocessed dataset for the client.
    """
    # Get the client's preprocessed data directory
    client_data_dir = os.path.join(config["data"]["preprocessed_data_dir"], f"client_{client_id}")

    # Get the CSV file path for the client's data
    client_csv_file = os.path.join(client_data_dir, "data.csv")

    # Fetch label column index from the config file
    label_column_index = config["data"]["label_column_index"]

    # Load the preprocessed dataset for the client
    dataset = PreprocessedDataset(
        data_dir=client_data_dir,
        csv_file=client_csv_file,
        label_column_index=label_column_index,
        transform=None  # Apply any necessary transformations if needed
    )

    return dataset

def split_dataset(dataset: PreprocessedDataset, train_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Split the dataset into training and testing sets.

    Args:
        dataset (PreprocessedDataset): The full dataset.
        train_split (float): Fraction of the dataset to use for training.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    # Determine the size for training and testing splits
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    """
    Main function to run the federated client.
    """
    config = load_config('config.yaml')

    # Read client-specific environment variables
    client_id = int(os.getenv("CLIENT_ID", "0"))  # Default to 0 if not set
    print(f"Client {client_id} is starting...")

    # Load the preprocessed data for this client
    dataset = load_preprocessed_data(client_id, config)

    # Split the dataset into training and testing sets
    train_loader, test_loader = split_dataset(
        dataset,
        train_split=config["data"]["train_split"],
        batch_size=config["data"]["batch_size"]
    )

    # Create the client instance
    client = FederatedClient(config, client_id, (train_loader, test_loader))

    # Start the federated client
    fl.client.start_client(
        server_address=config["network"]["client"]["address"],
        client=client.to_client()  # Convert the client to the appropriate type
    )

if __name__ == "__main__":
    print("Cleint Start")
    current_working_dir = os.getcwd()
    print("Current working directory:", current_working_dir)
    main()
