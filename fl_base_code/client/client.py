import json
import os
# from ..data_loader.utils import PreprocessedDataset
from typing import Any, Dict, Tuple

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split

from model import Net
from model_creator import build_model_from_config
from utils import PreprocessedDataset, load_config, train


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
        self.model = build_model_from_config(config)
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
        # client_id = config.get("client_id", "unknown_client")
        round_number = config.get("evaluation_round", 0)
        print(f"Round Number: {round_number} -> Client {self.client_id}: Training for {self.epochs} epochs")

        try:
            # Load model parameters
            self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

            # Train the model
            train(self.model, self.train_loader, round_number=round_number, epochs=self.epochs,
                  learning_rate=self.learning_rate,
                  client_id=self.client_id,
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
        all_labels = []
        all_predictions = []
        total_loss = 0.0

        # Switch model to evaluation mode
        self.model.eval()

        # Define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Perform evaluation
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = criterion(output, label)
                total_loss += loss.item()
                _, predictions = torch.max(output, dim=1)

                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        # Calculate metrics

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        # Calculate the average loss
        average_loss = total_loss / len(self.test_loader)

        # Convert numpy types to Python-native types
        all_labels = [int(label) for label in all_labels]
        all_predictions = [int(pred) for pred in all_predictions]

        # Retrieve configuration values
        round_number = config.get("evaluation_round", 0)
        experiment_dir = config.get("experiment_dir", "experiment_results")
        print(f"Client {self.client_id}: Evaluating for round {round_number} in experiment directory {experiment_dir}")

        # Prepare results
        results = {
            "client_id": self.client_id,
            "round_number": round_number,
            "loss": average_loss,
            "num_samples": len(self.test_loader.dataset),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "labels": all_labels,
            "predictions": all_predictions
        }

        # Save results in the experiment directory
        client_results_dir = os.path.join(experiment_dir, "client_results")
        os.makedirs(client_results_dir, exist_ok=True)
        save_path = os.path.join(client_results_dir, f"client_{self.client_id}_round_{round_number}.json")
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}. Skipping save.")
            return total_loss / len(self.test_loader), len(self.test_loader.dataset), {}

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved for client {self.client_id} at round {round_number} to {save_path}")

        # print("Client Matrics for Client ID :", self.client_id)
        # print("average_loss: " + str(average_loss))
        # print("accuracy: " + str(accuracy))
        # print("precision: " + str(precision))
        # print("recall: " + str(recall))
        # print("f1: " + str(f1))
        # print("all_labels: " + str(all_labels))
        # print("all_predictions: " + str(all_predictions))

        # Return loss, number of samples, and metrics but this is not usable in server
        # return average_loss, len(self.test_loader.dataset), {
        #     "accuracy": accuracy,
        #     "precision": precision,
        #     "recall": recall,
        #     "f1_score": f1,
        #     "labels": ",".join(map(str, all_labels)),  # Convert list to comma-separated string
        #     "predictions": ",".join(map(str, all_predictions))  # Convert list to comma-separated string
        # }
        return average_loss, len(self.test_loader.dataset), {}

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
        client=client.to_client(),  # Convert the client to the appropriate type
        grpc_max_message_length=2_147_483_647
    )


if __name__ == "__main__":
    # print("Cleint Start")
    # current_working_dir = os.getcwd()
    # print("Current working directory:", current_working_dir)
    main()
