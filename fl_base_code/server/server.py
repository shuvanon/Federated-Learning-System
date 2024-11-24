import os
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import start_server, ServerConfig
# from data_loader.utils import PreprocessedDataset
from utils import load_config, test, PreprocessedDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from model import Net
from typing import Any, Dict, Tuple


def load_benchmark_data(config: Dict[str, Any]) -> DataLoader:
    """
    Load the global benchmark dataset.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        DataLoader: DataLoader for the benchmark dataset.
    """
    print("Config: load_benchmark_data")
    print(config)
    benchmark_config = config["benchmark"]
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Fetch label column index from the config
    label_column_index = config["data"]["label_column_index"]

    # Initialize the dataset with the correct label column
    benchmark_dataset = PreprocessedDataset(
        data_dir=benchmark_config["img_dir"],
        csv_file=benchmark_config["csv_file"],
        label_column_index=label_column_index,
        transform=transform
    )

    benchmark_loader = DataLoader(benchmark_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    return benchmark_loader

def evaluate_global_model(parameters: fl.common.NDArrays, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the global model on the benchmark dataset.

    Args:
        parameters (fl.common.NDArrays): The global model parameters from the server.
        config (Dict[str, Any]): Configuration dictionary containing data paths and settings.

    Returns:
        Tuple[float, Dict[str, float]]: The evaluation loss and metrics (accuracy).
    """
    # Initialize the model and load parameters
    model = Net()
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=True)  # Load the global model parameters
    model.eval()

    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the benchmark data for evaluation
    print("Config: evaluate_global_model")
    print(config)
    benchmark_loader = load_benchmark_data(config)

    # Evaluate the model on the test set
    loss, accuracy = test(model, benchmark_loader, device)
    return loss, {"accuracy": accuracy}

def main():
    """
    Main function to start the federated learning server with server-side evaluation.
    """
    #config = load_config('config.yaml')

    config = load_config('config.yaml')
    print(config)  # Debug: Check if 'benchmark' is present
    if "benchmark" in config:
        print("Benchmark section is present")
    else:
        print("Benchmark section is missing")

    # Define the evaluation function for the server
    def server_evaluation_fn(server_round: int, parameters: fl.common.NDArrays, evaluation_config: Dict[str, Any]) -> \
    Tuple[float, Dict[str, float]]:
        print("config: server_evaluation_fn")
        print(config)
        return evaluate_global_model(parameters, config)

    # Define the strategy with server-side evaluation
    strategy = FedAvg(
        fraction_fit=1.0,  # 100% of available clients are used for training
        fraction_evaluate=0.0,  # No client-side evaluation
        min_fit_clients=2,  # Minimum number of clients to participate in training
        min_available_clients=2,  # Minimum number of clients that need to be connected
        evaluate_fn=server_evaluation_fn  # Server-side evaluation function
    )

    # Start Flower server
    fl.server.start_server(
        server_address=config["network"]["server"]["address"],
        config=ServerConfig(num_rounds=config["network"]["server"]["num_rounds"]),
        strategy=strategy
    )

if __name__ == "__main__":
    print("Server Start")
    current_working_dir = os.getcwd()
    print("Current working directory:", current_working_dir)
    main()
