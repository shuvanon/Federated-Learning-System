import os
from datetime import datetime
from typing import Any, Dict, Tuple

import flwr as fl
import torch
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader
from torchvision import transforms

from MetricSaver import MetricSaver
from model_creator import build_model_from_config
from utils import PreprocessedDataset, load_config


def load_benchmark_data(config: Dict[str, Any]) -> DataLoader:
    """
    Load the global benchmark dataset.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        DataLoader: DataLoader for the benchmark dataset.
    """
    print("Config: load_benchmark_data")
    # print(config)
    benchmark_config = config["benchmark"]
    # make 224*224 if transformer otherwise 64*64
    if config["model"]["type"] == "transformer":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust for transformers
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif config["model"]["type"] == "custom_cnn":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Adjust for CNNs
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


def evaluate_global_model(parameters: fl.common.NDArrays, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates the global model on the benchmark dataset.

    Args:
        parameters (fl.common.NDArrays): The global model parameters from the server.
        config (Dict[str, Any]): Configuration dictionary containing data paths and settings.

    Returns:
        Tuple[float, Dict[str, Any]]: The evaluation loss and metrics (accuracy, precision, recall, F1-score).
    """
    # Initialize the model and load parameters
    model = build_model_from_config(config)
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=True)  # Load the global model parameters
    model.eval()

    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the benchmark data for evaluation
    benchmark_loader = load_benchmark_data(config)

    # Initialize variables to collect results
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        for data, labels in benchmark_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Predictions and labels
            _, predictions = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Compute final loss
    average_loss = total_loss / len(benchmark_loader)

    # Compute additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")

    # Return loss and metrics
    return average_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": all_labels,
        "predictions": all_predictions
    }


def main():
    """
    Main function to start the federated learning server with server-side evaluation.
    """
    config = load_config('config.yaml')
    num_clients = config["num_clients"]
    # print(config)  # Debug: Check if 'benchmark' is present
    # if "benchmark" in config:
    #     print("Benchmark section is present")
    # else:
    #     print("Benchmark section is missing")

    experiment_name = config.get("experiment_name") + "_" + config.get("use_split_strategy") + config.get(
        "use_manipulation_technique") + config.get("use_preprocessing_technique") + "_" + str(
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    experiment_dir = os.path.join("experiments", experiment_name)
    metric_saver = MetricSaver(experiment_name, config, "experiments")

    # Define the evaluation function for the server
    def server_evaluation_fn(server_round: int, parameters: fl.common.NDArrays, evaluation_config: Dict[str, Any]) -> \
            Tuple[float, Dict[str, Any]]:
        """
        Custom evaluation function for the server to aggregate client metrics, including labels and predictions.
        """
        print(f"Server Evaluation - Round {server_round}: Starting evaluation")

        # Perform global evaluation on the server
        loss, server_metrics = evaluate_global_model(parameters, config)
        print(loss, server_metrics)

        # Add metrics to the saver
        metric_saver.add_round_metrics(server_round, server_metrics)

        return loss, server_metrics

    def on_evaluate_config_fn(round_num: int) -> Dict[str, Any]:
        """
        Provide evaluation configuration for each round.
        """
        return {
            "evaluation_round": round_num,  # Pass the current round number
            "experiment_dir": experiment_dir  # Pass the experiment directory
        }

    # Define the strategy with server-side evaluation
    strategy = FedAvg(
        fraction_fit=1.0,  # 100% of available clients are used for training
        fraction_evaluate=1.0,  # No client-side evaluation
        min_fit_clients=num_clients,  # Minimum number of clients to participate in training
        min_available_clients=num_clients,  # Minimum number of clients that need to be connected
        evaluate_fn=server_evaluation_fn,  # Server-side evaluation function
        on_evaluate_config_fn=on_evaluate_config_fn
    )

    # strategy = FedProx(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_fit_clients=num_clients,
    #     min_available_clients=num_clients,
    #     evaluate_fn=server_evaluation_fn,
    #     on_evaluate_config_fn=on_evaluate_config_fn,
    #     proximal_mu=0.1
    # )

    # Start Flower server
    fl.server.start_server(
        server_address=config["network"]["server"]["address"],
        config=ServerConfig(num_rounds=config["network"]["server"]["num_rounds"]),
        grpc_max_message_length=2_147_483_647,
        strategy=strategy
    )

    # Save metrics and configuration after training completes
    metric_saver.save_metrics()

    # Merge client results into the main JSON
    metric_saver.merge_client_results(client_results_dir=os.path.join(experiment_dir, "client_results"))


if __name__ == "__main__":
    print("Server Start")
    # current_working_dir = os.getcwd()
    # print("Current working directory:", current_working_dir)
    main()
