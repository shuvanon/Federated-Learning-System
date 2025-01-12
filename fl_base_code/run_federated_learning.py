import os
import subprocess
import sys
import time
from typing import Any, Dict

import yaml

# Set the working directory to the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load the configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_data_loader():
    """
    Run the data loader by executing main.py.
    """
    print("Running data loader (main.py)...")
    data_loader_command = ["python", "data_loader/main.py"]
    subprocess.run(data_loader_command, check=True)


def generate_client_paths(client_id: int) -> Dict[str, str]:
    """
    Dynamically generate data paths for each client.

    Args:
        client_id (int): ID of the client.

    Returns:
        Dict[str, str]: Dictionary with 'csv_file' and 'img_dir' for the client.
    """
    csv_file = f"client_{client_id}/data.csv"  # Example: client_0/data.csv
    img_dir = f"client_{client_id}_final"  # Example: client_0_final
    return {"csv_file": csv_file, "img_dir": img_dir}


def run_server(config: Dict[str, Any]):
    """
    Start the Flower server.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing the server details.
    """
    print("Starting the server...")
    server_command = ["python", "server/server.py"]
    # Start the server process
    server_process = subprocess.Popen(server_command)
    time.sleep(5)
    print(server_process)
    return server_process


def run_client(client_id: int, config: Dict[str, Any]):
    """
    Start a Flower client with the given client ID and dynamically generated paths.

    Args:
        client_id (int): The client ID to assign to the client process.
        config (Dict[str, Any]): Configuration dictionary containing the client details.
    """
    # Dynamically generate the client data paths
    client_data = generate_client_paths(client_id)


    # Pass the client ID and generated paths as environment variables
    client_command = ["python", "client/client.py"]

    # Start the client process and set environment variables for this client
    client_process = subprocess.Popen(client_command, env={
        **os.environ,  # Keep existing environment variables
        "CLIENT_ID": str(client_id),  # Set client ID
        "CLIENT_CSV_FILE": client_data["csv_file"],  # Set client-specific CSV path
        "CLIENT_IMG_DIR": client_data["img_dir"]  # Set client-specific image directory
    })

    return client_process


def main():
    """
    Main function to run the federated learning server and clients.
    """
    config_file = 'config.yaml'
    config = load_config(config_file)

    # Run the data loader to prepare data
    run_data_loader()

    # Start the server process
    server_process = run_server(config)
    time.sleep(15)

    # Get the number of clients from the configuration
    num_clients = config["num_clients"]

    # Start each client in parallel based on the number of clients specified in the config
    client_processes = []
    for client_id in range(num_clients):
        client_process = run_client(client_id, config)
        client_processes.append(client_process)
        time.sleep(5)

    # Wait for all clients to finish
    for client_process in client_processes:
        client_process.wait()

    # Once all clients have finished, stop the server
    # server_process.terminate()
    # server_process.wait()
    # torch.cuda.empty_cache()

    print("Server and all clients have finished execution.")


if __name__ == "__main__":
    current_working_dir = os.getcwd()
    print("Current working directory:", current_working_dir)
    main()
