import glob
import json
import os

import numpy as np


class MetricSaver:
    def __init__(self, experiment_name: str, config: dict, save_dir: str = "experiments"):
        """
        Initialize the MetricSaver for an experiment.

        Args:
            experiment_name (str): Unique name of the experiment.
            config (dict): The configuration used for the experiment.
            save_dir (str): Directory to save experiment results.
        """
        self.experiment_name = experiment_name
        self.config = config  # Store the experiment configuration
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the experiment folder exists
        self.metrics = {
            "config": config,
            "server_results": [],  # Server_results on Benchmark data
            "client_results": {}  # Client results
        }

    @staticmethod
    def _convert_comma_string_to_list(comma_separated_str: str) -> list:
        """
        Convert a comma-separated string to a list of integers.

        Args:
            comma_separated_str (str): Comma-separated string of integers.

        Returns:
            list: List of integers.
        """
        if comma_separated_str:
            return list(map(int, comma_separated_str.split(",")))
        return []

    def add_round_metrics(self, round_num: int, server_metrics: dict):
        """
        Add performance metrics for a specific round.

        Args:
            round_num (int): Round number.
            server_metrics (dict): Server-side benchmark metrics.
        """

        # Add the metrics for the current round
        self.metrics["server_results"].append({
            "round": round_num,
            "server_metrics": server_metrics  # Store server-only metrics
        })

    def save_metrics(self):
        """
        Save the metrics and configuration to a JSON file.
        Converts all non-JSON serializable types (e.g., numpy types) to native Python types.
        """

        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj

        serializable_metrics = convert_to_serializable(self.metrics)

        # Sort `client_results` by round numbers before saving
        if "client_results" in serializable_metrics:
            serializable_metrics["client_results"] = {
                int(round_num): clients
                for round_num, clients in sorted(
                    serializable_metrics["client_results"].items(), key=lambda x: int(x[0])
                )
            }

        save_path = os.path.join(self.save_dir, f"{self.experiment_name}.json")
        with open(save_path, "w") as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"Metrics and configuration saved to {save_path}")

    # def save_metrics(self):
    #     """Save metrics to a JSON file."""
    #     save_path = os.path.join(self.save_dir, f"{self.experiment_name}.json")
    #     with open(save_path, "w") as f:
    #         json.dump(self.metrics, f, indent=4)
    #     print(f"Metrics saved to {save_path}")

    def merge_client_results(self, client_results_dir: str):
        """Merge client results into the main JSON."""
        # Ensure existing rounds are tracked as integers
        merged_rounds = {int(round_num): list(clients.keys()) for round_num, clients in
                         self.metrics["client_results"].items()}

        # Collect and merge client results
        client_results_files = glob.glob(f"{client_results_dir}/*.json")
        for file in client_results_files:
            with open(file, "r") as f:
                client_result = json.load(f)
                round_number = int(client_result["round_number"])  # Convert to int
                client_id = client_result["client_id"]

                # Check for redundancy
                if round_number in merged_rounds and client_id in merged_rounds[round_number]:
                    print(f"Skipping redundant result: Client {client_id}, Round {round_number}")
                    continue

                # Add result to the main JSON
                if round_number not in self.metrics["client_results"]:
                    self.metrics["client_results"][round_number] = {}
                self.metrics["client_results"][round_number][client_id] = client_result

        # Save updated metrics
        self.save_metrics()
