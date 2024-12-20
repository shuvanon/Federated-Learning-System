import json
import logging
import os
from datetime import datetime
from typing import List


class LoggingManager:
    """
    Handles centralized logging for federated learning experiments.
    """

    @staticmethod
    def initialize_logger(name: str, log_file: str) -> logging.Logger:
        """
        Initialize a logger to log events to the given log file.

        Args:
            name (str): Name of the logger.
            log_file (str): Path to the log file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger

    @staticmethod
    def log_event(log_file: str, event: dict) -> None:
        """
        Append an event to the centralized log file.

        Args:
            log_file (str): Path to the log file.
            event (dict): The event to log.
        """
        with open(log_file, "r") as f:
            logs = json.load(f)
        logs["events"].append(event)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

    @staticmethod
    def merge_logs(main_log_file: str, client_log_files: List[str]) -> None:
        """
        Merge client log files into the centralized log file.

        Args:
            main_log_file (str): Path to the centralized log file.
            client_log_files (List[str]): List of client log file paths.
        """
        with open(main_log_file, "r") as f:
            main_logs = json.load(f)

        for client_log in client_log_files:
            with open(client_log, "r") as f:
                client_logs = json.load(f)
            main_logs["events"].extend(client_logs["events"])

        with open(main_log_file, "w") as f:
            json.dump(main_logs, f, indent=4)

    @staticmethod
    def initialize_client_logger(log_file: str) -> logging.Logger:

        logger = logging.getLogger(log_file)
        logger.setLevel(logging.INFO)

        # Clear existing handlers to prevent duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set up a new file handler that overwrites the log file
        handler = logging.FileHandler(log_file, mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

        return logger

    @staticmethod
    def client_log_event(log_file: str, event_data: dict) -> None:
        """
        Log a structured event to the specified log file.

        Args:
            log_file (str): Path to the log file.
            event_data (dict): Event data to log as a JSON string.
        """
        logger = logging.getLogger(log_file)

        # Ensure the logger is initialized before logging
        if not logger.handlers:
            LoggingManager.initialize_logger(log_file, log_file)

        # Convert the event data to a JSON string for structured logging
        logger.info(json.dumps(event_data))

    @staticmethod
    def merge_and_save_logs(single_json, json_list, another_json, config):
        """
        Merges a single JSON, a list of JSON files, and another JSON file into a structure
        that simulates duplicate keys using a list of dictionaries.
        Saves the merged data with a timestamped name based on information in the config file.
        """
        merged_data = []

        def load_json_file(file_path):
            try:
                with open(file_path, 'r') as file:
                    return json.load(file)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return {}

        # Load single JSON
        single_data = load_json_file(single_json)
        if single_data:
            merged_data.append(single_data)

        # Load JSON list
        for json_file in json_list:
            file_data = load_json_file(json_file)
            if file_data:
                merged_data.append(file_data)

        # Load another JSON
        another_data = load_json_file(another_json)
        if another_data:
            merged_data.append(another_data)

        # Generate log file name
        run_name = config.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = os.path.join("logs", f"{run_name}_{timestamp}.json")

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Save merged data
        try:
            with open(log_file, 'w') as file:
                json.dump(merged_data, file, indent=4)
            print(f"Logs saved to {log_file}")
        except Exception as e:
            print(f"Error saving log file: {e}")

        return log_file
