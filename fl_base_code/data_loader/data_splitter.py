import os
from typing import Dict

import numpy as np
import pandas as pd
from numpy.random import dirichlet


class DataSplitter:
    """
    DataSplitter class for splitting data among different clients based on various strategies.

    Args:
        data_frame (pd.DataFrame): DataFrame containing the dataset.
        splitting_config (dict): Configuration for data splitting strategies.

    Returns:
        Dict[int, pd.DataFrame]: Dictionary mapping client IDs to their respective data subsets.
    """

    def __init__(self, data_frame: pd.DataFrame, splitting_config: dict, num_clients: int) -> None:
        self.data_frame = data_frame
        self.splitting_config = splitting_config
        self.num_clients = num_clients

    def split_data(self) -> Dict[int, pd.DataFrame]:
        """
        Splits the dataset into subsets based on the specified strategy.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of client-specific data subsets.
        """
        strategy = self.splitting_config.get('strategy')
        # num_clients = self.splitting_config['num_clients']
        if strategy == 'quantity_skew':
            return self.quantity_skew_split(self.num_clients, self.splitting_config['alpha'])
        elif strategy == 'feature_based':
            return self.feature_based_split(self.num_clients, self.splitting_config['feature_column'])
        else:
            return self.random_split(self.num_clients)

    def random_split(self, num_clients: int) -> Dict[int, pd.DataFrame]:
        """
        Randomly splits the data into the specified number of clients.

        Args:
            num_clients (int): Number of clients.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of client-specific data subsets.
        """
        print("Random Split")
        client_data = np.array_split(self.data_frame.sample(frac=1, random_state=42), num_clients)
        return {i: client_data[i] for i in range(num_clients)}

    def quantity_skew_split(self, num_clients: int, alpha: float) -> Dict[int, pd.DataFrame]:
        """
        Splits the data among clients using a Dirichlet distribution for quantity skew.

        Args:
            num_clients (int): Number of clients.
            alpha (float): Dirichlet distribution parameter.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of client-specific data subsets.
        """
        print("quantity_skew_split")
        client_data = {i: [] for i in range(num_clients)}
        labels = self.data_frame['label'].unique()

        for label in labels:
            label_data = self.data_frame[self.data_frame['label'] == label]
            if len(label_data) < num_clients:
                print(f"Warning: Label {label} has fewer samples ({len(label_data)}) than clients ({num_clients}).")

            # Generate proportions using Dirichlet distribution
            proportions = dirichlet([alpha] * num_clients)
            proportions /= proportions.sum()  # Normalise proportions

            print(f"Proportions for label {label}: {proportions}")

            # Shuffle and split the data
            splits = (np.cumsum(proportions)[:-1] * len(label_data)).astype(int)
            client_splits = np.split(label_data.sample(frac=1, random_state=42), splits)

            # Assign splits to clients
            for i in range(num_clients):
                client_data[i].append(client_splits[i])

        # Concatenate data for each client
        for i in range(num_clients):
            client_data[i] = pd.concat(client_data[i], ignore_index=True)

        return client_data

    def feature_based_split(self, num_clients: int, feature_column: str) -> Dict[int, pd.DataFrame]:
        """
        Splits the data by assigning entire feature groups to specific clients.

        Args:
            feature_column (str): Column name for the feature to split on.
            num_clients (int): Number of clients.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of client-specific data subsets.
        """
        print("feature_based_split")
        unique_features = self.data_frame[feature_column].unique()
        client_data = {i: pd.DataFrame() for i in range(num_clients)}

        for i, feature in enumerate(unique_features):
            feature_group = self.data_frame[self.data_frame[feature_column] == feature]
            client_data[i % num_clients] = pd.concat([client_data[i % num_clients], feature_group], ignore_index=True)

        return client_data

    def save_split_data(self, client_data: Dict[int, pd.DataFrame], output_dir: str = 'client_data') -> None:
        """
        Saves the split data for each client to separate directories.

        Args:
            client_data (Dict[int, pd.DataFrame]): Dictionary of client-specific data subsets.
            output_dir (str): Directory to save the client data.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for client_id, data in client_data.items():
            client_dir = os.path.join(output_dir, f'client_{client_id}')
            if not os.path.exists(client_dir):
                os.makedirs(client_dir)
            data.to_csv(os.path.join(client_dir, 'data.csv'), index=False)
