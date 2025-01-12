# Project Title

The Impact of Artificial Client Data Generation Parameters on the Performance of Federated Learning Systems

## Overview
This project investigates the impact of artificial client data generation parameters on the performance of Federated Learning (FL) systems. It involves the development of a modular FL pipeline with configurable data splitting and manipulation techniques, along with dynamic model selection capabilities. The system is built using the Flower framework and integrates custom CNN and Vision Transformer models.

## Architecture

**High-Level Workflow:**
- **Data Loading & Preprocessing:** Handles loading datasets and applying preprocessing techniques.
- **Data Splitting:** Implements various data splitting strategies: random, quantity skew, and feature-based splitting.
- **Data Manipulation:** Applies configurable image manipulation techniques (contrast, brightness, noise, etc.).
- **Model Training:** Federated learning setup using Flower framework with server-client architecture.
- **Evaluation:** Performance metrics (accuracy, precision, recall, F1-score) are logged and analysed.

**Main Components:**
- **`run_federated_learning.py`**: Orchestrates the server and clients.
- **`server/server.py`**: Configures and initiates the federated learning server.
- **`server/MetricSaver.py`**: Records and saves evaluation metrics.
- **`client/client.py`**: Defines the logic for client-side training and evaluation.
- **`data_loader/main.py`**: Loads and control splitting, manipulation and preprocesses data for FL, also usable as a standalone data generator.
- **`data_loader/data_splitter.py`**: Implements data splitting strategies.
- **`data_loader/data_manipulation.py`**: Applies data manipulation.
- **`data_loader/data_preprocessing.py`**: Applies data augmentation
- **`model_creator.py`**: Builds models dynamically (CNN/Transformer).

## Dependencies
```
torch~=2.3.1+cu121
pandas~=2.2.2
yaml~=0.2.5
pyyaml~=6.0.1
pillow~=10.2.0
scikit-learn~=1.5.2
torchvision~=0.18.1+cu121
flwr~=1.8.0
numpy~=1.26.4
timm~=1.0.11
opencv-python~=4.10.0.84
```

### Installation
```bash
# Clone the repository
git clone <https://gitlab.com/BenRachinger/fl_client_splitting_shuvanon.git>
cd <fl_client_splitting_shuvanon>

# Install required dependencies
pip install -r requirements.txt
```

## Usage Instructions

### Running the Federated Learning Pipeline
```bash
python run_federated_learning.py
```

### Configuration
Adjust experiment settings in `config.yaml`. Some settings are describe here:
- **Experiment Name:** Set a experiment name
- **Splitting Strategy:** `random`, `quantity_skew`, `feature_based`
- **Manipulation Technique:** `contrast`, `brightness`,`white_balance`, `gaussian_noise`, `sharpening_blurring`, or `none` if you don't want to manipulate the data.
- - **Manipulation Technique:** `fixed_flip`, `rotation` or `none` if you don't want to augment the data.
- **Model Type:** `custom_cnn`, `transformer`

### Logs and Results
Logs and metrics are saved in the `experiments/` directory.

## Project Structure
```
fl_client_splitting_shuvanon
├── fl_base_code
│   ├── client
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── model_creator.py
│   │   └── utils.py
│   ├── data_loader
│   │   ├── __init__.py
│   │   ├── data_manipulation.py
│   │   ├── data_preprocessing.py
│   │   ├── data_splitter.py
│   │   ├── main.py
│   │   └── utils.py
│   ├── experiments
│   │   ├── client_visualizations.ipynb
│   │   ├── FairnessAnalysis.ipynb
│   │   ├── MultiClient.ipynb
│   │   ├── multiple_experiment_analysis.ipynb
│   │   ├── splitcompare.ipynb
│   │   └── visualizations.ipynb
│   ├── server
│   │   ├── __init__.py
│   │   ├── MetricSaver.py
│   │   ├── model_creator.py
│   │   ├── server.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── config.yaml
│   └── run_federated_learning.py
│── data-cleaner-and_exploration
├   ├── data_cleaner.ipynb
│   └── data_exploration.ipynb
├── data
├── final_data
├── requirements.txt
└── README.md
```

## Contact
For any inquiries or contributions, please contact:

**Author:** [Shuvanon Razik]  
**Email:** [shuvanon.razik@fau.de]  
**Institution:** [Friedrich-Alexander-Universität Erlangen-Nürnberg]

