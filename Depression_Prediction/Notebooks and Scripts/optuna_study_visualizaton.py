import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import os

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path
relative_path = "data/depression_study_ann.db"

# Construct absolute path using os.path.join
storage_path = os.path.join(base_dir, relative_path)

# Construct Optuna storage string
storage = f"sqlite:///{storage_path}"

# Load Optuna study

study = optuna.load_study(
    study_name="depression_study_ann",
    storage=storage
)

plot_optimization_history(study)