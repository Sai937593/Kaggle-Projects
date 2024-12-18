from linear_regression_training import (
    prepare_train_validate,
    scorer,
    df_train_not_scaled,
    target,
)
import optuna


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


X = df_train_not_scaled.drop(columns=target)
y = df_train_not_scaled.loc[:, target]
print(X.shape, y.shape)

corr = X.corrwith(y)
X_train, X_test, y_train, y_test = prepare_train_validate(
    X, y, corr, top_n_cols=-1, scale=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0
)

train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
val_dmatrix = xgb.DMatrix(data=X_val, label=y_val)

print(train_dmatrix.num_row(), train_dmatrix.num_col())
print(val_dmatrix.num_row(), val_dmatrix.num_col())
print(X_train.shape, X_val.shape)
print(X_train.shape, X_val.shape)


# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Custom evaluation function for RMSLE
def rmsle_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    score = rmsle(y_true, y_pred)
    return "rmsle", score


# Objective function for Optuna
def objective(trial):
    num_boost_round = trial.suggest_int("num_boost_round", low=100, high=2500, step=50)
    learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=1e-1, log=True)
    max_depth = trial.suggest_int("max_depth", low=2, high=12)
    reg_alpha = trial.suggest_float("reg_alpha", low=0.1, high=10.0)
    reg_lambda = trial.suggest_float("reg_lambda", low=0.1, high=10.0)
    subsample = trial.suggest_float("subsample", low=0.8, high=0.99)
    colsample_bytree = trial.suggest_float("colsample_bytree", low=0.4, high=0.9)
    gamma = trial.suggest_float("gamma", low=0.0, high=5.0)

    # Create the DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set parameters for XGBoost
    params = {
        "objective": "reg:squaredlogerror",  # Use squared log error objective
        "eval_metric": "rmsle",  # Track RMSLE
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "random_state": 42,
        "tree_method": "hist",  # Use 'hist' for faster training
        "device": "cuda",  # Use GPU
    }

    # Perform cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=5,
        early_stopping_rounds=10,
        as_pandas=True,
        seed=42,
        custom_metric=rmsle_eval,  # Use custom RMSLE evaluation function
    )

    # Return the best RMSLE from cross-validation
    return cv_results["test-rmsle-mean"].min()


# Prepare your data (assuming X and y are defined)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create or load the study
xgb_reg_study = optuna.create_study(
    storage="sqlite:///xgb_reg_study.db",
    direction="minimize",  # Minimize the RMSLE
    load_if_exists=True,
    study_name="xgb_reg_study",
)

# Optimize the study
xgb_reg_study.optimize(objective, n_trials=100, show_progress_bar=True)

# Print the best parameters
print("Best parameters:", xgb_reg_study.best_params)
print("Best validation score (RMSLE):", xgb_reg_study.best_value)
