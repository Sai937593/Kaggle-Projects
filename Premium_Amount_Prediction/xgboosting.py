import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_log_error

X_train = 1
y_train = 0
X_val = 1
y_val = 0

def rmlse(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))



