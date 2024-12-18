from linear_regression_training import (
    prepare_train_validate,
    scorer,
    df_train_not_scaled,
    target,
)
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np

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


model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    reg_alpha=0,
    reg_lambda=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    random_state=42
)

scores = cross_validate(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)
print(np.mean(scores['train_score']))
print(np.mean(scores['test_score']))