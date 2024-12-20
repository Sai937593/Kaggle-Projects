from feature_eng_and_dataset_preparation import df_train_not_scaled, df_test_not_scaled
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_top_n_imp_cols(corr_matrix, n):
    c = abs(corr_matrix).sort_values(ascending=False)
    top_n_cols = c[:n].index
    return top_n_cols.tolist()


def prepare_train_validate(
    X, y,
    corr,
    top_n_cols=-1,
    target="premium_amount",
    norm=True,
    scale=False,
    
):
    
    predictor_cols = get_top_n_imp_cols(corr, n=top_n_cols)
    X = X.loc[:, predictor_cols]
    if norm:
        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X=X)
    if scale:
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X=X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return (X_train, X_test,y_train,  y_val)


df_train_final = df_train_not_scaled
df_test_final = df_test_not_scaled


def train_lin_reg(model, X_train, y_train, cv):
    scores = cross_validate(estimator=model, X=X_train, y=y_train, scoring=scorer, cv=cv, n_jobs=-1, return_train_score=True)
    mean_train_rmlse = np.mean(scores['train_score'])
    mean_val_rmsle = np.mean(scores['test_score'])
    train_mean_rmlse_scores.append(mean_train_rmlse)
    val_mean_rmlse_scores.append(mean_val_rmsle)
    print('rmlse_mean_train_scores', mean_train_rmlse)
    print('rmlse_mean_validation_scores',mean_val_rmsle)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


lin_reg = LinearRegression(fit_intercept=True)


scorer = make_scorer(rmsle)


train_mean_rmlse_scores = []
val_mean_rmlse_scores = []

df_train = df_train_not_scaled
target = 'premium_amount'
X = df_train.drop(columns=target)
y = df_train.loc[:, target]
corr = X.corrwith(y)
n_top_cols_array = np.random.choice(np.arange(2, 27), size=10, replace=False)
print(n_top_cols_array)
for n_top_col in n_top_cols_array:
    print('number of top cols = ', n_top_col)
    X_train, X_test, y_train,  y_test = prepare_train_validate(X, y, corr, n_top_col)
    train_lin_reg(lin_reg, X_train, y_train, cv=100)

print(train_mean_rmlse_scores, val_mean_rmlse_scores)

plt.plot(train_mean_rmlse_scores, label='train_mean_rmlse')
plt.plot(val_mean_rmlse_scores, label='val_mean_rmlse')
plt.legend()
plt.show()
