import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
)

df = pd.DataFrame({"n": range(10), "y": range(20)})

X = df.drop(columns=["Depression"])
y = df.loc[:, "Depression"]

sss = StratifiedShuffleSplit(n_splits=1, random_state=2, test_size=0.2)
for full_idx, test_idx in sss.split(X, y):
    X_full, y_full = X.iloc[full_idx], y.iloc[full_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

for tr_idx, val_idx in sss.split(X_full, y_full):
    X_train, y_train = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
    X_val, y_val = X_full.iloc[val_idx], y_full.iloc[val_idx]

print(X_train.shape, X_val.shape, X_test.shape)

pipe = Pipeline(
    [
        (
            "poly",
            PolynomialFeatures(include_bias=False, interaction_only=True, degree=2),
        ),
        ("scaler", StandardScaler()),
    ]
)

X_train_transformed = pipe.fit_transform(X_train)
X_full_transformed = pipe.fit_transform(X_full)
X_val_transformed = pipe.fit_transform(X_val)
X_test_transformed = pipe.fit_transform(X_test)

print(X_train_transformed.shape, X_val_transformed.shape, X_test_transformed.shape)


ridge_classifier = RidgeClassifier(random_state=32, alpha=1.0)
scores = {}
scores["accuracy"] = make_scorer(accuracy_score)
scores["precision"] = make_scorer(precision_score)
scores["f1_score"] = make_scorer(f1_score)
scores["recall_score"] = make_scorer(recall_score)
cv = StratifiedKFold(n_splits=50, random_state=32, shuffle=True)
scores = cross_validate(
    ridge_classifier, X_full_transformed, y_full, scoring=scores, cv=cv, n_jobs=-1
)


for score, values in scores.items():
    if score not in ("fit_time", "score_time"):
        print(score, np.mean(values))


ridge_classifier.fit(X_full_transformed, y_full)
y_pred = ridge_classifier.predict(X_test_transformed)

for score in (accuracy_score, precision_score, f1_score, recall_score):
    print(score(y_test, y_pred))

rfc = RandomForestClassifier(
    n_estimators=250,
    criterion="entropy",
    max_depth=5,
    bootstrap=True,
    n_jobs=-1,
    random_state=32,
)
rfc_scores = cross_validate(
    rfc, X_full_transformed, y_full, scoring=scores, cv=cv, n_jobs=-1
)
for score, values in rfc_scores.items():
    if score not in ("fit_time", "score_time"):
        print(score, np.mean(values))

rfc.fit(X_full_transformed, y_full)
y_pred = rfc.predict(X_test_transformed)

for score in (accuracy_score, precision_score, f1_score, recall_score):
    print(score(y_test, y_pred))


dfull = xgb.DMatrix(data=X_full_transformed, label=y_full)
dtest = xgb.DMatrix(data=X_test_transformed, label=y_test)
xgb_params = {
    "booster": "dart",
    "max_depth": 5,
    "learning_rate": 0.1,
    "sample_type": "uniform",
    "normalize_type": "tree",
    "rate_drop": 0.3,
    "skip_drop": 0.1,
}

num_round = 50

bst = xgb.train(xgb_params, dfull, num_round)
preds = bst.predict(dtest)

preds_array = preds >= 0.5
preds_array = preds_array.astype(int)
for score in (accuracy_score, precision_score, f1_score, recall_score):
    print(score(y_test, preds_array))


 

