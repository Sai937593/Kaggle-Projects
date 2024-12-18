import pandas as pd
from sklearn.impute import SimpleImputer

train_file_path = "/kaggle/input/playground-series-s4e12/train.csv"
test_file_path = "/kaggle/input/playground-series-s4e12/test.csv"
submission_file_path = "/kaggle/input/playground-series-s4e12/sample_submission.csv"

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)
df_submission = pd.read_csv(submission_file_path)

target = "premium_amount"
df_train.columns = [col.lower().replace(" ", "_") for col in df_train.columns]
df_test.columns = [col.lower().replace(" ", "_") for col in df_test.columns]

df_train = df_train.drop(columns=["id"])
cat_cols = [
    col for col in df_train.select_dtypes(include=["object"]).columns if col != target
]
num_cols = [
    col
    for col in df_train.select_dtypes(include=["int", "float"]).columns
    if col != target
]

cat_cols, num_cols

for col in df_train.columns:
    print(col, df_train[col].dtype, df_train[col].isna().sum())


cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="mean")

df_train_imputed = df_train.copy()
df_train_imputed[cat_cols] = cat_imputer.fit_transform(df_train[cat_cols], cat_cols)
df_train_imputed[num_cols] = num_imputer.fit_transform(df_train[num_cols], num_cols)
print(df_train_imputed.isna().sum())


test_cat_imputer = SimpleImputer(strategy="most_frequent")
test_num_imputer = SimpleImputer(strategy="mean")
test_cat_cols = df_test.select_dtypes(include=["object"]).columns
test_num_cols = df_test.select_dtypes(exclude=["object"]).columns
df_test_imputed = df_test.copy()
df_test_imputed[test_cat_cols] = test_cat_imputer.fit_transform(df_test[test_cat_cols])
df_test_imputed[test_num_cols] = test_num_imputer.fit_transform(df_test[test_num_cols])
print(df_test_imputed.isna().sum())
for col in num_cols:
    if df_test_imputed[col].nunique() <= 10:
        df_test_imputed[col] = df_test_imputed[col].astype("int64")

for col in num_cols:
    if df_train_imputed[col].nunique() <= 10:
        df_train_imputed[col] = df_train_imputed[col].astype("Int64")

import seaborn as sns
import matplotlib.pyplot as plt

n_num_cols = len(num_cols)
fig, axes = plt.subplots(n_num_cols // 2, 2, figsize=(8, 8))

for col, ax in zip(num_cols, axes.flatten()):
    sns.boxplot(df_train_imputed[col], ax=ax)
    ax.set_title(str(col))
    plt.show()


from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=250)
outlier_labels = lof.fit_predict(df_train_imputed[num_cols])
df_train_imputed["outlier"] = outlier_labels

df_clean = df_train_imputed[df_train_imputed["outlier"] == 1]
print(df_clean.shape, df_train_imputed.shape)


def check_train_test_cols(train_df, test_df):
    test_cols = set(test_df.columns)
    train_cols = set(train_df.columns)

    return train_cols.difference(test_cols)


df_clean = df_clean.drop(columns=["policy_start_date"])
df_test_imputed = df_test_imputed.drop(columns=["policy_start_date"])

df_train_clean = df_clean.copy()
train_cat_cols = [col for col in cat_cols if col != "policy_start_date"]
train_num_cols = num_cols

for col in train_cat_cols:
    print(col, df_clean[col].unique())


df_train_encoded = pd.get_dummies(data=df_train_clean, columns=train_cat_cols)
df_test_encoded = pd.get_dummies(
    data=df_test_imputed,
    columns=[col for col in test_cat_cols if col != "policy_start_date"],
)
df_train_encoded = df_train_encoded.drop(columns=["outlier"])
check_train_test_cols(df_train_encoded, df_test_encoded)

all_corr = df_train_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 20))  # Adjusted figure size

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    data=all_corr,
    vmin=-1.0,
    vmax=1.0,
    cmap="coolwarm",
    annot=True,
    annot_kws={"size": 8},
    linewidths=0.5,
)

# Set the title
plt.title("Correlation Matrix Heatmap", fontsize=20)

# Display the heatmap
plt.show()


def find_collinear_columns(corr_matrix, threshold):
    collinear_pairs = []
    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_pair = (columns[i], columns[j])
                collinear_pairs.append(col_pair)

    return collinear_pairs


collinear_columns = find_collinear_columns(all_corr, 0.8)
print("Column pairs with collinearity exceeding threshold:", collinear_columns)

collinear_cols_to_remove = [collinear_columns[i][0] for i in range(len(collinear_columns))]

df_train_collinear_removed = df_train_encoded.drop(columns=collinear_cols_to_remove)
df_test_collinear_removed = df_test_encoded.drop(columns=collinear_cols_to_remove)

predictor_cols = [col for col in df_train_encoded.columns if col != target]
X_corr_y = df_train_encoded[predictor_cols].corrwith(df_train_encoded[target])

abs(X_corr_y).sort_values(ascending=False)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Your existing code
scaler = StandardScaler()
X = df_train_encoded.drop(columns=target)
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

# Get feature importance scores
feature_importance = np.abs(pca.components_)

# Get top features for each component
n_features = X.shape[1]  # Number of original features
selected_features = []

for i in range(pca.n_components_):
    # Get indices of top features for this component
    top_features_idx = np.argsort(feature_importance[i])[::-1]
    # Get the corresponding feature names
    top_features = X.columns[top_features_idx]
    # Add to selected features
    selected_features.extend(top_features)

# Remove duplicates while preserving order
selected_features = list(dict.fromkeys(selected_features))
# Take only as many features as we have components
selected_features = selected_features[:X_reduced.shape[1]]

# Create DataFrame with selected original feature names
df_train_reduced = pd.DataFrame(X_reduced, columns=selected_features, index=X.index)
df_train_reduced[target] = df_train_encoded.loc[:, target]
print(df_train_reduced.shape, df_train_encoded.shape)
df_train_reduced.head(5)

df_test_reduced = df_test_encoded[[col for col in df_train_reduced.columns if col!= target]]
df_test_reduced_scaled = pd.DataFrame(scaler.fit_transform(df_test_reduced), columns=df_test_reduced.columns, index=df_test_reduced.index)
check_train_test_cols(df_train_reduced, df_test_reduced_scaled)


df_train_final = df_train_reduced.copy()
df_test_final = df_test_reduced_scaled.copy()
df_train_final.to_csv('df_train_preprocessed.csv')
df_test_final.to_csv('df_test_preprocessed.csv')

df_train_not_scaled = df_train_encoded[selected_features + [target]]
df_test_not_scaled = df_test_encoded[selected_features]
print(check_train_test_cols(df_train_not_scaled, df_test_not_scaled))
df_train_not_scaled.to_csv('df_train.csv')
df_test_not_scaled.to_csv('df_test.csv')
