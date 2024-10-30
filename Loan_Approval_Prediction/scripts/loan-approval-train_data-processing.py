#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install imblearn\n')


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("chilledwanker/loan-approval-prediction")

print("Path to dataset files:", path)


# In[4]:


import pandas as pd

train_df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')
original_df = pd.read_csv('/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv')


# In[5]:


train_df = train_df.drop(columns="id")
train_df = pd.concat([train_df, original_df], axis=0)


# In[6]:


cat_cols = [col for col in train_df.select_dtypes(exclude=['int', 'float']).columns if col not in ('id', 'loan_status')]
num_cols = [col for col in train_df.select_dtypes(include=['int', 'float']).columns if col not in ('id', 'loan_status')]

cat_cols, num_cols


# In[7]:


train_df.isna().sum()


# In[8]:


na_df = train_df.isna().sum()
na_cols  = [col for col in na_df.index if na_df[col] > 0]
na_cols


# In[9]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso
import copy

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

imputed_dfs = {}
for model in [ridge, lasso]:
    df = copy.deepcopy(train_df)
    df = df[num_cols]
    print(f'imputing using {model}')
    imputer = IterativeImputer(estimator=model, max_iter=1500, random_state=0, tol=1e-1)
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index )
    model_name = model.__class__.__name__
    imputed_dfs[model_name] = imputed_df



# In[10]:


ridge_imputed_df = imputed_dfs['Ridge']
train_df[num_cols] = ridge_imputed_df[num_cols]
train_df.isna().any()


# In[11]:


train_df['income_to_loan_ratio'] = train_df['person_income'] / train_df['loan_amnt']
train_df['age_emp_length']  = train_df['person_age'] * train_df['person_emp_length']


# In[12]:


cat_cols = [col for col in train_df.select_dtypes(exclude=['int', 'float']).columns if col not in ('id', 'loan_status')]
num_cols = [col for col in train_df.select_dtypes(include=['int', 'float']).columns if col not in ('id', 'loan_status')]

cat_cols, num_cols


# In[13]:


import matplotlib.pyplot as plt
import copy 
def box_plots_num_cols(df:pd.DataFrame, num_columns:list[str]):
    df = copy.deepcopy(df)
    base_width = 5
    base_height = 3
    columns = num_columns
    cols = round(len(columns) / 2)
    rows = len(columns) - cols
    fig_width = cols * base_width
    fig_height = rows * base_height
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if rows > 1 else [axes]
    for i, col in enumerate(columns):
        axes[i].boxplot(df[col])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()

box_plots_num_cols(train_df, num_cols)


# In[14]:


from scipy import stats
import numpy as np
import pandas as pd

def remove_outliers(df: pd.DataFrame, cols: list[str], beta: float, cols_2: list[str] = None, beta2: float = None):
    # Avoid deep copy unless truly needed to optimize memory usage
    df_filtered = df.copy()
    beta_dict = {col: beta2 if cols_2 and col in cols_2 and beta2 is not None else (6.0 if col == 'person_income' else beta) for col in cols}

    for col in cols:
        data = df_filtered[col]
        
        if data.empty:
            print(f"Column {col} is empty, skipping...")
            continue
        
        # Use pandas quantile for faster percentile calculation
        iqr = stats.iqr(data)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        beta_val = beta_dict[col]
        
        print(f'{col} - beta: {beta_val}')
        outlier_low = q1 - beta_val * iqr
        outlier_high = q3 + beta_val * iqr
        df_filtered = df_filtered[(data >= outlier_low) & (data <= outlier_high)]
        
    return df_filtered

# Run with specified parameters
df_clean = remove_outliers(train_df, num_cols, beta=1.5, cols_2=['loan_amnt'], beta2=3.0)


# In[15]:


def ins(indexx, clean_desc, orig_desc):
    # Fetch the row for the current index without extra conversions
    clean_values = clean_desc.loc[indexx].values
    orig_values = orig_desc.loc[indexx].values
    diff = orig_values - clean_values
    # Create DataFrame once for both sets of values
    res_df = pd.DataFrame({
        'orig': orig_values, 
        'clean': clean_values,
        'difference':diff
    }, index=num_cols)
    print(res_df.sort_values(by=['difference'], ascending=False, key= lambda x: abs(x)))

# Compute describe() once for both DataFrames
clean_desc = df_clean[num_cols].describe()
orig_desc = train_df[num_cols].describe()

# Loop through the index values
for index in ['count', 'mean', 'std', 'min', 'max']:
    
    print(f'\t\t\t{index.upper()}')
    print('*' * 50)
    ins(index, clean_desc, orig_desc)  # Pass precomputed descriptions
    print('-' * 50)


# In[16]:


cat_cols = [col for col in df_clean.select_dtypes(exclude=['int', 'float']).columns if col not in ('id', 'loan_status')]
num_cols = [col for col in df_clean.select_dtypes(include=['int', 'float']).columns if col not in ('id', 'loan_status')]

cat_cols, num_cols


# In[17]:


for col in cat_cols:
    print(df_clean[col].value_counts())
    print()


# In[18]:


df_temp = df_clean[['person_home_ownership', 'loan_intent', 'loan_status']]
def generate_new_features(df:pd.DataFrame, feature_cols:list[str]=None):
    new_cols = []
    if feature_cols is None or len(feature_cols) != 2  :
        print('provide the 2 feature cols to create new features')
        return 
    df = df.copy()
    agg_sums = df.groupby([feature_cols[0], feature_cols[1]]).loan_status.sum().sort_values(ascending=False)
    print(agg_sums)
    
    min_sum_count = int(input('enter the minimum sum count for to create a combination of the features: '))
    filtered_indices = agg_sums[agg_sums >= min_sum_count].index
    
    print(f'creating new features from combinations of top {len(filtered_indices)} features: ')
    for index in filtered_indices:
        new_col = f'{index[0]}_{index[1]}'
#         print(f'creating new column {new_col}:')
        df[new_col] = ((df[feature_cols[0]] == str(index[0])) & (df[feature_cols[1]] == str(index[1]))).astype(int)
        new_cols.append(new_col)
    return df, new_cols


# In[19]:


import itertools
from IPython.display import clear_output
df_temp_2 = df_clean.copy()
new_features = []
for col_pair in itertools.combinations(cat_cols, 2):
    df_temp_2, new_cols = generate_new_features(df=df_temp_2, feature_cols=col_pair)
    new_features.extend(new_cols)
df_temp_2.columns


# In[20]:


df_temp_2.info()


# In[21]:


for col in cat_cols:
    print(col, df_temp_2[col].unique())


# In[22]:


df_temp_2['cb_person_default_on_file'] = df_temp_2['cb_person_default_on_file'].map(lambda x: 1 if x == 'Y' else 0)
df_temp_2[cat_cols].head()


# In[23]:


cat_cols.remove('cb_person_default_on_file')
cat_cols


# In[24]:


for col in cat_cols:
    print(col, df_temp_2[col].unique())
    


# In[25]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ordencoder = OrdinalEncoder()
df_temp_2['loan_grade'] = ordencoder.fit_transform(df_temp_2[['loan_grade']])
df_temp_2['loan_grade'].unique()
cat_cols.remove('loan_grade')


# In[26]:


category_mapping = {category: code for category, code in zip(ordencoder.categories_[0], range(len(ordencoder.categories_[0])))}
print(category_mapping)


# In[27]:


df_temp_2 = pd.get_dummies(data=df_temp_2, columns=cat_cols)


# In[28]:


bool_cols = df_temp_2.select_dtypes('bool').columns
df_temp_2[bool_cols] = df_temp_2[bool_cols].astype(int)
df_temp_2.info()


# In[29]:


df_temp_2.describe()


# In[30]:


df_temp_2['loan_status'].value_counts(normalize=True)

df_temp_2.shape


# In[31]:


from imblearn.over_sampling import ADASYN, SVMSMOTE

adasyn = ADASYN(n_jobs=-1, n_neighbors=250, random_state=0, sampling_strategy=0.3)
# svmsmote = SVMSMOTE(sampling_strategy=0.3, random_state=0, k_neighbors=30, n_jobs=-1)
all_feature_cols = [col for col in df_temp_2.columns if col != 'loan_status']
target = 'loan_status'
X = df_temp_2.drop(columns=[target])
y = df_temp_2.loc[:, target]

X_resampled, y_resampled = adasyn.fit_resample(X, y)

df_over_sampled = pd.DataFrame(X_resampled, columns=all_feature_cols)
df_over_sampled[target] = y_resampled

df_temp_2[target].value_counts(normalize=True), df_over_sampled[target].value_counts(normalize=True)


# In[32]:


df_over_sampled.info()


# In[41]:


import copy

df_over_sampled_low_memory = copy.deepcopy(df_over_sampled)
for col in df_over_sampled_low_memory.select_dtypes(include=['int64']).columns:
    df_over_sampled_low_memory[col] = df_over_sampled_low_memory[col].astype('int32')

for col in df_over_sampled_low_memory.select_dtypes(include=['float64']).columns:
    df_over_sampled_low_memory[col] = df_over_sampled_low_memory[col].astype('float32')
df_over_sampled_low_memory.info()


# In[40]:


for col in df_over_sampled.select_dtypes('float64').columns:
    print(col)
    print(np.max(df_over_sampled[col]))
    print(np.min(df_over_sampled[col]))
    print('-'*50)


# In[43]:


df_over_sampled.to_csv('loan_approval_data_train_original_processed_high_memory.csv')
df_over_sampled_low_memory.to_csv('loan_approval_data_train_original_processed_low_memory.csv')


# In[ ]:




