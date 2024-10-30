import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
import itertools

def preprocess_test_data(test_df, train_data_stats=None):
    """
    Preprocess test data following the same steps as training data
    
    Parameters:
    test_df: pandas DataFrame containing the test data
    train_data_stats: dict containing training data statistics for consistent preprocessing
    
    Returns:
    processed_df: preprocessed test DataFrame
    """
    # Create a copy to avoid modifying the original
    df = test_df.copy()
    
    # Drop ID column if it exists
    if 'id' in df.columns:
        df = df.drop(columns="id")
    
    # Identify numeric and categorical columns
    cat_cols = [col for col in df.select_dtypes(exclude=['int', 'float']).columns 
                if col not in ('id', 'loan_status')]
    num_cols = [col for col in df.select_dtypes(include=['int', 'float']).columns 
                if col not in ('id', 'loan_status')]
    
    # Handle missing values in numeric columns using Ridge imputation
    if df[num_cols].isna().any().any():
        ridge = Ridge(alpha=1.0)
        imputer = IterativeImputer(estimator=ridge, max_iter=1500, random_state=0, tol=1e-1)
        imputed_df = pd.DataFrame(
            imputer.fit_transform(df[num_cols]), 
            columns=num_cols, 
            index=df.index
        )
        df[num_cols] = imputed_df[num_cols]
    
    # Create new features
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['age_emp_length'] = df['person_age'] * df['person_emp_length']
    
    # Update numeric columns list after feature creation
    num_cols = [col for col in df.select_dtypes(include=['int', 'float']).columns 
                if col not in ('id', 'loan_status')]
    
    
    
    # Generate combination features from categorical variables
    def generate_new_features(df:pd.DataFrame, feature_cols:list[str]=None, top_features:int=5):
        df = df.copy()
        for col_pair in itertools.combinations(feature_cols, 2):
            new_col = f'{col_pair[0]}_{[1]}'
#           
            df[new_col] = ((df[feature_cols[0]] == str(index[0])) & (df[feature_cols[1]] == str(index[1]))).astype(int)
        return df, new_cols

    df_temp_2 = df.copy()
    new_features = []
    for col_pair in itertools.combinations(cat_cols, 2):
        df_temp_2, new_cols = generate_new_features(df=df_temp_2, feature_cols=col_pair)
        new_features.extend(new_cols)
    df = df_temp_2.copy()
    # Handle categorical variables
    # Convert cb_person_default_on_file to numeric
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    
    # Handle loan_grade using OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    df['loan_grade'] = ordinal_encoder.fit_transform(df[['loan_grade']])
    
    # Update categorical columns list
    cat_cols_for_onehot = [col for col in cat_cols 
                          if col not in ['cb_person_default_on_file', 'loan_grade']]
    
    # One-hot encode remaining categorical variables
    df = pd.get_dummies(data=df, columns=cat_cols_for_onehot)
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes('bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Optional: Convert to lower memory usage datatypes
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df

def verify_columns_match(train_df, test_df):
    """
    Verify that the columns in test_df match train_df (excluding loan_status)
    
    Parameters:
    train_df: processed training DataFrame
    test_df: processed test DataFrame
    
    Returns:
    bool: True if columns match, False otherwise
    """
    train_cols = set(train_df.columns) - {'loan_status'}
    test_cols = set(test_df.columns)
    
    missing_cols = train_cols - test_cols
    extra_cols = test_cols - train_cols
    
    if missing_cols:
        print("Missing columns in test data:", missing_cols)
    if extra_cols:
        print("Extra columns in test data:", extra_cols)
        
    return len(missing_cols) == 0 and len(extra_cols) == 0

# Example usage:
"""
# Load test data
test_df = pd.read_csv('test.csv')

# Preprocess test data
processed_test_df = preprocess_test_data(test_df)

# Load processed training data for verification
processed_train_df = pd.read_csv('loan_approval_data_train_original_processed_low_memory.csv')

# Verify columns match
columns_match = verify_columns_match(processed_train_df, processed_test_df)
print("Columns match:", columns_match)

if columns_match:
    # Save processed data
    processed_test_df.to_csv('processed_test_data.csv', index=False)
"""