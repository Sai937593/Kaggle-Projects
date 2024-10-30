#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from imblearn.over_sampling import ADASYN
import copy

def load_data(train_path, test_path, submission_path, original_data_path):
    """Load and combine the datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(submission_path)
    original_df = pd.read_csv(original_data_path)
    
    # Drop ID and combine with original data
    train_df = train_df.drop(columns="id")
    train_df = pd.concat([train_df, original_df], axis=0)
    
    return train_df, test_df, sub_df

def identify_columns(df):
    """Identify categorical and numerical columns"""
    cat_cols = [col for col in df.select_dtypes(exclude=['int', 'float']).columns 
                if col not in ('id', 'loan_status')]
    num_cols = [col for col in df.select_dtypes(include=['int', 'float']).columns 
                if col not in ('id', 'loan_status')]
    return cat_cols, num_cols

def handle_missing_values(df, num_cols):
    """Handle missing values using iterative imputation"""
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)
    
    imputed_dfs = {}
    for model in [ridge, lasso]:
        df_copy = copy.deepcopy(df)
        df_copy = df_copy[num_cols]
        print(f'imputing using {model}')
        imputer = IterativeImputer(estimator=model, max_iter=1500, random_state=0, tol=1e-1)
        imputed_df = pd.DataFrame(imputer.fit_transform(df_copy), 
                                columns=df_copy.columns, 
                                index=df_copy.index)
        model_name = model.__class__.__name__
        imputed_dfs[model_name] = imputed_df
    
    return imputed_dfs['Ridge']

def create_features(df):
    """Create new features"""
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['age_emp_length'] = df['person_age'] * df['person_emp_length']
    return df

def remove_outliers(df, num_cols):
    """Remove outliers using IQR method"""
    df_filtered = df.copy()
    
    for col in num_cols:
        data = df_filtered[col]
        if data.empty:
            print(f"Column {col} is empty, skipping...")
            continue
        
        iqr = stats.iqr(data)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        beta = 6.0 if col == 'person_income' else (3.0 if col == 'loan_amnt' else 1.5)
        
        print(f'{col} - beta: {beta}')
        outlier_low = q1 - beta * iqr
        outlier_high = q3 + beta * iqr
        df_filtered = df_filtered[(data >= outlier_low) & (data <= outlier_high)]
    
    return df_filtered

def generate_categorical_combinations(df, cat_cols):
    """Generate new features from combinations of categorical variables"""
    df_temp = df.copy()
    new_features = []
    
    for col_pair in itertools.combinations(cat_cols, 2):
        agg_sums = df_temp.groupby(list(col_pair)).loan_status.sum().sort_values(ascending=False)
        # Note: In the original notebook this was interactive. 
        # Here we're setting a default minimum sum count of 5
        min_sum_count = 5
        filtered_indices = agg_sums[agg_sums >= min_sum_count].index
        
        for index in filtered_indices:
            new_col = f'{index[0]}_{index[1]}'
            df_temp[new_col] = ((df_temp[col_pair[0]] == str(index[0])) & 
                               (df_temp[col_pair[1]] == str(index[1]))).astype(int)
            new_features.append(new_col)
    
    return df_temp, new_features

def encode_categorical_variables(df, cat_cols):
    """Encode categorical variables"""
    # Convert default_on_file to numeric
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    cat_cols.remove('cb_person_default_on_file')
    
    # Encode loan_grade
    ordencoder = OrdinalEncoder()
    df['loan_grade'] = ordencoder.fit_transform(df[['loan_grade']])
    cat_cols.remove('loan_grade')
    
    # One-hot encode remaining categorical variables
    df = pd.get_dummies(data=df, columns=cat_cols)
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df

def balance_dataset(df, target='loan_status'):
    """Balance the dataset using ADASYN"""
    all_feature_cols = [col for col in df.columns if col != target]
    X = df.drop(columns=[target])
    y = df.loc[:, target]
    
    adasyn = ADASYN(n_jobs=-1, n_neighbors=250, random_state=0, sampling_strategy=0.3)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    df_balanced = pd.DataFrame(X_resampled, columns=all_feature_cols)
    df_balanced[target] = y_resampled
    
    return df_balanced

def optimize_memory(df):
    """Optimize memory usage by converting to smaller datatypes"""
    df_optimized = copy.deepcopy(df)
    
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = df_optimized[col].astype('int32')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = df_optimized[col].astype('float32')
    
    return df_optimized

def main():
    # Define file paths
    train_path = 'playground-series-s4e10/train.csv'
    test_path = 'playground-series-s4e10/test.csv'
    submission_path = 'playground-series-s4e10/sample_submission.csv'
    original_data_path = 'loan-approval-prediction/credit_risk_dataset.csv'
    
    # Load data
    train_df, test_df, sub_df = load_data(train_path, test_path, submission_path, original_data_path)
    
    # Process training data
    cat_cols, num_cols = identify_columns(train_df)
    
    # Handle missing values
    ridge_imputed_df = handle_missing_values(train_df, num_cols)
    train_df[num_cols] = ridge_imputed_df[num_cols]
    
    # Create new features
    train_df = create_features(train_df)
    
    # Remove outliers
    df_clean = remove_outliers(train_df, num_cols)
    
    # Generate categorical combinations
    df_clean, new_features = generate_categorical_combinations(df_clean, cat_cols)
    
    # Encode categorical variables
    df_clean = encode_categorical_variables(df_clean, cat_cols.copy())
    
    # Balance dataset
    df_balanced = balance_dataset(df_clean)
    
    # Create memory-optimized version
    df_balanced_low_memory = optimize_memory(df_balanced)
    
    # Save processed datasets
    df_balanced.to_csv('loan_approval_data_train_original_processed_high_memory.csv', index=False)
    df_balanced_low_memory.to_csv('loan_approval_data_train_original_processed_low_memory.csv', index=False)

if __name__ == "__main__":
    main()