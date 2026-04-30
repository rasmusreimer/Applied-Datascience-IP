"""
This module contains utility functions for data preprocessing, feature selection, and other common tasks in the project.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import os



def fast_preprocess_data(data_path, target_col, test_size=0.2): 
    """
    This function performs fast data preprocessing, including loading the dataset, checking for missing values, selects features based on mutual information and correlation,
    splits into training and testing sets, and applies standard scaling to the features. It returns the preprocessed training and testing data.

    Most suitable for the NN model as it is faster than the more comprehensive preprocess_data function and still provides a good selection of features based on mutual information and correlation.
    data_path and target_col should be given as apostrophated  eg: 'Data/fast_dataprep_input.h5' and 'p_Truth_isElectron', test size should be given as a float between 0 and 1 eg: 0.2 for 20% test data and 80% training data.
    """
    # Load the dataset
    data = pd.read_hdf(data_path)
    y = data[target_col]
    data.drop(columns=[target_col], inplace=True)
    X = data

    #Check for missing values, raise alert if there are any
    if X.isnull().sum().sum() > 0:
        print("Warning: Missing values detected in the dataset NOT suitable for NN.")
        print(X.isnull().sum())
    else:
        print("No missing values detected in the dataset.")




    # Run MI first
    mi_scores = pd.Series(mutual_info_classif(X, y), index=X.columns)

    # Then drop the lower-MI feature from each correlated pair
    upper = X.corr().abs()
    upper = upper.where(np.triu(np.ones(upper.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > 0.95].tolist()
        for other in correlated_with:
            # Drop whichever has lower MI
            if mi_scores[col] < mi_scores[other]:
                to_drop.add(col)
            else:
                to_drop.add(other)

    X = X.drop(columns=to_drop)

    print(f"Dropped {len(to_drop)} highly correlated features")

    # MI on remaing features
    mi_scores = pd.Series(mutual_info_classif(X, y), index=X.columns)   
    mi_scores.sort_values(ascending=False, inplace=True)

    # Assigns the 15 highest MI scores to X to use as input features for the model
    X = X[mi_scores.head(15).index]

    #split the data into training and testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


    return X_train, X_test, y_train, y_test


