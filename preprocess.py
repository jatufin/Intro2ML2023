import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ['parentspecies']

def preprocess(df,
               categorical_features = CATEGORICAL_FEATURES,
               numerical_features=None,
               one_hot=None,
               scaler=None,
               target_column=None,
               drop_columns=[]
            ):
    """ Preprocesses the data and returns feature matrix X and target vector y

        Parameters:
            df (DataFrame)
            categorical_features (list)    
            target_column (string)
        
        Returns:
            X, y, feature_names, numerical_features, one_hot, scaler

            The feature_names array contains one_hot encoded names
    """
    # Drop given columns
    for d in drop_columns:
        if d in df.columns:
            df = df.drop([d], axis=1)

    if target_column:
        # log_10 transformation
        log10_column = f"{target_column}_log10"
        df[log10_column] = np.log10(df[target_column])
        # X-y split
        y = df[log10_column]
        df = df.drop([target_column, log10_column], axis=1)
    else:
        y = None

    X = df.copy()

    # The categorical features    
    # Replacing the "None" entries by the mode/most frequent entry
    # (tested without this step and result is so close)
    for feature in categorical_features:
        if feature not in X.columns:
            continue
        mode = X[feature].mode().iloc[0]
        # If the NA value is represented as a "None" string in the data
        X[feature] = X[feature].replace('None', None)
        X[feature] = X[feature].fillna(mode)

    # One-hot encoding the cateogrical feature
    # Categories are split to exclusive boolean columns
    if np.in1d(categorical_features, X.columns):
        if not one_hot:
            one_hot = OneHotEncoder()
            X_encoded = one_hot.fit_transform(X[categorical_features]).toarray()
        else:
            X_encoded = one_hot.transform(X[categorical_features]).toarray()
        one_hot_feature_names = one_hot.get_feature_names_out()
    else:
        X_encoded = []
        one_hot_feature_names = []

    # Feature scaling for the numerical features only (crucial for the SVR in particular)
    if not numerical_features:
        numerical_features = [col for col in X.columns if col not in categorical_features]
    X_scaled = X.copy()

    if not scaler:
        scaler = StandardScaler()
        X_scaled[numerical_features] = scaler.fit_transform(X_scaled[numerical_features])
    else:
        X_scaled[numerical_features] = scaler.transform(X_scaled[numerical_features])

    if len(X_encoded) != 0:    
        X_encoded = np.concatenate([X_scaled[numerical_features].values, X_encoded], axis=1)
    else:
        X_encoded = X_scaled

    feature_names = np.concatenate((numerical_features, one_hot_feature_names))

    return X_encoded, y, feature_names, numerical_features, one_hot, scaler
