import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from preprocess import preprocess

#RANDOM_SEED = 42
RANDOM_SEED = None

TARGET_COLUMN = "pSat_Pa"
DROP_COLUMNS = ["Id"]

DATA_FILE = "train.csv"

df = pd.read_csv(DATA_FILE)

X, y, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)

# Polynomial degrees
N = 1

for n in range(1,N+1):
    features = PolynomialFeatures(degree=n, interaction_only=True, include_bias=False)
    X_poly = features.fit_transform(X)
    feature_names = features.get_feature_names_out()

    print("Feature names:")
    print(feature_names)
    #model = LinearRegression()
    model = Lasso()
    model.fit(X_poly, y)

    print(f"POLYNOMIAL: {n}")
    print(f"    Non-zero LASSO coefficients (total: {len(feature_names)})")

    for i in range(len(feature_names)):
        coef = model.coef_[i]
        if  coef != 0:
            print(f"        {i:4} {feature_names[i]:20} = {coef:.4f}")


