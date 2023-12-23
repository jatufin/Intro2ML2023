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

X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)

X_df = pd.DataFrame(X, columns=columns)

# Polynomial degrees
N = 1

# LASSO PARAMETERS
ALPHA = 0.02
MAX_ITER = 10000

for n in range(1,N+1):
    features = PolynomialFeatures(degree=n, interaction_only=True, include_bias=False)
    X_poly = features.fit_transform(X_df)
    feature_names = features.get_feature_names_out()

    #model = LinearRegression()
    model = Lasso(alpha=ALPHA, max_iter=MAX_ITER)
    model.fit(X_poly, y)

    print(f"POLYNOMIAL: {n}")
    print(f"    Non-zero LASSO coefficients (total: {len(feature_names)})")

    for i in range(len(feature_names)):
        coef = model.coef_[i]
        if  coef != 0:
            #print(f"        {i:4} {feature_names[i]:20} = {coef:.4f}")
            print(feature_names[i])

