import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from preprocess import preprocess

#RANDOM_SEED = 42
RANDOM_SEED = None

TARGET_COLUMN = "pSat_Pa"
DROP_COLUMNS = ["Id"]

# SVR HYPERPARAMETERS
SVR_KERNEL="rbf"
SVR_C=1.0
SVR_GAMMA="scale"

#SVR_KERNEL="rbf"
#SVR_C=0.3
#SVR_GAMMA=0.1

# TRAIN TEST SPLIT
TEST_SIZE = 0.3

# CROSS VALIDATION
CV = 10

DROP_COLUMNS = ["Id"]

selected_columns_01 = [
    "aldehyde",
    "aromatic.hydroxyl",
    "C.C.C.O.in.non.aromatic.ring",
    "carbonylperoxyacid",
    "carbonylperoxynitrate",
    "carboxylic.acid",
    "hydroperoxide",
    "hydroxyl..alkyl.",
    "ketone",
    "nitro",
    "nitroester",
    "NumHBondDonors",
    "NumOfAtoms",
    "NumOfC",
    "NumOfConf",
    "NumOfConfUsed",
    "NumOfN",
    "parentspecies",
    "peroxide",
    TARGET_COLUMN # Features are selected from the original dataset
]

selected_columns_02 = [
    "aldehyde",
    "aromatic.hydroxyl",
    "C.C..non.aromatic.",
    "carbonylperoxyacid",
    "carbonylperoxynitrate",
    "carboxylic.acid",
    "ether..alicyclic.",
    "hydroperoxide",
    "ketone",
    "MW",
    "nitroester",
    "NumHBondDonors",
    "NumOfC",
    "NumOfConf",
    "parentspecies_apin",
    "peroxide",
]

TRAIN_DATA_FILE = "train.csv"

df_data_file = pd.read_csv(TRAIN_DATA_FILE)

print(f"SVR hyperparameters: kernel={SVR_KERNEL} C={SVR_C} gamma={SVR_GAMMA}")

method = "feature reduction"
print("====================================")
print(method)
print("Selected columns:")
print(selected_columns_01)

# The features are selected from the original dataset
df = df_data_file[selected_columns_01].copy()

X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

print(f"Training SVR...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

model_svr.fit(X_train, y_train)
print("trained!")

print("Predicting...")
y_test_pred = model_svr.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
print(f"R2 score for SVM with selected columns ({method}): {r2}")

method = "LASSO"
print("====================================")
print(method)
print("Selected columns:")
print(selected_columns_02)

df = df_data_file.copy()
X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)

# The features are selected from the one-hot encoded dataset
X_df = pd.DataFrame(X, columns=columns)
X = X_df[selected_columns_02].to_numpy()
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

print(f"Training SVR...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

model_svr.fit(X_train, y_train)
print("trained!")

print("Predicting...")
y_test_pred = model_svr.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
print(f"R2 score for SVM with selected columns ({method}): {r2}")

