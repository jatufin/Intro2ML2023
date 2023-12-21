import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_validate
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
    
print(f"Cross validating SVR (CV={CV})...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

cv_results = cross_validate(model_svr, X, y, cv=CV)
print("Cross validation scores:")
scores = cv_results["test_score"]
print(scores)

r2 = scores.mean()
print(f"Mean R2 score for SVM with selected columns ({method}): {r2}")

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
    
print(f"Cross validating SVR (CV={CV})...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

cv_results = cross_validate(model_svr, X, y, cv=CV)
print("Cross validation scores:")
scores = cv_results["test_score"]
print(scores)

r2 = scores.mean()
print(f"Mean R2 score for SVM with selected columns ({method}): {r2}")

# SVR hyperparameters: kernel=rbf C=1.0 gamma=scale
# ====================================
# feature reduction
# Selected columns:
# ['aldehyde', 'aromatic.hydroxyl', 'C.C.C.O.in.non.aromatic.ring', 'carbonylperoxyacid', 'carbonylperoxynitrate', 'carboxylic.acid', 'hydroperoxide', 'hydroxyl..alkyl.', 'ketone', 'nitro', 'nitroester', 'NumHBondDonors', 'NumOfAtoms', 'NumOfC', 'NumOfConf', 'NumOfConfUsed', 'NumOfN', 'parentspecies', 'peroxide', 'pSat_Pa']
# Cross validating SVR (CV=10)...
# Cross validation scores:
# [0.76405281 0.71947577 0.62028288 0.73629226 0.7156401  0.64183068
#  0.68875946 0.71053148 0.74660047 0.71636969]
# Mean R2 score for SVM with selected columns (feature reduction): 0.7059835603642994
# ====================================
# LASSO
# Selected columns:
# ['aldehyde', 'aromatic.hydroxyl', 'C.C..non.aromatic.', 'carbonylperoxyacid', 'carbonylperoxynitrate', 'carboxylic.acid', 'ether..alicyclic.', 'hydroperoxide', 'ketone', 'MW', 'nitroester', 'NumHBondDonors', 'NumOfC', 'NumOfConf', 'parentspecies_apin', 'peroxide']
# Cross validating SVR (CV=10)...
# Cross validation scores:
# [0.77043208 0.70616803 0.60902101 0.72338666 0.7164575  0.6280325
#  0.6817502  0.70392669 0.74398637 0.70570995]
# Mean R2 score for SVM with selected columns (LASSO): 0.698887100043793


