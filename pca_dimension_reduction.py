import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.decomposition import PCA
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from preprocess import preprocess

# PCA COMPONENTS
N_PCA = 18
#N_PCA = None

# SVR HYPERPARAMS
SVR_KERNEL="rbf"
SVR_C=1.0
SVR_GAMMA="scale"
#SVR_KERNEL = "rbf"
#SVR_C = 0.3
#SVR_GAMMA = 0.1

# TRAIN TEST SPLIT
TEST_SIZE = 0.3

# CROSS VALIDATION
CV = 10

#RANDOM_SEED = 42
RANDOM_SEED = None

TARGET_COLUMN = "pSat_Pa"


#DROP_COLUMNS = ["Id"]
# Drop categorical
DROP_COLUMNS = ["Id", "parentspecies"]

DATA_FILE = "train.csv"

df = pd.read_csv(DATA_FILE)

X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)

print(f"SVR hyperparameters: kernel={SVR_KERNEL} C={SVR_C} gamma={SVR_GAMMA}")

########################################################
# Plain regression without PCA

# print(f"Cross validating SVR without PCA (CV={CV})...")
# model_svr = SVR(
#     kernel=SVR_KERNEL,
#     C=SVR_C,
#     gamma=SVR_GAMMA
# )

# cv_results = cross_validate(model_svr, X, y, cv=CV)
# print("Cross validation scores:")
# scores = cv_results["test_score"]
# print(scores)

# r2 = scores.mean()
# print(f"Mean R2 score for SVR without PCA: {r2}")

########################################################
# Regression with PCA
model_pca = PCA(n_components=N_PCA)
model_pca.fit(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

X = model_pca.transform(X)

print(f"Cross validating SVR with PCA (number of PCA components: {N_PCA}, CV={CV})...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

cv_results = cross_validate(model_svr, X, y, cv=CV)
print("Cross validation scores:")
scores = cv_results["test_score"]
print(f"'PCA-{N_PCA}': {scores}")

print(scores)

r2 = scores.mean()
print(f"Mean R2 score for SVR with PCA: {r2}")

plot_y = model_pca.explained_variance_ratio_
plot_x = range(len(plot_y))
plt.title("Explained variance ratio of each principal component")
plt.xlabel("Principal component")
plt.bar(plot_x, plot_y)
plt.show()
