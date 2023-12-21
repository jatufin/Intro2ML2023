import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.decomposition import PCA
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from preprocess import preprocess

# PCA COMPONENTS
N_PCA = 18

# SVR HYPERPARAMS
SVR_KERNEL = "rbf"
SVR_C = 0.3
SVR_GAMMA = 0.1

# TRAIN TEST SPLIT
TEST_SIZE = 0.3

#RANDOM_SEED = 42
RANDOM_SEED = None

TARGET_COLUMN = "pSat_Pa"
DROP_COLUMNS = ["Id"]

DATA_FILE = "train.csv"

df = pd.read_csv(DATA_FILE)

X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)

########################################################
# Plain regression without PCA
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
print(f"R2 score for SVM without PCA: {r2}")


########################################################
# Regression with PCA
model_pca = PCA(n_components=N_PCA)
model_pca.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

X_train_pca = model_pca.transform(X_train)
X_test_pca = model_pca.transform(X_test)

print(f"Training SVR (number of PCA components: {N_PCA})...")
model_svr = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    gamma=SVR_GAMMA
)

model_svr.fit(X_train_pca, y_train)
print("trained!")

print("Predicting with SVR...")
y_test_pred = model_svr.predict(X_test_pca)

r2 = r2_score(y_test, y_test_pred)
print(f"R2 score for SVM with PCA (n_comp={N_PCA}): {r2}")



plot_y = model_pca.explained_variance_ratio_
plot_x = range(len(plot_y))
plt.title("Explained variance ratio of each principal component")
plt.xlabel("Principal component")
plt.bar(plot_x, plot_y)
plt.show()



