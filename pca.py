import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.decomposition import PCA

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

model = PCA()
model.fit_transform(X_df)

print(model.explained_variance_ratio_)

plot_y = model.explained_variance_ratio_
plot_x = range(len(plot_y))
plt.title("Explained variance ratio of each principal component")
plt.xlabel("Principal component")
plt.bar(plot_x, plot_y)
plt.show()

