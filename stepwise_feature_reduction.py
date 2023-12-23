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

# Feature reduction target
MIN_FEATURES=10

# Cross validation
CV=5

# Dataset
N = 5000 # Use a random sample of N rows from the data set. None ==> all
TRAIN_DATA_FILE = "train.csv"
#TEST_DATA_FILE = "test.csv"

def score_with_columns(
        data,
        features=None,
        min_features=20,
        target_column=TARGET_COLUMN,
        results=[],
        score_type=None
    ):
    if features is None:
        columns = data.columns
        features = columns[columns != TARGET_COLUMN]
        features = pd.Index([f for f in features if f not in DROP_COLUMNS])

    if len(features) <= min_features:
        return results
    
    best_score = -10000
    best_feature_set = None

    for feature in features:
        selected_features = features[features != feature]

        df = data[selected_features].copy()
        df[TARGET_COLUMN] = data[TARGET_COLUMN]
    
        X, y, columns, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS)
        
        # one run validation
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

        # cross validation
        X_train = X.copy()
        y_train = y.copy()

        model = SVR(kernel=SVR_KERNEL, C=SVR_C, gamma=SVR_GAMMA)
        ### one run score
        #model.fit(X_train, y_train)
        #y_pred = model.predict(X_test)
        #r2 = r2_score(y_test, y_pred)

        ### cross validation score
        cv_results = cross_validate(model, X, y, cv=CV)
        scores = cv_results["test_score"]
        r2 = scores.mean()

        if score_type == "r2":
            score = r2
        if score_type == "adjusted_r2":
            n, p = X_train.shape
            adjusted_r2 = 1-(1-r2)*((n-1)/(n-p-1))
            score = adjusted_r2
        
        if score > best_score:
            best_score = score
            best_feature_set = selected_features
    
    print(f"Score: {score_type} Features: {len(features)} Best score: {best_score}")
          
    return score_with_columns(
        data,
        features=best_feature_set,
        min_features=min_features,
        results=results + [(best_score, list(best_feature_set))],
        score_type=score_type
    )

data = pd.read_csv(TRAIN_DATA_FILE)

# For testing with a smaller data subset
if N is not None:
    data = data.sample(n=N)

results_r2 = score_with_columns(data, min_features=MIN_FEATURES, score_type="r2")


print("R2 scores")
scores = []
feature_num = []

max_score = 0
max_features = None
for score, features in results_r2:
    if score > max_score:
        max_score = score
        max_features = features    
    scores.append(score)
    feature_num.append(len(features))

plt.plot(feature_num, scores)
plt.title("Maximum R2 scores", fontsize=18)
plt.ylabel("R2", fontsize=14)
plt.xlabel("number of features", fontsize=14)
plt.show()

print("========================")
print(f"Maximum R2 score: {max_score}")
print(f"Features: ({len(max_features)})")
print("selected columns = [")
for f in max_features:
    print(f"    '{f}',")
print("]")
print("========================")
print("")

