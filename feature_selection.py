import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

#RANDOM_SEED = 42
RANDOM_SEED = None

N = 5000 # Use a random sample of N rows from the data set
TRAIN_DATA_FILE = "train.csv"
#TEST_DATA_FILE = "test.csv"

DROP_COLUMNS = ["Id"]
TARGET_COLUMN = "pSat_Pa"
CATEGORICAL_FEATURES = ['parentspecies']

# SVR HYPERPARAMETERS
SVR_KERNEL="rbf"
SVR_C=0.3
SVR_GAMMA=0.1

# Feature reduction target
MIN_FEATURES=2

def preprocess(df,
               categorical_features = CATEGORICAL_FEATURES,
               numerical_features=None,
               one_hot=None,
               scaler=None,
               target_column=None,
               drop_columns=DROP_COLUMNS
            ):
    """ Preprocesses the data and returns feature matrix X and target vector y

        Parameters:
            df (DataFrame)
            categorical_features (list)    
            target_column (string)
        
        Returns:
            X, y, numerical_features, one_hot, scaler
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
    else:
        X_encoded = []

    # Feature scaling for the numerical features only (cruical for the SVR in particular)
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

    return X_encoded, y, numerical_features, one_hot, scaler

def score_with_columns(data, features=None, min_features=20, target_column=TARGET_COLUMN, results=[], score_type=None):
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
    
        X, y, numerical_features, one_hot, scaler = preprocess(df, target_column=TARGET_COLUMN)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

        model = SVR(kernel=SVR_KERNEL, C=SVR_C, gamma=SVR_GAMMA)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
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
results_adjusted_r2 = score_with_columns(data, min_features=MIN_FEATURES, score_type="adjusted_r2")

fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")

fig.suptitle("The highest test scores with number of selected features", fontsize=20)

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

ax1.plot(feature_num, scores)
ax1.set_title("Maximum R2 scores", fontsize=18)
ax1.set_ylabel("R2", fontsize=14)
ax1.set_xlabel("number of features", fontsize=14)

print("========================")
print(f"Maximum R2 score: {max_score}")
print(f"Features: ({len(max_features)})")
for f in max_features:
    print(f)
print("========================")
print("")

print("Adjusted R2 scores")
scores = []
feature_num = []

max_score = 0
max_features = None
for score, features in results_adjusted_r2:
    if score > max_score:
        max_score = score
        max_features = features
    scores.append(score)
    feature_num.append(len(features))

ax2.plot(feature_num, scores)
ax2.set_title("Maximum adjusted R2 scores", fontsize=18)
ax2.set_ylabel("adjusted R2", fontsize=14)
ax2.set_xlabel("number of features", fontsize=14)

print("========================")
print(f"Maximum adjusted R2 score: {max_score}")
print(f"Features: ({len(max_features)})")
for f in max_features:
    print(f)
print("========================")
print("RESULTS R2")
print(results_r2)
print("========================")
print("RESULTS ADJUSTED R2")
print(results_adjusted_r2)
print("========================")
plt.show()


