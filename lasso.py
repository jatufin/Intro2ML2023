import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

RANDOM_SEED = 42

train_data = pd.read_csv('train.csv')

train_data['pSat_Pa_log10'] = np.log10(train_data['pSat_Pa']) #log_10 transformation
y = train_data['pSat_Pa_log10']

X = train_data.drop(['pSat_Pa', 'pSat_Pa_log10','Id'], axis=1)

# Fill the NA values with the mode
mode = X['parentspecies'].mode().iloc[0]
X['parentspecies'] = X['parentspecies'].fillna(mode)

# Label encoder for the categorial variable
encoder = LabelEncoder()
X['parentspecies'] = encoder.fit_transform(X['parentspecies'])

# Feature scaling for the numerical features only (cruical for the SVR in particular)
parentspecies = X['parentspecies']
X.drop("parentspecies", axis=1)
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
X['parentspecies'] = parentspecies

## Regressor

# Polynomial degrees
N = 3

for n in range(1,N+1):
    features = PolynomialFeatures(degree=n, interaction_only=True, include_bias=False)
    X_poly = features.fit_transform(X)
    feature_names = features.get_feature_names_out()

    #model = LinearRegression()
    model = Lasso()
    model.fit(X_poly, y)

    print(f"POLYNOMIAL: {n}")
    print(f"    Non-zero LASSO coefficients (total: {len(feature_names)})")

    for i in range(len(feature_names)):
        coef = model.coef_[i]
        if  coef != 0:
            print(f"        {i:4} {feature_names[i]:20} = {coef:.4f}")


