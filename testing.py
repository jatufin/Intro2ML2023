import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/train.csv")

#df.drop(["Id", "pSat_Pa"], axis=1).hist(bins=100)
#np.log10(df["pSat_Pa"]).hist(bins=100)

df.isna().any(axis=1)
#nor = np.random.normal(0,1,10000)
#norex = pd.Series(np.power(10,nor))

#np.log10(norex).hist(bins=100)

plt.show()