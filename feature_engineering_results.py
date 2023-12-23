import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Results from:
# selected_features.py
# pca.py

df = pd.DataFrame({
    'all features (25)': [0.78047184, 0.72575102, 0.63982184, 0.74425028, 0.73205254, 0.64731357, 0.68739672, 0.71181153, 0.75414576, 0.72117559],
    'feature reduction (19)': [0.76405281, 0.71947577, 0.62028288, 0.73629226, 0.7156401, 0.64183068, 0.68875946, 0.71053148, 0.74660047, 0.71636969],
    'LASSO (18)': [0.77043208, 0.70616803, 0.60902101, 0.72338666, 0.7164575, 0.6280325, 0.6817502, 0.70392669, 0.74398637, 0.70570995],
    'PCA (16)': [0.77292034, 0.71530254, 0.61872856, 0.73272891, 0.70959616, 0.62907664, 0.67918397, 0.67606884, 0.72361419, 0.70383028],
    'PCA (18)': [0.77819526, 0.71949894, 0.63348564, 0.73624252, 0.71650156, 0.63676838, 0.68239944, 0.68280755, 0.72970341, 0.70950674]
})

plt.title("10-fold cross validation R2 scores")
plt.ylabel("R2 score")
plt.xlabel("Dimension reduction method")
df.boxplot(
    whis=0,             # Do not plot whiskers
    showcaps=False,     # Do not polt the caps in the end of whiskers
    showfliers=False    # Do not plot oumliers
)
plt.show()

#########################
# Results from: selected_features.py
# SVR hyperparameters: kernel=rbf C=1.0 gamma=scale
# ====================================
# all features
# Cross validating SVR (CV=2)...
# Cross validation scores:
# (test_env)  ~/Documents/kurssit/introductionToML/project/Intro2ML2023 $ /opt/conda/envs/test_env/bin/python /home/jatuja/Documents/kurssit/introductionToML/project/Intro2ML2023/selected_features.py
# SVR hyperparameters: kernel=rbf C=1.0 gamma=scale
# ====================================
# all features
# Cross validating SVR (CV=10)...
# Cross validation scores:
# 'all features': [0.78047184 0.72575102 0.63982184 0.74425028 0.73205254 0.64731357
#  0.68739672 0.71181153 0.75414576 0.72117559]
# Mean R2 score for SVM with all columns: 0.7144190702517861
# ====================================
# feature reduction
# Selected columns:
# ['aldehyde', 'aromatic.hydroxyl', 'C.C.C.O.in.non.aromatic.ring', 'carbonylperoxyacid', 'carbonylperoxynitrate', 'carboxylic.acid', 'hydroperoxide', 'hydroxyl..alkyl.', 'ketone', 'nitro', 'nitroester', 'NumHBondDonors', 'NumOfAtoms', 'NumOfC', 'NumOfConf', 'NumOfConfUsed', 'NumOfN', 'parentspecies', 'peroxide', 'pSat_Pa']
# Cross validating SVR (CV=10)...
# Cross validation scores:
# 'feature reduction': [0.76405281 0.71947577 0.62028288 0.73629226 0.7156401  0.64183068
#  0.68875946 0.71053148 0.74660047 0.71636969]
# Mean R2 score for SVM with selected columns (feature reduction): 0.7059835603642994
# ====================================
# LASSO
# Selected columns:
# ['aldehyde', 'aromatic.hydroxyl', 'C.C..non.aromatic.', 'carbonylperoxyacid', 'carbonylperoxynitrate', 'carboxylic.acid', 'ether..alicyclic.', 'hydroperoxide', 'ketone', 'MW', 'nitroester', 'NumHBondDonors', 'NumOfC', 'NumOfConf', 'parentspecies_apin', 'peroxide']
# Cross validating SVR (CV=10)...
# Cross validation scores:
# 'LASSO': [0.77043208 0.70616803 0.60902101 0.72338666 0.7164575  0.6280325
#  0.6817502  0.70392669 0.74398637 0.70570995]
# Mean R2 score for SVM with selected columns (LASSO): 0.698887100043793


# PCA Results
# SVR hyperparameters: kernel=rbf C=1.0 gamma=scale
# Cross validating SVR with PCA (number of PCA components: 16, CV=10)...
# Cross validation scores:
# 'PCA-16': [0.77292034 0.71530254 0.61872856 0.73272891 0.70959616 0.62907664
#  0.67918397 0.67606884 0.72361419 0.70383028]
# [0.77292034 0.71530254 0.61872856 0.73272891 0.70959616 0.62907664
#  0.67918397 0.67606884 0.72361419 0.70383028]
# Mean R2 score for SVR with PCA: 0.696105042976832

# SVR hyperparameters: kernel=rbf C=1.0 gamma=scale
# Cross validating SVR with PCA (number of PCA components: 18, CV=10)...
# Cross validation scores:
# 'PCA-18': [0.77819526 0.71949894 0.63348564 0.73624252 0.71650156 0.63676838
#  0.68239944 0.68280755 0.72970341 0.70950674]
# [0.77819526 0.71949894 0.63348564 0.73624252 0.71650156 0.63676838
#  0.68239944 0.68280755 0.72970341 0.70950674]
# Mean R2 score for SVR with PCA: 0.7025109446477119