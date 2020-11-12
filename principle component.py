import pandas as pd
import os
df = pd.read_csv(os.path.abspath('python_code/dataset.csv'),header=None)
y = df.iloc[:960, 11].values #class 1과 0의 비율을 1:1로 하기 위해 960개의 데이터만 가져옴
X = df.iloc[:960, :11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

import numpy as np
cov_mat = np.cov(X_train_std.T) #nomalize된 X_train_set의 공분산
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1,12), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,12), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout
plt.show()