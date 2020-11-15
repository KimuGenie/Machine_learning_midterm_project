import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.utils import resample

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('dataset.csv'),header=None)
y = df.iloc[:, 11].values
X = df.iloc[:, [2, 3, 4, 5]].values #SBS에서 가장 정확도가 높았던 네 개의 feature
X_org = X #upsampling 하지 않을 원본 데이터
y_org = y

# class 1과 0의 비율을 1:1로 upsampling함. 총 9040개의 데이터를 사용함.
X_upsampled, y_upsampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=1)
X = np.vstack((X[y==0], X_upsampled))
y = np.hstack((y[y==0], y_upsampled))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X_org, y_org, test_size=0.3, random_state=1, stratify=y_org)
################################################################################

from sklearn.svm import SVC
sv = SVC(C=100, gamma=10, random_state=1)

from sklearn.pipeline import Pipeline
pipeSV = Pipeline([['sc', StandardScaler()], ['clf', sv]])

pipeSV.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print("train/test accuracy: %0.4f/%0.4f" %(accuracy_score(y_train_org, pipeSV.predict(X_train_org)), accuracy_score(y_test_org, pipeSV.predict(X_test_org))))
print("ROCAUC Score: %0.4f" % (roc_auc_score(y_test_org, pipeSV.predict(X_test_org))))