import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.utils import resample

df = pd.read_csv(os.path.abspath('dataset.csv'),header=None)
y = df.iloc[:, 11].values
X = df.iloc[:, :11].values
X = df.iloc[:, [2, 3, 4, 5]].values #SBS에서 가장 정확도가 높았던 네 개의 feature
# class 1과 0의 비율을 1:1로 upsampling함. 총 9040개의 데이터를 사용함.
X_upsampled, y_upsampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=1)
X = np.vstack((X[y==0], X_upsampled))
y = np.hstack((y[y==0], y_upsampled))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', penalty='l2', C=1, random_state=1)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)

from sklearn.svm import SVC
sv = SVC(C=100, gamma=10, random_state=1)

from sklearn.pipeline import Pipeline
pipeKNN = Pipeline([['sc', StandardScaler()], ['clf', knn]])
pipelr = Pipeline([['sc', StandardScaler()], ['clf', lr]])
pipeSV = Pipeline([['sc', StandardScaler()], ['clf', sv]])

# print(pipeKNN.get_params().keys())
# print(pipelr.get_params().keys())
# print(tree.get_params().keys())
# print(pipeSV.get_params().keys())

##### Grid Search #####
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 

#KNN Grid Search
param_range = [1, 3, 5, 7, 9]
params = {'clf__n_neighbors': param_range}
grid = GridSearchCV(estimator = pipeKNN, param_grid=params, cv=2, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print('[KNN]')
print('best: %s' % grid.best_params_)
scores = cross_val_score(grid, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/-%.3f' % (np.mean(scores), np.std(scores)))

#lr Grid Search
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
params = {'clf__C': param_range}
grid = GridSearchCV(estimator = pipelr, param_grid=params, cv=2, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print('[Logistic Regression]')
print('best: %s' % grid.best_params_)
scores = cross_val_score(grid, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/-%.3f' % (np.mean(scores), np.std(scores)))

#tree Grid Search
param_range = [1, 2, 3, 4, 5]
params = {'max_depth': param_range}
grid = GridSearchCV(estimator = tree, param_grid=params, cv=2, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print('[Decision tree]')
print('best: %s' % grid.best_params_)
scores = cross_val_score(grid, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/-%.3f' % (np.mean(scores), np.std(scores)))

#SV Grid Search
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
params = {'clf__C': param_range, 'clf__gamma': param_range}
grid = GridSearchCV(estimator = pipeSV, param_grid=params, cv=2, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print('[SVC]')
print('best: %s' % grid.best_params_)
scores = cross_val_score(grid, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/-%.3f' % (np.mean(scores), np.std(scores)))