import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
df = pd.read_csv(os.path.abspath('python_code/dataset.csv'),header=None)
y = df.iloc[:960, 11].values #class 1과 0의 비율을 1:1로 하기 위해 960개의 데이터만 가져옴
X = df.iloc[:960, [2,6]].values #SBS에서 가장 정확도가 높았던 두 개의 feature

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', penalty='l2', C=1, random_state=1)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)

from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(max_iter=70, eta0=0.01, tol=1e-3, random_state=1)

from sklearn.pipeline import Pipeline
pipeKNN = Pipeline([['sc', StandardScaler()], ['clf', knn]])
pipelr = Pipeline([['sc', StandardScaler()], ['clf', lr]])
pipeSGD = Pipeline([['sc', StandardScaler()], ['clf', SGD]])

clf_labels = ['KNN', 'Logistic Regression', 'Decision tree', 'SGD']
all_clf = [pipeKNN, pipelr, tree, pipeSGD]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    clf.fit(X_train, y_train)
    print("train/test accuracy: %0.3f/%0.3f" %(accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))))

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

###decision region###
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product

x_min = X_train_std[:, 0].min() -1
x_max = X_train_std[:, 0].max() +1
y_min = X_train_std[:, 1].min() -1
y_max = X_train_std[:, 1].max() +1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7,5))
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha = 0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0], X_train_std[y_train == 0, 1], c='blue', marker='^', s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], c='green', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)
    plt.text(-3.5, -3.5, s = 'Feature 3 [standardized]', ha='center', va='center', fontsize=12)
    plt.text(-11.5, 6.5, s = 'Feature 7 [standardized]', ha='center', va='center', fontsize=12, rotation=90)
plt.show()