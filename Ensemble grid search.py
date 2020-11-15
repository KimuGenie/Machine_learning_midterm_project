from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import resample

############################## Data import ##############################
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
################################################################################

#################### Majority Vote Classifier ##########################
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self,X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' %(name, key)] = value
            return out
#################################################################


###ensemble classifier###
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=1)

from sklearn.svm import SVC
sv = SVC(C=100, gamma=10, random_state=1)

from sklearn.pipeline import Pipeline
pipeKNN = Pipeline([['sc', StandardScaler()], ['clf', knn]])
pipeSV = Pipeline([['sc', StandardScaler()], ['clf', sv]])

esb = MajorityVoteClassifier(classifiers=[pipeKNN, tree, pipeSV])

# print(esb.get_params().keys())

clf_labels = ['KNN', 'Decision tree', 'SVM', 'Ensemble']
all_clf = [knn, tree, sv, esb]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy')
    print("accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))

##### Grid Search #####
from sklearn.model_selection import GridSearchCV

params = {'pipeline-1__clf__n_neighbors':[1, 3], 'decisiontreeclassifier__max_depth': [4, 5, 6], 'pipeline-2__clf__C': [10, 100, 1000], 'pipeline-2__clf__gamma': [1, 10, 100]}
grid = GridSearchCV(estimator = esb, param_grid=params, cv=5, scoring='accuracy', iid=False, n_jobs=-1)
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.3f %r" % (grid.cv_results_['mean_test_score'][r],
    grid.cv_results_['std_test_score'][r]/2.0,
    grid.cv_results_['params'][r]))

print('best: %s' % grid.best_params_)
print('best score: %0.3f' % grid.best_score_)