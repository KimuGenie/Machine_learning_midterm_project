from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
import pandas as pd
import os
from sklearn.utils import resample

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('dataset.csv'),header=None)
y = df.iloc[:, 11].values
X = df.iloc[:, [2, 3, 4, 5]].values #SBS에서 가장 정확도가 높았던 네 개의 feature

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set

# train set을 upsampling해줌 총 6328개의 데이터를 훈련에 사용
X_upsampled, y_upsampled = resample(X_train[y_train == 1], y_train[y_train == 1], replace=True, n_samples=X_train[y_train == 0].shape[0], random_state=1)
X_train = np.vstack((X_train[y_train==0], X_upsampled))
y_train = np.hstack((y_train[y_train==0], y_upsampled))
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
knn = KNeighborsClassifier(n_neighbors=1)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=1)

from sklearn.svm import SVC
sv = SVC(C=10, gamma=100, random_state=1)

from sklearn.pipeline import Pipeline
pipeKNN = Pipeline([['sc', StandardScaler()], ['clf', knn]])
pipeSV = Pipeline([['sc', StandardScaler()], ['clf', sv]])

esb = MajorityVoteClassifier(classifiers=[pipeKNN, tree, pipeSV])

esb.fit(X_train, y_train)

y_test_pred = esb.predict(X_test)

##### score#####
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print("train/test accuracy: %0.4f/%0.4f" %(accuracy_score(y_train, esb.predict(X_train)), accuracy_score(y_test, y_test_pred)))
print("ROCAUC Score: %0.4f" % (roc_auc_score(y_test, y_test_pred)))

###### confusion matrix ######
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.tight_layout()
plt.show()