"""
================================
k-nearest neighbors & logistic regression classifier with cross validation
================================

"""
print(__doc__)

import numpy as np
from sklearn import datasets, neighbors, linear_model, model_selection

# TODO: Replace X and y with actual feature vectors and labels (using "digits" dataset temporarily)
digits = datasets.load_digits()
X = digits.data[:40] / digits.data.max()
y = digits.target[:40]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                           multi_class='multinomial')

loo = model_selection.LeaveOneOut()

knn_scores = model_selection.cross_val_score(knn, X, y, cv=loo, n_jobs=-1)
logistic_scores = model_selection.cross_val_score(logistic, X, y, cv=loo, n_jobs=-1)

print('KNN score: %f' % np.mean(knn_scores))
print('LogisticRegression score: %f' % np.mean(logistic_scores))
