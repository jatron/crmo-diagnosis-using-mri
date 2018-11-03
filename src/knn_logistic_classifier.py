"""
================================
k-nearest neighbors & logistic regression classifier
================================

"""
print(__doc__)

from sklearn import datasets, neighbors, linear_model

# TODO: Replace X and y with actual feature vectors and labels (using "digits" dataset temporarily)
digits = datasets.load_digits()
X = digits.data / digits.data.max()
y = digits.target

n_samples = len(X)

X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                           multi_class='multinomial')

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
