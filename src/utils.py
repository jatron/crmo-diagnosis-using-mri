import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn import model_selection

def generate_validation_curve(estimator, X, y, param_name, param_range, cv,
    scoring, n_jobs, title, xlabel):

  train_scores, test_scores = model_selection.validation_curve(
    estimator, X, y, param_name=param_name, param_range=param_range,
    cv=cv, scoring=scoring, n_jobs=n_jobs)

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel("Accuracy")
  plt.ylim(0.0, 1.1)
  lw = 2
  plt.semilogx(param_range, train_scores_mean, label="Training accuracy",
               color="darkorange", lw=lw)
  plt.fill_between(param_range, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.2,
                   color="darkorange", lw=lw)
  plt.semilogx(param_range, test_scores_mean, label="Cross-validation accuracy",
               color="navy", lw=lw)
  plt.fill_between(param_range, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.2,
                   color="navy", lw=lw)
  plt.legend(loc="best")
  plt.show()
