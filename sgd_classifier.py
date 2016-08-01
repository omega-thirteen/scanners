from multiprocessing import cpu_count
from sklearn.decomposition import FastICA
from scipy.stats import lognorm
from scipy.stats import randint
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


random_state = np.random.RandomState(42)
n_cores = cpu_count()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=random_state)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
pipe = Pipeline(estimators)
sgd = SGDClassifier(average=True,
                    class_weight=None,
                    fit_intercept=False,
                    learning_rate='optimal',
                    loss='hinge',
                    n_jobs=n_cores,
                    penalty='l2',
                    random_state=random_state,
                    verbose=0)
#                     fit_intercept=True,
param_dist = {'alpha': lognorm(s=2, scale=np.exp(-4)),
              'n_iter': randint(5, 26),
              'shuffle': [True, False]}

# run randomized search

n_iter_search = 20
random_search = RandomizedSearchCV(sgd,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search)
#                     warm_start=False)
