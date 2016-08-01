from glob import glob
from mne import read_epochs
from mne.decoding import EpochsVectorizer
from multiprocessing import cpu_count
from operator import itemgetter
from scipy.stats import lognorm
from scipy.stats import randint
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from time import time
import numpy as np
import pandas as pd


# Utility function to report best scores (borrowed from
# http://scikit-learn.org/stable/_downloads/randomized_search.py)

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


random_state = np.random.RandomState(42)
n_cores = cpu_count()

df = pd.read_hdf('data/misc/evokeds.h5', key='evokeds')

X = df[df.columns[1:15]]
y = df['response']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=random_state)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDClassifier(average=True,
                    class_weight=None,
                    fit_intercept=False,
                    learning_rate='optimal',
                    loss='hinge',
                    n_jobs=n_cores,
                    penalty='l2',
                    random_state=random_state,
                    verbose=0)

param_dist = {'alpha': lognorm(s=2, scale=np.exp(-4)),
              'n_iter': randint(5, 26),
              'shuffle': [True, False]}

# run randomized search

n_iter_search = 20
random_search = RandomizedSearchCV(sgd,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
