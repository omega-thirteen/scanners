from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.decomposition import FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


random_state = np.random.RandomState(42)

scaler = StandardScaler()
ica = FastICA()
sgd = SGDClassifier()

estimators = [('scaler', StandardScaler()),
              ('ica', FastICA()),
              ('sgd', SGDClassifier())]

pipe = Pipeline(estimators)

# When loss='hinge', SGDClassifier becomes a linear SVM.
# When loss='log', SGDClassifier becomes a logistic regression.

# clf = SGDClassifier(alpha=0.0001,
#                     average=False,
#                     class_weight=None,
#                     epsilon=0.1,
#                     eta0=0.0,
#                     fit_intercept=True,
#                     l1_ratio=0.15,
#                     learning_rate='optimal',
#                     loss='hinge',
#                     n_iter=1,
#                     n_jobs=2,
#                     penalty='l2',
#                     power_t=0.5,
#                     random_state=random_state,
#                     shuffle=True,
#                     verbose=0,
#                     warm_start=False)