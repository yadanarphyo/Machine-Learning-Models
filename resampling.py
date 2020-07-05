from sklearn.datasets import make_classification
from collections import Counter

#For illustration I manually create a training dataset
X, y = make_classification(n_samples=420, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.1405, 0.8595],
                           class_sep=0.8, random_state=0)

# print the number of two categories
Counter(y)
# Ouput: Counter({1: 357, 0: 63})
# now ratio is like 6:1 


# This is resampling part
# The idea is randomly picking up from the smaller sample group, 
# which is called over-sampling.

from imblearn.over_sampling import RandomOverSampler
#define the model
ros = RandomOverSampler(random_state=0)
#resample command
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())

# print the number of two categories
Counter({1: 357, 0: 63})
 # Output: [(0, 357), (1, 357)]
 # now we get half positive/ half negative 