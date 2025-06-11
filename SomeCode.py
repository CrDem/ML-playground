import numpy as np
import pandas as pd

# 0
ones_matrix = np.ones((5, 5))
ones_submatrix_view = ones_matrix[::2, ::2]
ones_submatrix_view2 = ones_matrix[1::2, 1::2]
print(ones_matrix)
print(ones_submatrix_view)
print(ones_submatrix_view2)
ones_submatrix_view[:] = np.zeros((3, 3))
ones_submatrix_view2[:] = np.zeros((2, 2))
print(ones_matrix)

# 1
def construct_matrix(first_array, second_array):
    return np.column_stack([first_array, second_array])

print(construct_matrix([1,2], [3,4]))

# 2
def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    elMax = 0
    maxSeen = 0
    d = {}
    for el in nums:
        if el in d:
            d[el] += 1
        else:
            d[el] = 1
        if d[el] > maxSeen:
            maxSeen = d[el]
            elMax = el
    return elMax

# 3
import pandas as pd

base = 'tables/'

data = pd.read_csv(base + 'organisations.csv')
features = pd.read_csv(base + 'features.csv')
rubrics = pd.read_csv(base + 'rubrics.csv')

d_rubrics = {}
d_features = {}
d_average_bill = {}

for index, row in rubrics.iterrows():
    d_rubrics[row['rubric_id']] = row['rubric_name']

for index, row in features.iterrows():
    d_features[row['feature_id']] = row['feature_name']

print(data.head())
print(d_features)
print(d_rubrics)

clean_data = data[~(data['average_bill'].isna()) & ~(data['average_bill'] > 2500)]

print(data.shape[0])
print(clean_data.shape[0])

# 4.0
import pandas as pd
import statistics

base = 'tables/'

dirty_data = pd.read_csv(base + 'organisations.csv')
data = dirty_data[~(dirty_data['average_bill'].isna()) & ~(dirty_data['average_bill'] > 2500)]

data_msk = data[data['city'] == 'msk']
data_spb = data[data['city'] == 'spb']

diff = abs(statistics.mean(data_msk['average_bill']) - statistics.mean(data_spb['average_bill']))
print(int(diff))
# 4.1
def has_cafe_rubtic(rubrics):
    return '30774' in rubrics.split()

isCafe = dirty_data['rubrics_id'].apply(has_cafe_rubtic)

data = dirty_data[~(dirty_data['average_bill'].isna()) & ~(dirty_data['average_bill'] > 2500) & (isCafe)]

average_bill_by_city = data.groupby('city')['average_bill'].mean()

difference = average_bill_by_city['msk'] - average_bill_by_city['spb']

print(difference)

# 5-6   
from scipy.stats import mode

from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

base = '/home/'

dirty_data = pd.read_csv(base + 'organisations.csv')
data = dirty_data[~(dirty_data['average_bill'].isna()) & ~(dirty_data['average_bill'] > 2500)]

data_train, data_test = train_test_split(
    data, stratify=data['average_bill'], test_size=0.33, random_state=42)

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        if y is None:
            self.mean_ = None
        else:
            self.mean_ = y.mean()

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        if X is not None:
             return np.full(X.shape[0], self.mean_)
        elif self.mean_ is not None:
             return self.mean_
        else:
             return None

class MostFrequentClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        if y is None:
            self.mean_ = None
        else:
            res = mode(y, keepdims=True)
            self.mean_ = res.mode[0]

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        if X is not None:
             return np.full(X.shape[0], self.mean_)
        elif self.mean_ is not None:
             return self.mean_
        else:
             return None

reg = MeanRegressor()
reg.fit(y=data_train['average_bill'])
clf = MostFrequentClassifier()
clf.fit(y=data_train['average_bill'])

actual_values = data_test['average_bill']

predictions1 = reg.predict(X=data_test) 
mse1 = mean_squared_error(actual_values, predictions1)

predictions2 = clf.predict(X=data_test) 
mse2 = mean_squared_error(actual_values, predictions2)

print(mse1)
print(mse2)
print(reg.mean_)
print(clf.mean_)

# 7
class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        if y is None:
            self.means_ = None
            self.global_mean_ = None
        else:
            self.global_mean_ = y.mean()
            df = pd.DataFrame(X.copy())
            df['target'] = y
            self.means_ = df.groupby('city')['target'].mean()

    def predict(self, X=None):
        if X is None:
             return self.global_mean_
        elif self.means_ is not None:
             return X['city'].map(self.means_).fillna(self.global_mean_).values
        else:
             return None

reg = CityMeanRegressor()
reg.fit(X=data_train, y=data_train['average_bill'])

actual_values = data_test['average_bill']

predictions = reg.predict(X=data_test) 
mse = mean_squared_error(actual_values, predictions)

print(mse)
print(reg.global_mean_)
print(reg.means_)

# 8
