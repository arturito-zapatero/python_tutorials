# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:56:57 2018

@author: aszewczyk
"""

#get the data
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd
import numpy as np

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()


#create custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


#TransformerMixin combines fit() and transform() methods to get fit_transfrom()
#BaseEstimator gets methods get_params() and set_params() method (we need to avoid **args and **kargs)


#This transform can be implemented in pandas pipeline, will add to base d.f. two new columns (Room per household  
#and population per household) - using transfrom(). fit() will do nothing
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix] #adds room per household columns to input data
        population_per_household = X[:, population_ix] / X[:, household_ix] #adds population per household
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]   #if  add_bedrooms_per_room = True adds one extra column
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs2 = attr_adder.transform(housing.values)
housing_extra_attribs1= attr_adder.fit(housing.values)  #do nothing!


attr_adder.get_params()
attr_adder.set_params(add_bedrooms_per_room=2)
attr_adder.get_params()






from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(pd.DataFrame([1, 2, 3]))  
enc.n_values_
enc.feature_indices_
enc.transform(pd.DataFrame([1, 2, 3])).toarray()

enc.fit_transform(pd.DataFrame([1, 2, 3])).toarray()





