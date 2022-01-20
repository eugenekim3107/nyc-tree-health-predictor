import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# load data from files
os.chdir('/Users/eugenekim')
tree = pd.read_csv('Downloads/new_york_tree_census_2015.csv')


# extract the necessary columns from the data set
def column_extracter(data):
    important_col = []
    for i in [0, 3, 5, 7, 9, 10, 11, 12, 13, 14, 29, 37, 38]:
        important_col = important_col + [data.columns[i]]
    return data.loc[:, important_col]


tree = column_extracter(tree)

# choose the desired location in New York (Bronx for this project)
tree = tree[tree['boroname'] == 'Bronx'].drop(columns=['boroname'])
tree = tree.set_index('tree_id')

# stratify and split the data using the user_type category
X_train, X_test, y_train, y_test = train_test_split(tree, tree['health'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=tree['user_type'])


# custom transformer to remove outliers using interquartile range
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X


outlier_remover = OutlierRemover()


# custom transformer to clump all categories into either zero or one
class BinaryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, problems=True):
        self.problems = problems

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[x == 'None'] = 0
            x[x != 0] = 1
            X.iloc[:, i] = x
        return X


binary_converter = BinaryConverter(problems=True)


# custom transformer to change health label to integers
class HealthNumerical(BaseEstimator, TransformerMixin):
    def __init__(self, health=True):
        self.health = health

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[x.isna()] = 0
            x[x == 'Poor'] = 1
            x[x == 'Fair'] = 2
            x[x == 'Good'] = 3
            X.iloc[:, i] = x
        return X


health_numerical = HealthNumerical(health=True)


# custom transformer to change steward category to numerical values
class StewardNumerical(BaseEstimator, TransformerMixin):
    def __init__(self, steward=True):
        self.steward = steward

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[x.isna()] = 0
            x[x == 'None'] = 0
            x[x == '1or2'] = 1
            x[x == '3or4'] = 2
            x[x == '4orMore'] = 3
            X.iloc[:, i] = x
        return X


steward_numerical = StewardNumerical()

# create numerical pipeline
num_pipeline = Pipeline([('outlier', outlier_remover),
                         ('imputer', SimpleImputer(strategy='median')),
                         ('std_scaler', StandardScaler())])

# create ordinalencoder pipeline
oe_pipeline = Pipeline(
    [('imputer', SimpleImputer(strategy='constant', fill_value='NoDamage')),
     ('ordinal', OrdinalEncoder(categories='auto')),
     ('std_scaler', StandardScaler())])

# create pipeline for steward feature
steward_pipeline = Pipeline([('steward', steward_numerical),
                             ('std_scaler', StandardScaler())])

# create pipeline for health label
health_pipeline = Pipeline([('health', health_numerical)])

# create pipeline for problem feature
prob_pipeline = Pipeline([('binary', binary_converter),
                          ('std_scaler', StandardScaler())])

# create onehotencoder pipeline
one_hot_pipeline = Pipeline(
    [('imputer', SimpleImputer(strategy='constant', fill_value='Unsure')),
     ('one_hot', OneHotEncoder())])

# remove the training label from the training features
X_train = X_train.drop(['health'], axis=1)
X_test = X_test.drop(['health'], axis=1)

# form full pipeline for the training data
num_attribs = ['tree_dbh', 'latitude', 'longitude']
oe_attribs = ['curb_loc', 'sidewalk']
steward_attribs = ['steward']
prob_attribs = ['problems']
one_hot_attribs = ['guards', 'user_type']
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs),
                                   ('oe', oe_pipeline, oe_attribs),
                                   ('steward', steward_pipeline,
                                    steward_attribs),
                                   ('prob', prob_pipeline, prob_attribs),
                                   ('one', one_hot_pipeline, one_hot_attribs)])
train_features = full_pipeline.fit_transform(X_train)
train_label = health_pipeline.fit_transform(y_train)['health'].astype('int64')

# save data to files
np.savetxt('train_features.txt', train_features)
np.savetxt('train_label.txt', train_label)

# separate test data
test_features = full_pipeline.fit_transform(X_test)
test_label = health_pipeline.fit_transform(y_test)['health'].astype('int64')

# convert and save test set
np.savetxt('test_features.txt', test_features)
np.savetxt('test_label.txt', test_label)
