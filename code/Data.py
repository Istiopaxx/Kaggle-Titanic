import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
import csv
import sklearn as skl
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.model_selection import train_test_split
from scipy import sparse


TITANIC_PATH = "datasets"


def fetch_titanic_data(titanic_path=TITANIC_PATH):
    if not os.path.isdir(titanic_path):
        os.makedirs(titanic_path)


def load_titanic_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, "train.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    np.random.seed(97)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def split_data_label(data):
    """return dataframe, nparray"""
    titanic_data = data.drop("Survived", axis=1)
    titanic_label = data["Survived"].values
    return titanic_data, titanic_label


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, type='categorical'):
        self.attribute_names = attribute_names
        self.type = type
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if(self.type == 'categorical'):
            return X[self.attribute_names].values.reshape(-1, 1)
        if(self.type == 'numerical'):
            return X[self.attribute_names].values


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

def full_data_pipeline(raw_data):
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=97)
    train_data, train_label = split_data_label(train_set)
    test_data, test_label = split_data_label(test_set)
    """data : DataFrame,  label : numpy array"""
    titanic_num_data = train_data.drop(["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    num_attribs = list(titanic_num_data)
    cat_attribs1 = ['Sex']
    cat_attribs2 = ['Pclass']
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs, type='numerical')),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline1 = Pipeline([
        ('selector', DataFrameSelector(cat_attribs1)),
        ('cat_encoder', CategoricalEncoder(encoding='onehot-dense')),
    ])
    cat_pipeline2 = Pipeline([
        ('selector', DataFrameSelector(cat_attribs2)),
        ('cat_encoder', CategoricalEncoder(encoding='onehot-dense')),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline1", cat_pipeline1),
        ("cat_pipeline2", cat_pipeline2),
    ])
    titanic_train_prepared = full_pipeline.fit_transform(train_data)
    titanic_test_prepared = full_pipeline.fit_transform(test_data)
    return titanic_train_prepared, train_label, titanic_test_prepared, test_label

