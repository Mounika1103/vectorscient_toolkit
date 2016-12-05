from functools import partial
from operator import mul

from sklearn.preprocessing import MinMaxScaler, Imputer
from attrdict import AttrDict
import pandas as pd
import numpy as np

from ..sources.data import CsvDataSource, DatabaseTableSource, DBConnection
from ..exceptions.exc import ConfigurationError
from .technique import DataPreprocessor


class ConstantFilter(DataPreprocessor):
    """
    Drops all dataset columns filled with constant values, i.e. NaNs,
    zeros, etc.

    For example, let's define a data frame:
    >>> c1 = [1, 2, 3, 4, 5]
    >>> c2 = [0, 0, 0, 0, 0]
    >>> c3 = [1, None, 3, None, None]
    >>> c4 = [None, None, None, None, None]
    >>> cols = c1, c2, c3, c4
    >>> d = {'c' + str(i): c for i, c in enumerate(cols, 1)}
    >>> data = pd.DataFrame(d)

    Next, create filter and apply to the data frame:
    >>> prep = ConstantFilter()
    >>> cleaned = prep.apply(data)
    >>> cleaned
       c1  c3
    0   1   1
    1   2 NaN
    2   3   3
    3   4 NaN
    4   5 NaN
    """

    def apply(self, data, *args, **kwargs):
        columns_to_drop = []
        for name in list(data.columns):
            column = set(data[name].dropna())
            if len(column) <= 1:
                columns_to_drop.append(name)
        cleaned = data.drop(columns_to_drop, axis=1)
        return cleaned


class NonNumericFilter(DataPreprocessor):
    """
    Drops all non-numeric (and factor) attributes, i.e. string names,
    addresses, etc.
    """

    def apply(self, data, *args, **kwargs):
        numeric_only = data.select_dtypes(include=[np.number])
        return numeric_only


class ColumnNamesFilter(DataPreprocessor):
    """
    Drops columns with specific names.
    """

    def __init__(self, *names):
        self._names = names

    def apply(self, data, *args, **kwargs):
        data_columns = list(data.columns)
        cleaned = data
        for name in self._names:
            if name not in data_columns:
                continue
            cleaned = cleaned.drop(name, axis=1)
        return cleaned


class Standardizer(DataPreprocessor):
    """
    Applies selected standardization algorithm to the data set.
    """

    def __init__(self, **kwargs):
        self._method = kwargs.get("method")

    def apply(self, data, *args, **kwargs):
        if self._method == "min-max":
            scaler = MinMaxScaler()
            X = data.astype(float)
            df = pd.DataFrame(scaler.fit_transform(X))
            df.columns = list(X.columns)
            return df

        else:
            err = "Unexpected standardization method: '{}'".format(self._method)
            raise ConfigurationError(err)


class CentroidCalculator(DataPreprocessor):
    """
    Calculates a centroid for provided dataset and appends it as a
    last row to it.
    """

    def __init__(self, **kwargs):
        self._method = kwargs.get("method")

    def apply(self, data, *args, **kwargs):
        if self._method == 'mean':
            centroid = data.mean()

        elif self._method == 'median':
            centroid = data.median()

        else:
            err = "Unexpected ideal centroid calculation " \
                  "method: '{}'".format(self._method)
            raise ConfigurationError(err)

        data = data.append(centroid, ignore_index=True)
        return data


class MissingValuesFiller(DataPreprocessor):
    """
    Fills NaN and missed values using selected algorithm.

    For example, let's create a data frame:
    >>> data = pd.DataFrame({"x": [1, 2, 3], "y": [None, 30, None]})

    To fill NaN values with zeros use:
    >>> MissingValuesFiller(method='zeros').apply(data)
       x   y
    0  1   0
    1  2  30
    2  3   0

    Or, to fill with mean value:
    >>> MissingValuesFiller(method='mean').apply(data)
       x   y
    0  1  30
    1  2  30
    2  3  30
    """

    def __init__(self, **kwargs):
        self._method = kwargs.get("method")

    def apply(self, data, *args, **kwargs):

        if self._method == "zeros":
            cleaned_up = data.fillna(0)
            cleaned_up.dropna(inplace=True)
            return cleaned_up

        elif self._method in ("mean", "median"):
            imputer = Imputer(strategy=self._method)
            cleaned_up = imputer.fit_transform(data)
            df = pd.DataFrame(cleaned_up)
            df.columns = list(data.columns)
            return df

        else:
            err = "Unexpected missing values filling " \
                  "method: '{}'".format(self._method)
            raise ConfigurationError(err)


class WeightMultiplier(DataPreprocessor):
    """
    Multiplies data by weights vector.

    The weights can be stored in CSV file or database table.

    Params:
        source_type (str): A type of source where weights are stored.
        lookup_table (str): A name of table or file name with weights.
    """
    def __init__(self, **params):
        cfg = AttrDict(**params)

        if cfg.source_type == "csv":
            weights_source = CsvDataSource(
                file_name=cfg.lookup_table,
                reader_config=cfg.reader_config
            )
            weights_source.prepare()
            df = weights_source.data
            weights_lookup = df[[cfg.weights_column, cfg.feature_name_column]]

        elif cfg.source_type == "db":
            conn = DBConnection(database=cfg.database)
            weights_source = DatabaseTableSource(
                table_name=cfg.lookup_table,
                conn=conn
            )
            weights_source.prepare()
            df = weights_source.data
            weights_lookup = df[[cfg.weights_column, cfg.feature_name_column]]

        else:
            err = "Weights sources other then CSV or DB table are not supported"
            raise ConfigurationError(err)

        self._weights_lookup = weights_lookup

    def apply(self, data, *args, **kwargs):
        weights_lookup = self._weights_lookup
        for index, weight, feature_name in weights_lookup.itertuples():
            if feature_name not in data:
                continue
            weight_multiplier = partial(mul, weight)
            weighted = data[feature_name].apply(weight_multiplier)
            data[feature_name] = pd.Series(weighted)
        return data
