import pandas as pd
import numpy as np


def assert_equal_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Asserts that two dataframes have same structure and values.

    Args:
        df1 (pandas.DataFrame): first dataframe
        df2 (pandas.DataFrame): second dataframe
    """
    assert df1.columns.tolist() == df2.columns.tolist(), "Columns don't match"
    assert len(df1) == len(df2), "Number of rows doesn't match"
    assert df1.values.tolist() == df2.values.tolist(), "Content doesn't match"


def assert_almost_equal_sequences(s1, s2):
    assert np.allclose(s1, s2), "Sequences differ too much"
