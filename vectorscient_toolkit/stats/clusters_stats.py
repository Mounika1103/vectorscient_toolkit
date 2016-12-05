from enum import Enum

import pandas as pd

from ..exceptions.exc import DataSourceError, StatsError
from ..sources.data import DataSource


__all__ = ["calculate_cluster_stats"]


HEADER_LEVELS = 1


class GroupColumnsProcessing(Enum):
    Undefined = 1
    Ignore = 2
    Include = 3


class FullStatsCalculator:
    """
    Calculates full stats on provided data source.

    Resultant dataframe contains following statistics for each data source:
    Min, Max, Mean, Median, Standard Deviation, Variance and Quartile Deviation.
    """

    ORIGINAL_COLUMNS = (
        "min", "max", "mean", "median", "std", "var", "qd")

    VERBOSE_COLUMNS = (
        "Min", "Max", "Mean", "Median", "Standard Deviation",
        "Variance", "Quartile Deviation")

    def __init__(self, source: DataSource, **params):
        """
        Validates passed parameters and creates instance of stats calculator.

        Args:
            source (DataSource): data source with clustered entries

        Parameters:
            ignore (List[str]): column groups that should be ignored; cannot be
                specified if include list is already specified
            include (List[str]): the only column groups that should be
                processed cannot be specified if ignore list is already
                specified
        """
        params["source"] = source
        self._source = None
        self._ignore = None
        self._include = None
        self._predictors = False
        self._processing_mode = GroupColumnsProcessing.Undefined
        self._columns = None
        self._parse_and_validate_arguments(**params)

    def _parse_and_validate_arguments(self, **params):
        ignore = params.get("ignore", None)
        include = params.get("include", None)
        source = params["source"]

        if ignore is not None and include is not None:
            raise ValueError("one of 'ignore' and 'include' "
                             "arguments should be None")

        if not isinstance(source, DataSource):
            raise TypeError("can only process DataSource instances")

        if not source.ready:
            msg = "cannot calculate stats for not prepared source"
            raise DataSourceError(msg)

        if source.data.columns.nlevels != HEADER_LEVELS:
            msg = ("cannot process data frame: columns grouping names "
                   "were not found (probably source file has only one "
                   "level of headers)")
            raise StatsError(msg)

        if ignore is not None:
            self._columns = ignore
            self._processing_mode = GroupColumnsProcessing.Ignore
        else:
            self._columns = include
            self._processing_mode = GroupColumnsProcessing.Include

        self._source = source
        self._ignore = ignore
        self._include = include
        self._predictors = params.get("predictors", False)

    def calculate_group_stats(self):
        """
        Calculates descriptive stats for each data source.
        """
        data = self._source.data
        stats = self._calculate_stats(data)
        prepared_stats = pd.DataFrame(stats[list(self.ORIGINAL_COLUMNS)])
        return self._rename_columns(prepared_stats)

    def _calculate_stats(self, df):
        all_column_wise_sums = []
        needed_groups_only = self._filtered_groups(df)

        for group in needed_groups_only:
            cols_from_group = df[group]
            column_wise_sum = cols_from_group.sum(axis=1)
            column_wise_sum.name = group
            all_column_wise_sums.append(column_wise_sum)

        df_to_calculate_stats = \
            pd.DataFrame(pd.concat(all_column_wise_sums, axis=1))
        stats = self._calculate_descriptive_stats(df_to_calculate_stats)

        return stats

    def calculate_predictor_stats(self):
        """
        Calculates descriptive stats for each predictor.
        """
        data = self._source.data
        stats = self._calculate_stats_per_predictor(data)
        prepared_stats = pd.DataFrame(stats[list(self.ORIGINAL_COLUMNS)])
        return self._rename_columns(prepared_stats)

    def _calculate_stats_per_predictor(self, df):
        needed_groups_only = self._filtered_groups(df)
        all_predictors = [df[group] for group in needed_groups_only]

        df_to_calculate_stats = pd.DataFrame(pd.concat(all_predictors, axis=1))
        stats = self._calculate_descriptive_stats(df_to_calculate_stats)

        return stats

    def _filtered_groups(self, df):
        column_groups_names = list(df.columns.levels[0])
        named_columns_only = (name for name in column_groups_names
                              if not name.lower().startswith("unnamed"))
        columns_filter = self._get_columns_filter()
        yield from filter(columns_filter, named_columns_only)

    def _get_columns_filter(self):
        if self._processing_mode == GroupColumnsProcessing.Ignore:
            return lambda x: x not in self._columns
        elif self._processing_mode == GroupColumnsProcessing.Include:
            return lambda x: x in self._columns
        assert False, "cannot get there"

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        names_mapping = dict(zip(self.ORIGINAL_COLUMNS, self.VERBOSE_COLUMNS))
        return df.rename(columns=names_mapping)

    @staticmethod
    def _calculate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
        stats = df.describe().transpose()
        stats["var"] = df.var()
        stats["median"] = df.median()
        stats["qd"] = (stats["75%"] - stats["25%"])/2
        return stats
