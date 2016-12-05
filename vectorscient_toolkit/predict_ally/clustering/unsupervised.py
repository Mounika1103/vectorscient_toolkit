"""
Records clustering module.

So far implements K-Means clustering and applies PCA before plotting. Generated
plots can be displayed using standard matplotlib interface or saved into PNG
format.
"""
from collections import OrderedDict
from datetime import datetime, date
from operator import itemgetter
from time import time
from enum import Enum
import logging
import io

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import prettytable
import attrdict

from ...exceptions.exc import InvalidDataError, DataProcessingError
from ..mixins import ConfigParsingMixin, DataPreprocessingMixin
from ..reporting import ArchiveContext, PDFContext
from ..preprocessing import MissingValuesFiller
from ..preprocessing import CentroidCalculator
from ..preprocessing import ColumnNamesFilter
from ..preprocessing import NonNumericFilter
from ..preprocessing import WeightMultiplier
from ..technique import PredictAllyTechnique
from ...sources.info import ClusteringStats
from ..preprocessing import ConstantFilter
from ..preprocessing import Standardizer
from ...sources.data import DataSource
from ...utils import normalize, method_dispatch


__all__ = ['Opportunity', 'UnsupervisedClustering']


log = logging.getLogger(__name__)
log.setLevel(level=logging.DEBUG)


class Opportunity(Enum):
    """
    The enumeration of possible records classes.
    """
    IDEAL = -1
    EXTREMELY_LIKELY = 0
    LIKELY = 1
    SWING = 2
    UNLIKELY = 3
    EXTREMELY_UNLIKELY = 4

    @classmethod
    def to_verbose(cls, op):
        return op.name.lower().replace("_", " ").title()


class ClusteringResult:

    def __init__(self, error: bool = False, **clustering):
        self._error = error
        self._centroids = clustering.get("centroids")
        self._clusters = clustering.get("clusters")
        self._data = clustering.get("data")
        self._raw_data = clustering.get("raw_data")
        self._message = clustering.get("message", "Success")

    @property
    def failed(self):
        return self._error

    @property
    def status(self):
        return self._message

    @property
    def centroids(self):
        return self._centroids

    @property
    def clusters(self):
        return self._clusters

    @property
    def data(self):
        return self._data

    @property
    def raw_data(self):
        return self._raw_data


class UnsupervisedClustering(PredictAllyTechnique, DataPreprocessingMixin,
                             ConfigParsingMixin):
    """
    Supplementary class that deals with preliminary data processing and
    processed data clustering and plotting.
    """

    def __init__(self, **params):
        super().__init__()
        self._config = None
        self._raw_data = pd.DataFrame()
        self._normalized_data = pd.DataFrame()
        self._prepared_data = pd.DataFrame()
        self._ordered_centroids = pd.DataFrame()
        self._ideal_centroid = None
        self._estimator = None
        self._fig = None
        self._subtitle = None
        self._benchmark_results = []
        self._raw_label_to_opportunity = {}

        self.configure(**params)

    def configure(self, **params):
        self._config = self.parse_config(params)
        self._subtitle = self._config.get("subtitle", "")

    def describe(self):
        raw_rows, raw_cols = self._raw_data.shape
        predictors = list(self._prepared_data.columns)
        attributes = set(self._raw_data.columns) - set(predictors)
        return pd.Series({
            "records": raw_rows,
            "columns": raw_cols,
            "predictors": predictors,
            "attributes": attributes
        })

    @property
    def normalized_data(self):
        return self._normalized_data

    @property
    def centroids(self):
        return self._ordered_centroids

    @property
    def estimator(self):
        return self._estimator

    def run(self, source):
        self._setup_preprocessors()

        if isinstance(source, DataSource):
            if not source.ready:
                source.prepare()
            self._raw_data = source.data

        elif isinstance(source, pd.DataFrame):
            self._raw_data = source

        else:
            err = "Unexpected data source format: '{}'".format(type(source))
            log.error(err)
            raise ValueError(err)

        log.debug("Preparing data for clustering")
        data = pd.DataFrame(self._raw_data.copy())
        prep_data = self.apply_preprocessing(data, keep_intermediate=True)

        if prep_data.isnull().values.any():
            err = "Data still contains NaN values after preprocessing"
            log.error(err)
            raise InvalidDataError(err)

        results = self._intermediate_results
        centroid_was_calculated = results.get("Centroids") is not None
        ideal_centroid = prep_data.iloc[-1] if centroid_was_calculated else []

        self._prepared_data = prep_data

        if self._config.apply_benchmark:
            log.debug("Benchmarking")
            self.benchmark()

        self._ideal_centroid = ideal_centroid
        self._normalized_data = results.get("Normalization")
        self._estimator = self._apply_clustering(prep_data)
        self._ordered_centroids = \
            self._order_centroids_by_distance_to_ideal_centroid()

        log.debug("Plotting calculated clusters and centroids")
        # self._plot_pca_clusters(prep_data)
        self._plot_relative_clusters(prep_data)

    def predicted_classes(self) -> pd.DataFrame:
        """
        Returns names and numeric values of clusters that records belongs to.
        """
        ideal_centroid = self._ideal_centroid
        mapping = self._create_class_to_opportunity_mapping(
            ideal_centroid=ideal_centroid,
            estimator=self._estimator)
        raw_labels = [int(label) for label in self._estimator.labels_[:-1]]
        ops = [mapping[l] for l in raw_labels]
        table = pd.DataFrame({
            'cluster_class': [o.value for o in ops],
            'cluster_class_name': [o.name for o in ops]
        })
        table.index = self._raw_data.index
        return table

    def benchmark(self):
        """
        Benchmarking clustering technique implementation.
        """
        params = self._config.clustering.parameters
        params["init"] = "k-means++"
        self._bench_clustering_method(KMeans(**params), self._prepared_data)
        params["init"] = "random"
        self._bench_clustering_method(KMeans(**params), self._prepared_data)
        self._benchmark_report()

    def _setup_preprocessors(self):
        cfg = self._config

        if not cfg.preprocessing.apply:
            return

        steps = cfg.preprocessing.steps

        if steps.drop_constants.apply:
            self.register_preprocessor(ConstantFilter(), name="IgnoreConstants")

        if steps.drop_columns.apply:
            if len(steps.drop_columns.names) > 0:
                prep = ColumnNamesFilter(*steps.drop_columns.names)
                self.register_preprocessor(prep, name="IgnoreColumns")

        if steps.drop_non_numeric.apply:
            self.register_preprocessor(NonNumericFilter(), name="IgnoreFactors")

        if steps.missing_values_input.apply:
            prep = MissingValuesFiller(**steps.missing_values_input)
            self.register_preprocessor(prep, name="MissingInput")

        if steps.ideal_centroids_calculation.apply:
            prep = CentroidCalculator(
                **steps.ideal_centroids_calculation)
            self.register_preprocessor(prep, name="Centroids")

        if steps.standartization.apply:
            prep = Standardizer(**steps.standartization)
            self.register_preprocessor(prep, name="Standardization")

        if steps.normalization.apply:
            lo, hi = steps.normalization.interval
            prep = lambda data: normalize(data, low=lo, high=hi)
            self.register_preprocessor(prep, name="Normalization")

        if steps.weights.apply:
            prep = WeightMultiplier(**steps.weights)
            self.register_preprocessor(prep, name="Weights")

        log.debug("Configuration was done")

    def _bench_clustering_method(self, estimator, data):
        """
        Tests classifier performance and records results into benchmark log.

        Args:
            estimator (sklearn.cluster.BaseEstimator): A classifier to be tested.
            data (pd.DataFrame): A data to be used for classifier testing.
        """
        start_time = time()
        estimator.fit(data)
        elapsed = time() - start_time
        metric_scores = [
            ('%9s', estimator.init),
            ('%.2fs', elapsed),
            ('%i', estimator.inertia_)
        ]
        result = [fmt % value for fmt, value in metric_scores]
        self._benchmark_results.append(result)

    def _benchmark_report(self):
        """
        Prints collected benchmark results.
        """
        columns = ['init', 'elapsed time', 'inertia']
        table = prettytable.PrettyTable(field_names=columns)
        for result in self._benchmark_results:
            table.add_row(result)

        log.info("Clustering algorithm benchmark scores "
                 "for '{}':".format(self._subtitle))
        log.info(table)

    def _apply_clustering(self, prepared_data):
        X = StandardScaler().fit_transform(prepared_data)
        estimator = KMeans(**dict(self._config.clustering.parameters))
        estimator.fit(X)
        return estimator

    def _order_centroids_by_distance_to_ideal_centroid(self):
        """
        Orders centroids by Euclidean distance from ideal centroid and
        assigns appropriate class names for each of them.

        Returns:
            pandas.DataFrame: A data frame with ordered centroids extended with
                metadata and opportunity classes.
        """
        centers = self._estimator.cluster_centers_
        ideal = self._ideal_centroid
        all_centers = centers.tolist() + [ideal.tolist()]

        distances = [distance.euclidean(ideal, c) for c in all_centers]
        weight_vectors = zip(distances, all_centers)
        sorted_vectors = sorted(weight_vectors, key=itemgetter(0))
        new_columns = ['record_id', 'cluster_class',
                       'cluster_class_name', 'pred_run_data']
        clustering_columns = self._prepared_data.columns.tolist()

        record_id, rows = 1, []
        for index, (_, center) in enumerate(sorted_vectors, start=-1):
            cluster = Opportunity(index)
            row = dict(zip(
                new_columns,
                [record_id, cluster.value, cluster.name, date.today()]
            ))
            centroid_values = dict(zip(clustering_columns, center))
            row.update(centroid_values)

            rows.append(row)
            record_id += 1

        centroids_table = pd.DataFrame(rows)
        columns = new_columns + clustering_columns
        df_with_reordered_columns = centroids_table[columns]
        return df_with_reordered_columns

    def _plot_relative_clusters(self, prepared_data):
        """
        Creates clustering plot using distances to ideal and cluster centroids
        as coordinates
        """
        cfg = self._config
        model = KMeans(**cfg.clustering.parameters)
        labels = model.fit_predict(prepared_data)
        centers = model.cluster_centers_
        label_to_opportunity = self._create_class_to_opportunity_mapping(
            ideal_centroid=self._ideal_centroid,
            estimator=model)

        # calculate distances to local and ideal
        rows = [row.tolist() for _, row in prepared_data.iterrows()]
        cluster_distances = np.array(
            [distance.euclidean(centers[l], r)
             for l, r in zip(labels, rows)]).reshape(-1, 1)
        ideal_distances = np.array(
            [distance.euclidean(self._ideal_centroid, r)
             for r in rows]).reshape(-1, 1)

        min_max_scaler = MinMaxScaler()
        df = prepared_data.copy()
        df["x"] = min_max_scaler.fit_transform(cluster_distances).ravel()
        df["y"] = min_max_scaler.fit_transform(ideal_distances).ravel()
        df["labels"] = labels
        df["opportunity"] = [Opportunity.to_verbose(label_to_opportunity[l])
                             for l in labels]

        # plotting
        def reorder_legend(ax, legend_order):
            sort = list(sorted(
                [(l, lbl) for l, lbl in zip(*ax.get_legend_handles_labels())],
                key=lambda pair: legend_order[pair[1]]
            ))
            return zip(*sort)

        plot_params = self._config.clusters_plotting
        plt.ioff()
        self._fig = plt.figure(**plot_params.matplotlib_figure_parameters)
        self._fig.suptitle("Prospective Opportunity Classification")
        ax = self._fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")

        cluster_parameters = OrderedDict()
        plotting_config = [
            {"x_offset": 4.0, "y_offset": 4.0, "color": "darkgreen"},
            {"x_offset": 1.0, "y_offset": 4.0, "color": "green"},
            {"x_offset": 2.5, "y_offset": 2.5, "color": "gold"},
            {"x_offset": 4.0, "y_offset": 1.0, "color": "salmon"},
            {"x_offset": 1.0, "y_offset": 1.0, "color": "red"}
        ]
        ops = [op for op in Opportunity if op != Opportunity.IDEAL]
        for op, conf in zip(ops, plotting_config):
            key = Opportunity.to_verbose(op)
            cluster_parameters[key] = conf

        stats = [("Total Opportunities", len(df), "darkorange", "heavy")]
        for op, params in cluster_parameters.items():
            subset = df[df.opportunity == op].copy()
            stats.append((op, len(subset), "black", "normal"))
            dx = params["x_offset"]
            dy = params["y_offset"]
            subset.x += dx
            subset.y += dy
            colors = [params["color"]] * len(subset)

            if subset.empty:
                logging.warning(
                    "Dataset is empty for opportunity '{}'".format(op))
                continue

            subset.plot.scatter(x="x", y="y",
                                c=colors, ax=ax,
                                s=150, alpha=.5, label=op)
            cx = params["x_offset"] + 0.5
            cy = params["y_offset"] + 0.5
            ax.scatter(cx, cy,
                       s=plot_params.centroids_size,
                       marker=plot_params.centroids_sign,
                       lw=plot_params.centroids_linewidth,
                       edgecolor=plot_params.centroids_edgecolor,
                       facecolor=plot_params.centroids_facecolor)

        ax.grid("on")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(self._subtitle)

        legend_order = dict(
            [(v, k) for k, v in enumerate(cluster_parameters.keys())])
        lines, labels = reorder_legend(ax, legend_order)
        ax.legend(lines, labels,
                  title="Customer Conversion Likelihood",
                  loc="upper left", bbox_to_anchor=(0.0, -0.05))

        y, step = -0.085, -0.04
        for stat_name, count, color, weight in stats:
            text = stat_name.ljust(30) + str(count)
            ax.text(x=0.4, y=y, s=text,
                    color=color,
                    fontweight=weight,
                    fontsize=12,
                    family='monospace',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes)
            y += step

        if plot_params.hide_axis:
            ax.set_axis_off()

    def _plot_pca_clusters(self, prepared_data):
        """
        Applies dimension reduction procedure and plots clusters using
        principal components.
        """
        data, estimator = self._reduce_data_and_train_classifier(prepared_data)
        plot_params = self._config.clusters_plotting
        plt.ioff()

        self._fig = plt.figure(**plot_params.matplotlib_figure_parameters)
        self._fig.suptitle(plot_params.title)
        ax = self._fig.add_subplot(111)
        ax.set_title(plot_params.title)
        ax.set_title(self._subtitle)
        ax.axvline(x=0, color='gray', label=None)
        ax.axhline(y=0, color='gray', label=None)

        ideal_centroid = data.iloc[-1]
        label_to_opportunity = self._create_class_to_opportunity_mapping(
            ideal_centroid,
            estimator)

        color_map = get_cmap(plot_params.color_map)
        data["label"] = estimator.labels_
        cc = estimator.cluster_centers_
        legend_order = dict()

        for center_point, (label, group) in zip(cc, data.groupby("label")):
            opportunity = label_to_opportunity[int(label)]
            color = color_map(int(opportunity.value) * 10)
            colors = [color] * len(group)
            verbose_name = opportunity.name.replace("_", " ").lower().title()
            legend_order[verbose_name] = opportunity.value
            group.plot.scatter(x="x", y="y", c=colors, s=150,
                               label=verbose_name, alpha=0.5, ax=ax)
            x, y = center_point
            ax.scatter(x, y,
                       s=plot_params.centroids_size,
                       marker=plot_params.centroids_sign,
                       lw=plot_params.centroids_linewidth,
                       edgecolor=plot_params.centroids_edgecolor,
                       facecolor=plot_params.centroids_facecolor)

        ax.grid("on")
        ax.set_xlabel("Principal Component #1")
        ax.set_ylabel("Principal Component #2")

        def reorder_legend():
            sort = list(sorted(
                [(l, lbl) for l, lbl in zip(*ax.get_legend_handles_labels())],
                key=lambda pair: legend_order[pair[1]]))
            return zip(*sort)

        lines, labels = reorder_legend()
        ax.legend(lines, labels,
                  title="Customer Conversion Likelihood",
                  loc="upper center", bbox_to_anchor=(0.5, -0.1))

        if plot_params.hide_axis:
            ax.set_axis_off()

    def _reduce_data_and_train_classifier(self, prepared_data):
        """
        Reduces data dimension for further plotting.
        """
        cfg = self._config
        pca = PCA(**cfg.clusters_plotting.pca_parameters)
        reduced_data = pca.fit_transform(prepared_data)
        estimator = KMeans(**cfg.clustering.parameters)
        estimator.fit(reduced_data)
        df = pd.DataFrame(reduced_data, columns=['x', 'y'])
        return df, estimator

    @staticmethod
    def _create_class_to_opportunity_mapping(ideal_centroid, estimator):
        centers = estimator.cluster_centers_
        distances = {i: distance.euclidean(ideal_centroid, c)
                     for i, c in enumerate(centers)}
        opportunities = [Opportunity(i) for i, _ in enumerate(centers)]
        label_to_opportunity = dict(zip(
            [i for i, _ in sorted(distances.items(), key=itemgetter(1))],
            opportunities
        ))
        return label_to_opportunity

    def model(self) -> dict:
        return {"centroids": self._ordered_centroids,
                "classes": self.predicted_classes()}

    def report(self, context):
        buffer = io.BytesIO()
        self._fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        if isinstance(context, ArchiveContext):
            file_name = self._subtitle + '.png'
            context.add_to_report((file_name, buffer))

        elif isinstance(context, PDFContext):
            context.add_to_report(buffer)


class OpportunityClassifier(PredictAllyTechnique, ConfigParsingMixin):
    """
    Assigns an opportunity class for each predictors row.
    """

    def __init__(self, **config):
        self._config = attrdict.AttrDict(config)
        self._grouping = config.get("grouping", True)
        self._classifiers = []
        self._model = None

    def run(self, source):
        data = pd.DataFrame(source.data.copy())

        if data.empty:
            log.warning("Provided data source returned empty dataset")
            log.warning("Terminating clustering process")
            return

        if "id" not in data.columns:
            data["id"] = list(range(1, len(data) + 1))

        run_date = datetime.now().date().strftime("%Y-%m-%d")
        log.info("Clustering run date: " + str(run_date))

        overall_instance = self.overall_clustering(data)
        overall_instance.centroids["pred_run_date"] = run_date
        self._classifiers.append(overall_instance)

        if not self._grouping:
            self._model = ClusteringResult(
                centroids=overall_instance.centroids,
                clusters=overall_instance.predicted_classes(),
                data=None)
            return

        stage_instances = self.stages_clustering(data)
        self._classifiers.extend(stage_instances)
        overall_classes = overall_instance.predicted_classes()
        stage_classes = pd.concat(
            [uc.predicted_classes() for uc in stage_instances])
        clusters = pd.merge(
            overall_classes, stage_classes,
            left_index=True, right_index=True,
            suffixes=('', '_by_stage'))
        clusters_with_row_ids = pd.merge(
            data[['id']], clusters,
            left_index=True, right_index=True)
        clusters_with_row_ids["pred_run_date"] = run_date

        # normalized data frame
        data_with_row_ids = pd.merge(
            data[['id']], overall_instance.normalized_data,
            left_index=True, right_index=True)
        dropped_columns = [c for c in source.data.columns
                           if c not in data_with_row_ids.columns]
        backup = data[dropped_columns].copy()
        final_normalized_data = pd.DataFrame(
            pd.concat([data_with_row_ids, backup], axis=1))
        final_normalized_data["pred_run_date"] = run_date
        final_normalized_data["cluster_class"] = clusters["cluster_class"]

        self._model = ClusteringResult(
            centroids=overall_instance.centroids,
            clusters=clusters_with_row_ids,
            data=final_normalized_data)

    def configure(self, **params):
        self._config = self.parse_config(params)

    def overall_clustering(self, data):
        config = self._config
        config["subtitle"] = "Overall Dataset"
        uc = UnsupervisedClustering(**config)
        uc.run(data)
        return uc

    def stages_clustering(self, data):
        clustering_instances = []
        config = self._config

        for key, group in data.groupby(config.grouping_column):
            df = self._try_to_prepare_stages_group_for_clustering(key, group)

            if df.empty:
                continue

            subtitle = "{}={}".format(config.grouping_column, key)
            config["subtitle"] = subtitle
            uc = UnsupervisedClustering(**config)
            try:
                uc.run(df)

            except DataProcessingError as e:
                msg = ("Data error occurred while processing grouping key "
                       "'{}'. Probably there is no enough data in column. "
                       "This clustering group will be skipped.").format(key)
                log.warning(msg)

            else:
                clustering_instances.append(uc)

        return clustering_instances

    def _try_to_prepare_stages_group_for_clustering(self, key, data):
        """
        Prepares stage filtered data to be clustered by dropping irrelevant
        and grouping columns.
        """
        config = self._config

        try:
            prefix = config.sales_stage_prefix

            if prefix is not None:
                stage_cols = [c for c in data.columns if c.startswith(prefix)]
                not_relevant_stages = [
                    c for c in stage_cols if
                    int(c.replace(prefix, "")) > int(key)]
                drop_cols = not_relevant_stages + [config.grouping_column]

            else:
                drop_cols = [config.grouping_column]

            only_relevant_data = data.drop(drop_cols, axis=1)

        except InvalidDataError as e:
            log.error("Stage '{}' bad data error: {}".format(key, e))
            return pd.DataFrame()

        else:
            return only_relevant_data

    @method_dispatch
    def report(self, context):
        """
        Base method used for single dispatch pattern implementation to handle
        different types of reporting contexts.

        Args:
            context: An instance of reporting context
        """
        raise NotImplementedError()

    @report.register(ArchiveContext)
    def _(self, context):
        """
        Saves prediction results into archive.

        Args:
            context: An archive context.
        """
        for classifier in self._classifiers:
            classifier.report(context)

        dfs = [self._model.raw_data, self._model.clusters]
        columns = []
        for df in dfs:
            if df is None or df.empty:
                continue
            columns.extend(df.columns.tolist())
        result = pd.concat(dfs, axis=1, ignore_index=True)
        result.columns = columns
        context.add_to_report(('clusters.csv', result))

    @report.register(PDFContext)
    def _(self, context):

        if self._grouping:
            # stages groups clustering
            classifier = self._classifiers[0]
            db_name = self._config["records"]["database"]
            stats = ClusteringStats(database=db_name)
            input_info = classifier.describe()

            info = pd.Series()
            info["Number of Records"] = stats.number_of_records()
            info["Number of Columns"] = stats.number_of_columns()
            info["Number of Predictors"] = len(input_info["predictors"])
            info["Number of Attributes"] = len(input_info["attributes"])
            info["Number of Matched Web Records with CRM"] = \
                stats.number_of_matched_records()
            info["Number of Unmatched Web Records with CRM"] = \
                stats.number_of_unmatched_records()
            info["Number of Likely Matched Web Records with CRM"] = \
                stats.number_of_likely_matched_records()
            info.name = "This section contains the information that " \
                        "you have provided"
            context.add_to_report(info)

            context.break_page()
            predictors = input_info["predictors"]
            empty_strings = ['' for _ in enumerate(predictors)]
            ser = pd.Series(empty_strings, index=predictors)
            ser.name = "List of Predictors Found"
            context.add_to_report(ser)

            context.break_page_with_header("Predictors and Weights")
            weights_table = stats.predictors_and_weights()
            context.add_to_report(weights_table)

            context.break_page_with_header("Time Decay Factors")
            decay_factors_table = stats.time_decay_factors()
            context.add_to_report(
                decay_factors_table,
                column_widths=[200, 50, 50, 50, 50],
                ignore_index=True)

            context.break_page_with_header("Web Activity Segments")
            web_activity_segments = stats.web_activity_segments()
            context.add_to_report(web_activity_segments)

            context.break_page()
            classifier.report(context)

        else:
            # overall data clustering only
            classifier = self._classifiers[0]
            input_info = classifier.describe()

            info = pd.Series()
            info["Number of Records"] = input_info["records"]
            info["Number of Columns"] = input_info["columns"]
            info["Number of Predictors"] = len(input_info["predictors"])
            info["Number of Attributes"] = len(input_info["attributes"])
            info.name = "This section contains the information that " \
                        "you have provided"
            context.add_to_report(info)

            context.break_page()

            predictors = input_info["predictors"]
            empty_strings = ['' for _ in enumerate(predictors)]
            ser = pd.Series(empty_strings, index=predictors)
            ser.name = "List of Predictors Found"
            context.add_to_report(ser)

            context.break_page()

            classifier.report(context)

            context.break_page()

            def total(cs, category):
                return sum(1 if c == category else 0 for c in cs)

            classes = classifier.predicted_classes()["cluster_class"]
            results = pd.Series()
            results["Total Records Processed"] = len(classes)
            results["Most Likely to Convert"] = total(classes, 0)
            results["Somewhat Likely to Convert"] = total(classes, 1)
            results["Swing Customers"] = total(classes, 2)
            results["Somewhat Unlikely to Convert"] = total(classes, 3)
            results["Most Unlikely to Convert"] = total(classes, 4)
            results.name = "Classification Result - Summary"
            context.add_to_report(results)

            context.add_to_report(
                "The downloadable CSV file has classification "
                "result for every record")

    def model(self):
        return self._model
