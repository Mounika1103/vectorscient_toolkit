Predict Ally Engine
===================

The data analysing engine implementing opportunities clustering and reporting results. 
The main modules and packages are:

+ `engine` - Composes data sources, data processing and predicting algorithms together and launches prediction pipeline in accordance with provided configuration.
+ `mixins` - Several mixin classes implementing commonly needed functionality.
+ `preprocessing` - Custom data preprocessors invoked before data analysis process.
+ `reporting` - Reporting contexts, i.e. output formats (PDF, ZIP-archive, etc.).
+ `technique` - Abstract base classes for data preprocessing and ML algorithms.
+ `shortcuts` - A couple of functions that simplify the engine invocation.
 

Opportunity Values Clustering
-----------------------------
Instances of the ``predict_ally.clusters.UnsupervisedClustering`` class are
used to perform unsupervised records clustering using K-means algorithm
(`sklearn` implementation). The clustering algorithm is tuned via 
configuration file stored somewhere in file system. The ``predict_ally.clusters.OpportunityClassifier``
is built on top of the ``UnsupervisedClustering`` instances and basically
creates several of them and composes their outputs together.

The following table describes config keys meaning. Note that **bold** text
mean that the exactly same string should be used in config, whereas *italic*
text specify data type that should be used. Names enumerated within 
curly braces define set of allowed values.

Configuration files use YAML language which formal description can be found 
on its [official site](http://yaml.org/). Also an example of configuration file
can be found in `tests/settings` folder of this repository.

| Parameter        | Possible Values                    | Description        |
| ---------------- | ---------------------------------- | ------------------ |
| Common parameters ||
| `grouping_column`  | *string* | Dataset column that is used for grouping |
| `sales_stage_prefix` | *string* | Columns that should be dropped when clusterinsation is applied to filetered data |
| `apply_benchmark` | {**Yes**, **No**} | If **Yes**, then benchmark should be applied before clustering |
| `attribute_columns` | list of *string* | YAML list enumerating columns that do not take part in clustering |
| Data source configuration|||
| `source_type` | {**csv**, *db*} | The data source type (local file, database, etc.) |
| `source_file` |*string*| If local file selected as data source, then path to this file should be specified |
| `reader_config` | dict of *strings* | If local file selected as data source, YAML dictionary with `sources.CsvDataSource` reading parameters |
| Columns filter parameters |||
| `names` | list of *strings* | YAML list with columns names to be ignored in clustering process |
| Weights multiplier parameters |||
| `feature_name_column` | *string* | Column in the lookup table containing predictors names |
| `weights_column` | *string* | Column in the lookup table containing predictors weights |
| Normalization parameters |||
| `interval` | list of *int* | YAML list with two values defining normalization range |
| Clustering parameters |||
| `algorithm` | {**KMeans**} | Applied clustering algorithm |
| `parameters` | dictionary of *objects* | YAML dictionary with parameters for clustering algorithm tuning. Use [scikit-learn clustering reference](http://scikit-learn.org/stable/modules/clustering.html) to find out possible values | 
| Clusters plotting parameters |||
| `pca_parameters` | dictionary of *objects*| YAML dictionary with PCA parameters. Use [scikit-learn decomposition reference](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to find out possible values
| `matplotlib_figure_parameters`| dictionary of *objects*| YAML dictionary with `matplotlib.Figure` configuration |
| `mesh_step` | *float* | Granularity of the values grid used to plot clusters background |
| `color_map` | *string* | The color map used to plot clusters |
| `centroids_color` | *string* | The color used to plot centroids signs |
| `centroids_size` | *int* | The size of marker used to denote centroids |
| `centroids_sign` | *string* | The symbol used as a centroid marker |
| `show_centroid_values` | {**Yes**, **No**} | Show values near centroid markers or not |
| `hide_axis` | {**Yes**, **No**} | Hide plot axis or not |
| `display_mode`| {**screen**, **file**} | Show create plot on screen or save into file |
| `file_name` | *string* | If `display_mode` equals to **file** then specifies path to generated image |
| `title` | *string* | The string to be used as a main plot title |
| Generic parameters with same meaning for all the configuration sections |||
| `method` | *string* | Data processing method to be used |
| `apply` | {**Yes**, **No**} | If **Yes**, then specific data processing step should be applied |


Prediction Pipeline Invocation
------------------------------
The engine instance can be created and configured manually inside of backend code. However, there is no robust and well documented API so far.
That's why a couple of preconfigured shortcuts created to keep things clear. Examples of usage can be found inside of [test suites](../vectorscient_toolkit/functional_tests/test_engine.py).

Briefly, there are two clustering modes:

1. for a registered user (with configuration);
2. for a trial user (without configuration).

To invoke clustering pipeline for registered user, the following snippet can be used:
```python
from vectorscient_toolkit.predict_ally import shortcuts

path_to_config = some_function_to_retrieve_user_configuration_path()
shortcuts.run_opportunity_clustering_for_registered_user(
    path_to_config, output="path/to/output/file", context="pdf")     
```

Using aforementioned code, the clustering result will be saved into PDF file with path `path/to/output/file.pdf`.


PredictAlly Quality Index Calculation
-------------------------------------
```
from qfi.PredictAlly_quality_index_calc import QFI
qfi = QFI(database='clientdb')

# Create table
qfi.create_tables()

#Calculation quality index
qfi.process(input_cat='exist', pred_run_date='2016-01-08')
```

Import csv data to QFI tables
```
from qfi.csv2data import CsvImportMixin
csv2data = CsvImportMixin(database='clientdb')
csv2data.save_imports('<table name>', '<filepath>')
```


