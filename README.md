VectorScient Records Linkage Toolkit
====================================

Basically, project is separated into several subpackages:

|Package|Description|
|-------|-----------|
|`exceptions`| Custom exceptions raised inside of application packages. |
|`log`| Supplementary code to be used with builtin logging module (is not used too extensively). |
|`sources`| Thin wrappers over different kinds of data sources. All data sources use `pandas` library and return `pandas.DataFrame` as a result. |
|`stats`| Calculates different kinds of statistics for provided data source. Kind of proof-of-concept module. | 
|`phrasely`| Sentiment analysis module providing several (thought simple) approaches to estimate text sentiment score |
|`predict_ally`| The PredictAlly data analysis engine implementation. |
|`qfi`| A set of scripts to calculate QFI scores and save them into DB. |
|`resources`| Supplementary application resources (i.e. images, fonts). ||
|`tests`| Unit tests for application modules. |
|`functional_tests`| Functional (acceptance) tests to verify general correctness. Actually, some of these are not real tests (i.e. no assertions made) and just generate some output values to be verified manually. |

Also there are a couple of modules used here and there:

|Module|Description|
|------|-----------|
|`config`|Global parameters and paths|
|`utils`|A set of utility functions for data processing or doing some domain specific tasks|

Several parts of toolkit use instances of classes inherited from ``sources.DataSource`` 
abstract class. A few quite specific resources already implemented, e.g. ``CsvDataSource`` or ``CrmReportSource``. 
To create custom resource it is enough to inherit from aforementioned abstract class and provide implementation of 
needed methods/properties. Actually, each data source is just a suitable wrapper over ``pandas.DataFrame`` object 
that makes all needed preparations (reads file, downloads and parses JSON, etc.) before data frame is ready to be 
used by other parts of the system.


Available Data Sources
----------------------

Following data sources are already avaialbe:

1. `CsvDataSource` - creates source from CSV-file content;
2. `SpreadsheetDataSource` - create source from plain Excel table;
3. `WebAnalyticsSource` - quite specific resource that retrieves data from remote host and uses local `IP2Location` database to retrieve geo information from IP addresses;
4. `CrmReportSource` - another specific resource that retrieves data from Excel spreadsheet and extends it with latitude/longitude coordinates.

For instance, to implement retrieving JSON data from remote host, the following
snippet can be used:
```Python
url = "http://analytics.host/api/get_data"
database_with_geo_info = "data/ipdata.bin"

# create data source
source = WebAnalyticsSource(url, database_path=database_with_geo_info)

# remove bad records and extend with lat/long and domain values
try:
    source.prepare()    
    if source.ready:
    	do_something(source.data)
    else:
    	print("error: cannot prepare source")

except Exception as e:
    print("error: %s" % e)
```

Another custom data source is a CRM report that is an Excel table that should
be preprocessed in a special way before it can be used. 
```Python
file_path = "docs/report.xls"

# override default column names if needed
input_columns = {
    "address": "Company Address",
    "email": "Email Address"
}
output_columns = {
    "domain": "Company Domain",
    "lat": "Company Latitude",
    "lon": "Company Longitude"
}

# create data source
source = CrmReportSource(file_path, input=input_columns, output=output_columns)

# prepare and use
source.prepare()
do_something(source.data)
```


Custom Data Sources Creation
----------------------------
If it is needed to create a wrapper over fixed data frame to match DataSource
interface, the following code can be used:
```python
import pandas as pd
from sources.data import DataSource

class FixedValuesDataSource(DataSource):

    def __init__(self):
        self._name = "fixed_data"
    
    def prepare(self):
        pass  # Nothing to do there
    
    @property
    def ready(self):
        return True  # Source is always ready
        
    @property
    def data(self):
        df = pd.DataFrame({
            "first": ["a", "b", "c"],
            "second": [1, 2, 3]
        })
        return df
```

Then created data source can be used as followed:
```python
source = FixedValuesDataSource()
assert source.ready, "Usually, source should be prepared before usage"
source.prepare()
assert source.ready, "But this source is always ready"
```

Internal data frame can be accessed via **date** property:
```python
>>> print(source.data)
  first  second
0     a       1
1     b       2
2     c       3
```

Consider that several data source were created. The following snippet can be used to 
make slice of columns that are common between several sources and calculate features on 
them:
```python
# let's pretend that there is a function that creates data sources
source1, source2 = create_two_data_sources()

# create configuration for columns slicing
first_source_columns = "first", "second"
second_source_columns = "first", "another_name"
slice_config = {
    source1: first_source_columns,
    source2: second_source_columns
}

# create records linker and take only specified columns from data sources
from linkage.linker import RecordsLinker
linker = RecordsLinker()
linker.match_common_columns(slice_config)
```
Then, extracted column slices can be used to calculate features.


Phrasely Package
----------------
Simple sentiment analysis implementation. The package description can be found in 
the separate [README file](docs/Phrasely.md).
  

Predict Ally Package
--------------------
This package is also quite important and it's description can be found in [there](docs/PredictAlly.md).
 

Master Predictors LookUp
-------------------------
```
from qfi.master_predictors_lookup import MasterPredictorsLookupMixin
mpl = MasterPredictorsLookupMixin(database='clientdb')
mpl.update_table(db='client_db')
```

Transformation
--------------
```
python main.py transform -db=clientdb
```


Source Code
-----------
Source code can be found in the [Github repository](https://github.com/Vectorscient/python).
