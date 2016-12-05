# QFI Prediction #


## How to Run the script? ##

```
from PredictAlly_quality_index_calc import QFIMixin
calc = QFIMixin()
calc.process(db='client_db', input_cat='exist', pred_run_date='2016-01-08')
```

## Add column base on master_predictors_lookup ##

```
from master_predictors_lookup import MasterPredictorsLookupMixin
mixin = MasterPredictorsLookupMixin()
mixin.add_column(db='client_db')
```