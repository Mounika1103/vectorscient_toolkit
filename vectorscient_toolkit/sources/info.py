from sqlalchemy import select, func
import pandas as pd

from .connection import DBConnection


class ClusteringStats(DBConnection):
    """
    Implements database queries to request data needed for opportunity
    clustering reporting.

    Note that it's a kind of ad-hoc solution. It is worth to think about
    more structured implementation.
    """

    TABLE_CLUSTERED_FILE_NEW_PROS = "clustered_file_new_pros"
    TABLE_MASTER_PREDICTORS_LOOKUP = "master_predictors_lookup"
    TABLE_VS_WEB_CRM_RL = "vs_web_crm_rl"
    TABLE_VS_WEB_CLIENT_CONFIG = "vs_web_client_config"
    TABLE_VS_CLIENT_TIME_CONFIG = "vs_client_time_config"

    def __init__(self, database):
        super(ClusteringStats, self).__init__(database)

    def number_of_records(self):
        return len(self.select(self.TABLE_CLUSTERED_FILE_NEW_PROS))

    def number_of_columns(self):
        return len(self.select(self.TABLE_CLUSTERED_FILE_NEW_PROS)[0])

    def number_of_matched_records(self):
        return self._number_of_records_for_cluster("Match")

    def number_of_unmatched_records(self):
        return self._number_of_records_for_cluster("Unmatch")

    def number_of_likely_matched_records(self):
        return self._number_of_records_for_cluster("Likely Match")

    def _number_of_records_for_cluster(self, cluster_name):
        table = self.table(self.TABLE_VS_WEB_CRM_RL)
        n_matched = select([func.count()]).select_from(table).where(
            table.c.matched_status == cluster_name).scalar()
        return n_matched

    def predictors_and_weights(self):
        return self._get_table(
            self.TABLE_MASTER_PREDICTORS_LOOKUP,
            columns=["feature_name", "final_wt"],
            verbose=["Predictor Name", "Final Weight"])

    def time_decay_factors(self):
        columns = [
            "datasource_name",
            "time_scale_in_days",
            "time_bucket_1_in_percent",
            "time_bucket_2_in_percent",
            "time_bucket_3_in_percent",
            "time_bucket_4_in_percent",
            "time_bucket_5_in_percent"
        ]
        verbose_names = [
            "Data Source Name",
            "Time Horizon (in days)",
            "Most Recent Bucket (in %)",
            "Second Most Recent Bucket (in %)",
            "Third Most Recent Bucket (in %)",
            "Fourth Most Recent Bucket (in %)",
            "Fifth Most Recent Bucket (in %)"
        ]
        factors = self._get_table(
            self.TABLE_VS_CLIENT_TIME_CONFIG, columns, verbose_names)
        source_name = verbose_names[0]
        columns = factors[source_name]
        rest = pd.DataFrame(factors.drop(source_name, axis=1))
        t = rest.transpose()
        t.reset_index(inplace=True)
        t.columns = ["Percentile"] + columns.tolist()
        return t

    def web_activity_segments(self):
        return self._get_table(
            self.TABLE_VS_WEB_CLIENT_CONFIG,
            columns=["action_type_name",
                     "action_description",
                     "segment_name",
                     "web_activity_bucket_name"],
            verbose=["Action Type",
                     "Action Description",
                     "Segment Name",
                     "Web Activity Importance"])

    def _get_table(self, table_name, columns, verbose):
        rows = self.select(table_name, columns)
        result = pd.DataFrame.from_records(
            dict(zip(verbose, r)) for r in rows)
        return result[verbose]

    def crm_to_web_matched(self):
        return self._matched_records_for_cluster("Match")

    def crm_to_web_unmatched(self):
        return self._matched_records_for_cluster("Unmatch")

    def crm_to_web_likely_matched(self):
        return self._matched_records_for_cluster("Likely Match")

    def _matched_records_for_cluster(self, cluster_name):
        table = self.table(self.TABLE_VS_WEB_CRM_RL)
        rows = select([
            table.c.web_isp,
            table.c.crm_account_name,
            table.c.crm_address,
            table.c.web_visit_date
        ]).where(table.c.matched_status == cluster_name).execute()
        verbose_names = [
            "ISP",
            "Customer Name",
            "Billing Address",
            "Web Visit Date"
        ]
        result = pd.DataFrame.from_records(
            dict(zip(verbose_names, r)) for r in rows)
        return result
