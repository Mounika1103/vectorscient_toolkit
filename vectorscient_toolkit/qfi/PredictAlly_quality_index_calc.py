import logging

from scipy.spatial import distance

from ..sources.connection import DBConnection
from .models import Base


class QFI(DBConnection):
    """
    Calculate the Prediction Quality Index

    How to Run the Script:

    from PredictAlly_quality_index_calc import QFI
    qfi = QFI(database='clientdb')
    qfi.process(pred_run_date='2016-01-08')

    """

    TABLE_CHOICES = ('new', 'exist')

    def __init__(self, **kwargs):
        self.db = kwargs.get('database')
        super(QFI, self).__init__(database=self.db)

    def calculate_euclidean_distance(self, centroid_vector, records):
        """
        Calculate Euclidean distance to the CENTROID
        """
        try:
            result = distance.euclidean(centroid_vector, records)
        except:
            result = 0
        return result

    def compute_normalize(self, x, min_val, max_val, a=1, b=100):
        """

        Formula:
                (b-a)(x - min)
        f(x) = --------------  + a
                  max - min

        Sample: Calculate the Index value of this number "5.579767005" by applying the rule.
        [a, b] = [1, 100]
        Min = 3 and max = 8 ; x=5.579
        F(x) = ((100-1)(5.579 – 3) /(8-3) ) + 1 = ((99*2.579)/5) + 1 = 52.0642

        """

        ab_result = b - a
        xmin_result = round(x - min_val, 3)
        maxmin_result = round(max_val - min_val, 3)
        ab_xmin = round(ab_result * xmin_result, 3)
        try:
            result = round(ab_xmin / maxmin_result, 4) + 1
        except ZeroDivisionError as e:
            logging.error(str(e))
            return 0
        return result

    def get_master_predictors_lookup_feature_names(self, instance, data_source_cat=''):
        """Return the attr filter by data_source_cat
        """
        attrs = []

        if data_source_cat:
            query = "SELECT * FROM master_predictors_lookup WHERE data_source_cat='{data_source_cat}' ORDER BY feature_name".format(data_source_cat=data_source_cat)
        else:
            query = "SELECT * FROM master_predictors_lookup ORDER BY feature_name"

        master_predictors_lookup = self.session.execute(query)

        for obj in master_predictors_lookup:
            feature_name = obj.feature_name.lower().replace('-', '_')

            try:
                if feature_name == '% of time product sold':
                    feature_name = 'percent_of_time_product_sold'

                if getattr(instance, feature_name) is not None:
                    attrs.append(getattr(instance, feature_name))
            except:
                pass

        return attrs

    def get_qfi_overall_indexes(self, instance):
        """
        Return list of columns to use when calculating the Euclidean distance for the index QFI_OVERALL
        """
        return self.get_master_predictors_lookup_feature_names(instance)

    def get_qfi_web_intra_indexes(self, instance):
        """Return fields for WEB
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'WEB')

    def get_qfi_hd_intra_indexes(self, instance):
        """Return fields for HELPDESK
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'HELPDESK')

    def get_qfi_crm_intra_indexes(self, instance):
        """Return fields for CRM
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'CRM')

    def get_qfi_sh_intra_indexes(self, instance):
        """Return fields for qfi_sh_intra
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'SALES_HISTORY')

    def get_qfi_sm_intra_indexes(self, instance):
        """Return fields for SOCIAL_MEDIA
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'SOCIAL_MEDIA')

    def get_qfi_web_inter_indexes(self, instance):
        """Return fields for WEB
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'WEB')

    def get_qfi_hd_inter_indexes(self, instance):
        """Return fields for qfi HELPDESK
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'HELPDESK')

    def get_qfi_crm_inter_indexes(self, instance):
        """
        Return the field for CRM
        """
        return self.get_master_predictors_lookup_feature_names(instance, 'CRM')

    def get_qfi_sh_inter_indexes(self, instance):
        return self.get_master_predictors_lookup_feature_names(instance, 'SALES_HISTORY')

    def get_qfi_sm_inter_indexes(self, instance):
        return self.get_master_predictors_lookup_feature_names(instance, 'SOCIAL_MEDIA')

    def get_pred_run_date(self, table):
        """Return the latest pred_run_date
        """

        sql = """SELECT
                    COALESCE(max({table}.pred_run_date), CURDATE()) as pred_run_date
                FROM {table}
        """.format(table=table)
        qry = self.session.execute(sql).fetchall()
        for instance in qry:
            return instance.pred_run_date

    def calc_qfi(self, pred_run_date=''):
        """
        Return the values from table
        """

        ClusteredFilePros = 'clustered_file_new_pros'
        ClusteredFileProsNorm = 'clustered_file_new_pros_norm'
        CentroidsPros = 'centroids_new_pros'

        if not pred_run_date:
            pred_run_date = self.get_pred_run_date(table=CentroidsPros)

        centroid_qry = "SELECT * from {} WHERE pred_run_date='{}'".format(CentroidsPros, pred_run_date)
        centroids_pros = self.session.execute(centroid_qry).fetchall()

        for centroid_instance in centroids_pros:
            qfi_overall_distances = []
            qfi_web_intra_distances = []
            qfi_hd_intra_distances = []
            qfi_crm_intra_distances = []
            qfi_sh_intra_distances = []
            qfi_sm_intra_distances = []
            qfi_web_inter_distances = []
            qfi_hd_inter_distances = []
            qfi_crm_inter_distances = []
            qfi_sh_inter_distances = []
            qfi_sm_inter_distances = []

            centroid_qfi_overall_index = self.get_qfi_overall_indexes(centroid_instance)
            centroid_qfi_web_intra_index = self.get_qfi_web_intra_indexes(centroid_instance)
            centroid_qfi_hd_intra_index = self.get_qfi_hd_intra_indexes(centroid_instance)
            centroid_qfi_crm_intra_index = self.get_qfi_crm_intra_indexes(centroid_instance)
            centroid_qfi_sh_intra_index = self.get_qfi_sh_intra_indexes(centroid_instance)
            centroid_qfi_sm_intra_index = self.get_qfi_sm_intra_indexes(centroid_instance)
            centroid_qfi_web_inter_index = self.get_qfi_web_inter_indexes(centroid_instance)
            centroid_qfi_hd_inter_index = self.get_qfi_hd_inter_indexes(centroid_instance)
            centroid_qfi_crm_inter_index = self.get_qfi_crm_inter_indexes(centroid_instance)
            centroid_qfi_sh_inter_index = self.get_qfi_sh_inter_indexes(centroid_instance)
            centroid_qfi_sm_inter_index = self.get_qfi_sm_inter_indexes(centroid_instance)

            cluster_qry = "SELECT * from {table} WHERE cluster_class={cluster_class} and pred_run_date='{pred_run_date}' ORDER BY customer_id".format(table=ClusteredFileProsNorm, cluster_class=centroid_instance.cluster_class, pred_run_date=pred_run_date)
            clustered_file_pros_norm = self.session.execute(cluster_qry).fetchall()

            for record_instance in clustered_file_pros_norm:
                # euclidean distance for qfi_overall
                qfi_overall_indexes = self.get_qfi_overall_indexes(record_instance)
                qfi_overall_distance = self.calculate_euclidean_distance(centroid_qfi_overall_index, qfi_overall_indexes)
                qfi_overall_distances.append(qfi_overall_distance)

                # euclidean distance for get_qfi_web_intra_indexes
                qfi_web_intra_indexes = self.get_qfi_web_intra_indexes(record_instance)
                qfi_web_intra_distance = self.calculate_euclidean_distance(centroid_qfi_web_intra_index, qfi_web_intra_indexes)
                qfi_web_intra_distances.append(qfi_web_intra_distance)

                # euclidean distance for qfi_hd_intra
                qfi_hd_intra_indexes = self.get_qfi_hd_intra_indexes(record_instance)
                qfi_hd_intra_distance = self.calculate_euclidean_distance(centroid_qfi_hd_intra_index, qfi_hd_intra_indexes)
                qfi_hd_intra_distances.append(qfi_hd_intra_distance)

                # euclidean distance for qfi_crm_intra
                get_qfi_crm_intra_indexes = self.get_qfi_crm_intra_indexes(record_instance)
                qfi_crm_intra_distance = self.calculate_euclidean_distance(centroid_qfi_crm_intra_index, get_qfi_crm_intra_indexes)
                qfi_crm_intra_distances.append(qfi_crm_intra_distance)

                # euclidean distance for qfi_sh_intra
                get_qfi_sh_intra_indexes = self.get_qfi_sh_intra_indexes(record_instance)
                qfi_sh_intra_distance = self.calculate_euclidean_distance(centroid_qfi_sh_intra_index, get_qfi_sh_intra_indexes)
                qfi_sh_intra_distances.append(qfi_sh_intra_distance)

                # euclidean distance for qfi_sm_intra
                get_qfi_sm_intra_indexes = self.get_qfi_sm_intra_indexes(record_instance)
                qfi_sm_intra_distance = self.calculate_euclidean_distance(centroid_qfi_sm_intra_index, get_qfi_sm_intra_indexes)
                qfi_sm_intra_distances.append(qfi_sm_intra_distance)

                # euclidean distance for qfi_web_inter
                get_qfi_web_inter_indexes = self.get_qfi_web_inter_indexes(record_instance)
                qfi_web_inter_distance = self.calculate_euclidean_distance(centroid_qfi_web_inter_index, get_qfi_web_inter_indexes)
                qfi_web_inter_distances.append(qfi_web_inter_distance)

                # euclidean distance for qfi_hd_inter
                get_qfi_hd_inter_indexes = self.get_qfi_hd_inter_indexes(record_instance)
                qfi_hd_inter_distance = self.calculate_euclidean_distance(centroid_qfi_hd_inter_index, get_qfi_hd_inter_indexes)
                qfi_hd_inter_distances.append(qfi_hd_inter_distance)

                # euclidean distance for qfi_crm_inter
                get_qfi_crm_inter_indexes = self.get_qfi_crm_inter_indexes(record_instance)
                qfi_crm_inter_distance = self.calculate_euclidean_distance(centroid_qfi_crm_inter_index, get_qfi_crm_inter_indexes)
                qfi_crm_inter_distances.append(qfi_crm_inter_distance)

                # euclidean distance for qfi_sh_inter
                get_qfi_sh_inter_indexes = self.get_qfi_sh_inter_indexes(record_instance)
                qfi_sh_inter_distance = self.calculate_euclidean_distance(centroid_qfi_sh_inter_index, get_qfi_sh_inter_indexes)
                qfi_sh_inter_distances.append(qfi_sh_inter_distance)

                # euclidean distance for qfi_sm_inter
                get_qfi_sm_inter_indexes = self.get_qfi_sm_inter_indexes(record_instance)
                qfi_sm_inter_distance = self.calculate_euclidean_distance(centroid_qfi_sm_inter_index, get_qfi_sm_inter_indexes)
                qfi_sm_inter_distances.append(qfi_sm_inter_distance)

            # start the computation and save qfi attributes
            counter = 0

            clustered_file_pros = self.session.execute("SELECT * FROM {table} WHERE cluster_class={cluster_class} AND pred_run_date='{pred_run_date}' ORDER BY customer_id".format(table=ClusteredFilePros, cluster_class=centroid_instance.cluster_class, pred_run_date=pred_run_date))

            for record_instance in clustered_file_pros:
                qfi_overall = self.compute_normalize(qfi_overall_distances[counter], min(qfi_overall_distances), max(qfi_overall_distances))
                qfi_web_intra = self.compute_normalize(qfi_web_intra_distances[counter], min(qfi_web_intra_distances), max(qfi_web_intra_distances))
                qfi_hd_intra = self.compute_normalize(qfi_hd_intra_distances[counter], min(qfi_hd_intra_distances), max(qfi_hd_intra_distances))
                qfi_crm_intra = self.compute_normalize(qfi_crm_intra_distances[counter], min(qfi_crm_intra_distances), max(qfi_crm_intra_distances))
                qfi_sh_intra = self.compute_normalize(qfi_sh_intra_distances[counter], min(qfi_sh_intra_distances), max(qfi_sh_intra_distances))
                qfi_sm_intra = self.compute_normalize(qfi_sm_intra_distances[counter], min(qfi_sm_intra_distances), max(qfi_sm_intra_distances))
                qfi_web_inter = self.compute_normalize(qfi_web_inter_distances[counter], min(qfi_web_inter_distances), max(qfi_web_inter_distances))
                qfi_hd_inter = self.compute_normalize(qfi_hd_inter_distances[counter], min(qfi_hd_inter_distances), max(qfi_hd_inter_distances))
                qfi_crm_inter = self.compute_normalize(qfi_crm_inter_distances[counter], min(qfi_crm_inter_distances), max(qfi_crm_inter_distances))
                qfi_sh_inter = self.compute_normalize(qfi_sh_inter_distances[counter], min(qfi_sh_inter_distances), max(qfi_sh_inter_distances))
                qfi_sm_inter = self.compute_normalize(qfi_sm_inter_distances[counter], min(qfi_sm_inter_distances), max(qfi_sm_inter_distances))

                update_qry = """
                    UPDATE
                        {table_name}
                    SET
                        qfi_overall={qfi_overall},
                        qfi_web_intra={qfi_web_intra},
                        qfi_hd_intra={qfi_hd_intra},
                        qfi_crm_intra={qfi_crm_intra},
                        qfi_sh_intra={qfi_sh_intra},
                        qfi_sm_intra={qfi_sm_intra},
                        qfi_web_inter={qfi_web_inter},
                        qfi_hd_inter={qfi_hd_inter},
                        qfi_crm_inter={qfi_crm_inter},
                        qfi_sh_inter={qfi_sh_inter},
                        qfi_sm_inter={qfi_sm_inter},
                        distance_to_centroid={distance_to_centroid}
                    WHERE id={id}
                """.format(table_name=ClusteredFilePros,
                            id=record_instance.id,
                            qfi_overall=qfi_overall,
                            qfi_web_intra=qfi_web_intra,
                            qfi_hd_intra=qfi_hd_intra,
                            qfi_crm_intra=qfi_crm_intra,
                            qfi_sh_intra=qfi_sh_intra,
                            qfi_sm_intra=qfi_sm_intra,
                            qfi_web_inter=qfi_web_inter,
                            qfi_hd_inter=qfi_hd_inter,
                            qfi_crm_inter=qfi_crm_inter,
                            qfi_sh_inter=qfi_sh_inter,
                            qfi_sm_inter=qfi_sm_inter,
                            distance_to_centroid=qfi_overall_distances[counter]
                        )
                self.cursor.execute(update_qry)

                print ('----------------------------------------------------')
                print ('customer id: {}'.format(record_instance.customer_id))
                print ('distance_to_centroid: {}'.format(qfi_overall_distances[counter]))
                print ('qfi_overall: {}'.format(qfi_overall))
                print ('qfi_web_intra: {}'.format(qfi_web_intra))
                print ('qfi_hd_intra: {}'.format(qfi_hd_intra))
                print ('qfi_crm_intra: {}'.format(qfi_crm_intra))
                print ('qfi_sh_intra: {}'.format(qfi_sh_intra))
                print ('qfi_sm_intra: {}'.format(qfi_sm_intra))
                print ('qfi_web_inter: {}'.format(qfi_web_inter))
                print ('qfi_hd_inter: {}'.format(qfi_hd_inter))
                print ('qfi_crm_inter: {}'.format(qfi_crm_inter))
                print ('qfi_sh_inter: {}'.format(qfi_sh_inter))
                print ('qfi_sm_inter: {}'.format(qfi_sm_inter))

                counter += 1

    def process(self, pred_run_date=''):
        return self.calc_qfi(pred_run_date)

    def create_tables(self):
        """Create tables
        """
        print ('Creating QFI table ...')
        Base.metadata.create_all(self.cursor)
        print ('QFI table created.')
