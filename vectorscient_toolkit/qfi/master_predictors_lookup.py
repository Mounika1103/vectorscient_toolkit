import logging

from ..sources.connection import DBConnection


class MasterPredictorsLookupMixin(DBConnection):
    """
    Add column base on master_predictors_lookup

    How to run:
        from qfi.master_predictors_lookup import MasterPredictorsLookupMixin
        mpl = MasterPredictorsLookupMixin(database='clientdb')
        mpl.update_table(db='client_db')

    """

    def __init__(self, **kwargs):
        self.db = kwargs.get('database')
        super(MasterPredictorsLookupMixin, self).__init__(database=self.db)

    def update_table(self):
        """Create the field listed on master_predictors_lookup data
        """
        data_type = 'float'
        tables = [
            'centroids_exis_pros',
            'centroids_new_pros',
            'clustered_file_exis_pros_norm',
            'clustered_file_new_pros_norm',
            'clustered_file_exis_pros',
            'clustered_file_new_pros'

        ]
        master_predictors_lookup_qry = self.session.execute("SELECT * FROM master_predictors_lookup").fetchall()

        for table_name in tables:
            for instance in master_predictors_lookup_qry:
                column_name = instance.feature_name.lower()

                try:
                    self.add_column(table_name, column_name, data_type)
                except Exception as e:
                    logging.error(str(e))

    def add_column(self, table, column, data_type):
        """Add column to table
        """
        qry = """ALTER
                TABLE {table}
                ADD column {column} {data_type}
            """.format(table=table, column=column, data_type=data_type)
        self.cursor.execute(qry)
