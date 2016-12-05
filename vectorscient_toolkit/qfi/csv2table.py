import csv
import logging
import datetime

from sqlalchemy import Table, MetaData

from ..sources.connection import DBConnection


class CsvImportMixin(DBConnection):
    """
        ### Import CSV data ###

        from qfi.csv2table import CsvImportMixin
        csvmixin = CsvImportMixin(database='clientdb')
        csvmixin.save_imports('<table name>', '<filepath>')
    """

    def read_csv(self, filepath):
        """
        Read the csv file and return a dictionary
        """
        f = open(filepath, 'r', encoding='iso-8859-1')
        reader = csv.DictReader(f)
        logging.info('importing {}'.format(filepath))
        return reader

    def get_or_create_clusterclass(self, values):
        """
        get or save clusterclass and return the object
        """
        cluster_class = values['cluster_class']
        cluster_class_qry = self.session.execute("SELECT * FROM cluster_class WHERE cluster_class={}".format(cluster_class)).fetchall()
        table = Table('cluster_class', MetaData(bind=self.cursor), autoload=True)
        # check if cluster_class already added
        if len(cluster_class_qry) == 0:
            datatable = table.insert()
            qry = datatable.values(**values)
            self.session.execute(qry)
            self.session.commit()
            logging.info("{} cluster_class created".format(cluster_class))

        return cluster_class

    def empty_to_zero(self, val):
        """
        Convert the empty value to zero
        """
        return sum(int(float(item)) for item in val if item)

    def clean_dict(self, data):
        """
        Correct the keys from the csv file.
        """

        if 'EarliestYear' in data:
            data['earliest_year'] = data['EarliestYear']
            # remove the unneeded keys
            del data['EarliestYear']

        if 'LatestYear' in data:
            data['latest_year'] = data['LatestYear']
            # remove LatestYear
            del data['LatestYear']

        if '% of time product sold' in data:
            data['percent_of_time_product_sold'] = data['% of time product sold']
            # remove the unneeded keys
            del data['% of time product sold']

        if '' in data:
            del data['']

        if 'cluster_class_name' in data:
            del data['cluster_class_name']

        # convert keys to lowercase
        data = dict((k.lower().rstrip().replace('-', '_').replace(' ', '_'), v) for k, v in data.items())

        # convert empty to 0
        if 'distance_to_centroid' in data:
            data['distance_to_centroid'] = self.empty_to_zero(data['distance_to_centroid'])

        if 'qfi_overall' in data:
            data['qfi_overall'] = self.empty_to_zero(data['qfi_overall'])
            data['qfi_web_intra'] = self.empty_to_zero(data['qfi_web_intra'])
            data['qfi_hd_intra'] = self.empty_to_zero(data['qfi_hd_intra'])
            data['qfi_crm_intra'] = self.empty_to_zero(data['qfi_crm_intra'])
            data['qfi_sh_intra'] = self.empty_to_zero(data['qfi_sh_intra'])
            data['qfi_sm_intra'] = self.empty_to_zero(data['qfi_sm_intra'])
            data['qfi_hd_inter'] = self.empty_to_zero(data['qfi_hd_inter'])
            data['qfi_crm_inter'] = self.empty_to_zero(data['qfi_crm_inter'])
            data['qfi_sh_inter'] = self.empty_to_zero(data['qfi_sh_inter'])
            data['qfi_sm_inter'] = self.empty_to_zero(data['qfi_sm_inter'])
            data['qfi_web_inter'] = self.empty_to_zero(data['qfi_web_inter'])

        if 'pred_run_date' in data:

            # convert date to python datetime object
            if data['pred_run_date']:
                pred_run_date = datetime.datetime.strptime(data['pred_run_date'], '%m/%d/%y')
                logging.error('{} error date'.format(pred_run_date))
                data['pred_run_date'] = pred_run_date
            else:
                data['pred_run_date'] = datetime.date.today().strftime('%Y-%m-%d')

        if 'length' in data:
            if data['length'] == '':
                data['length'] = 0

        return data

    def save_imports(self, tablename, filepath):
        """
        Store the csv to db
        """

        # read the csv file
        csv_data = self.read_csv(filepath)
        logging.info('storing {} to {} table'.format(filepath, tablename))
        table = Table(tablename, MetaData(bind=self.cursor), autoload=True)

        for data in csv_data:
            if 'cluster_class' in data:
                # get cluster_class  values
                cluster_class_values = {
                    'cluster_class': data['cluster_class'],
                    'cluster_class_name': data['cluster_class_name']
                }
                # get or create the cluster_class
                cluster_class = self.get_or_create_clusterclass(cluster_class_values)
                cluster_class_qry = self.session.execute("SELECT * FROM cluster_class WHERE cluster_class={}".format(cluster_class)).fetchall()
                data['cluster_class'] = cluster_class_qry[0].id

            data = self.clean_dict(data)
            datatable = table.insert()
            qry = datatable.values(**data)
            self.cursor.execute(qry)
