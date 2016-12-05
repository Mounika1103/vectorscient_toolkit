import subprocess
import shlex
import numpy

from sqlalchemy import func, select, Table, MetaData
from sqlalchemy.sql import and_

from ..sources.connection import DBConnection
from ..sources.mixins import GeoInformation
from ..sources.models import WebLanding
from ..config import FEBRL_CONFIG
from ..utils import patch_ip


class WebDataSource(DBConnection, GeoInformation):
    """ Class that handles the activities of the
        web data source.
    """
    def __init__(self, **kwargs):
        super(WebDataSource, self).__init__(**kwargs)

    def save_to_db(self, df):
        table_name = 'vs_web_lnd_tbl'
        table = Table(table_name, MetaData(bind=self.cursor), autoload=True)
        records = self._records_to_dict(df)
        saved_records = 0

        records_retrieved = len([i for i in records])
        print('{count} Records Retrieved.'.format(count=records_retrieved))

        for raw_record in records:
            record = {}
            # clean unused data
            data_fields = [i.name for i in WebLanding.__table__.columns]
            for key, value in raw_record.items():
                if key in data_fields:
                    value = value.encode('utf-8') if type(value) == str else value
                    record.update({key: value})

            record = self._clean_record(record)

            # add geo information
            visit_ip = record['visitIp'].decode('utf-8') if type(record['visitIp']) == bytes else record['visitIp']
            # just a NoOp if PATCH_IP_ADDRESSES = False
            prepared_ip = patch_ip(visit_ip)

            record.update(self.get_geoinformation(prepared_ip))
            statement = select([table.c.record_set_id]).where(and_(
                table.c.Domain==record['Domain'],
                table.c.Isp==record['Isp'],
                table.c.Latitude_derived==record['Latitude_derived'],
                table.c.Longitude_derived==record['Longitude_derived'],
            ))
            result = list(self.cursor.execute(statement))
            if len(result) > 0:
                record.update({'record_set_id': result[0][0]})
                self.insert(table_name, [record])

            # no existing record. calculate new record_set_id
            record_id = self._get_latest_record_id()
            record.update({'record_set_id': record_id})
            result = self.insert(table_name, [record])
            if result['saved']:
                print(result['saved'])
                saved_records += 1

        print('Inserted {}'.format(saved_records))

    def _records_to_dict(self, df):
        total_row_count = len(df.axes[0])

        for index in range(total_row_count):
            yield df.loc[index].to_dict()

    def _get_latest_record_id(self):
        """ Checks the database and return the last record_set_id
        """
        web_landing = Table('vs_web_lnd_tbl', MetaData(bind=self.cursor), autoload=True)
        # get the latest record id
        record = list(self.cursor.execute(
            select([func.max(web_landing.c.record_set_id)])))[0][0]
        record_id = (record or 0) + 1

        self.latest_record_id = record_id
        return record_id

    def _clean_record(self, record):
        fields = {'City': 'city', 'Latitude': 'latitude',
            'Longitude': 'longitude', 'Country': 'country', 'IP': 'ip'}
        for key, value in fields.items():
            record[value] = record.pop(key, None)

        # convert nan
        for key, value in record.items():
            try:
                if numpy.isnan(value):
                    record[key] = None
            except:
                pass

        return record


class CrmDataSource(DBConnection, GeoInformation):
    """ class that handles the processing of
        crm structured data
    """
    def __init__(self, **kwargs):
        self.database = kwargs.get('database')
        super(CrmDataSource, self).__init__(**kwargs)

    def update(self):
        table_name = 'CRM_structured_lnd_tbl'
        table = Table(table_name, MetaData(bind=self.cursor), autoload=True)

        statement = select([
            table.c.vs_id,
            table.c.customer_billing_address,
            table.c.domain,
            table.c.customer_name,
        ])
        result = list(self.cursor.execute(statement))
        for item in result:
            # retrieve coordinates
            latitude, longitude = self.get_long_lat(item[1])
            if not latitude and not longitude:
                continue

            # assign record_set_id
            s = select([table.c.record_set_id]).where(and_(
                table.c.domain==item[2],
                table.c.customer_name==item[3],
                table.c.Customer_billing_addr_latitude==latitude,
                table.c.Customer_billing_addr_longitude==longitude,
            ))
            output = list(self.cursor.execute(s))

            # combination existed. will just copy the record_set_id
            # of the existing data
            record_set_id = output[0][0] if len(output) > 0 else self._get_latest_record_id()
            # update data
            obj = table.update().where(table.c.vs_id==item[0]).values(
                record_set_id=str(record_set_id),
                Customer_billing_addr_latitude=latitude,
                Customer_billing_addr_longitude=longitude,
            )

            self.cursor.execute(obj)

    def _get_latest_record_id(self):
        """ Checks the database and return the last record_set_id
        """
        result = Table('CRM_structured_lnd_tbl', MetaData(bind=self.cursor), autoload=True)
        # get the latest record id
        record = list(self.cursor.execute(
            select([func.max(result.c.record_set_id)])))[0][0]

        record_id = (int(record) if record else 0) + 1

        self.latest_record_id = record_id
        return record_id