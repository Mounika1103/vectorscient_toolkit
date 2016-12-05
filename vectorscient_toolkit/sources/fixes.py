from sqlalchemy import func, select, Table, MetaData
from sqlalchemy.sql import and_

from ..sources.connection import DBConnection
from ..sources.mixins import GeoInformation


class FixRecordSetId(DBConnection, GeoInformation):

    DB_TYPE = {
        'web': 'vs_web_lnd_tbl',
        'crm': 'CRM_structured_lnd_tbl',
    }

    def __init__(self, **kwargs):
        self.db_type = kwargs.get('db_type')
        self.database = kwargs.get('database')
        super(FixRecordSetId, self).__init__(**kwargs)

    def fix_record_set_id(self):
        table_name = self.DB_TYPE[self.db_type]
        table = Table(table_name, MetaData(bind=self.cursor), autoload=True)

        statement = select([
            table.c.id,
            table.c.record_set_id,
            table.c.Domain,
            table.c.Isp,
            table.c.Latitude_derived,
            table.c.Longitude_derived,
        ])
        data = list(self.cursor.execute(statement))

        # assign record set id
        for item in data:
            s = select([table.c.record_set_id]).where(and_(
                table.c.Domain==item[2],
                table.c.Isp==item[3],
                table.c.Latitude_derived==item[4],
                table.c.Longitude_derived==item[5],
            ))
            output = list(self.cursor.execute(s))

            # combination existed. will just copy the record_set_id
            # of the existing data
            #record_set_id = output[0][0] if len(output) > 0 else self._get_latest_record_id()
            if len(output) and output[0][0] != 0:
                record_set_id = output[0][0]
            else:
                record_set_id = self._get_latest_record_id()


            # update data
            obj = table.update().where(table.c.id==item[0]).values(
                record_set_id=record_set_id,
            )

            self.cursor.execute(obj)
            print("fix {}".format(item[0]))

    def _get_latest_record_id(self):
        """ Checks the database and return the last record_set_id
        """
        result = Table(self.DB_TYPE[self.db_type], MetaData(bind=self.cursor), autoload=True)
        # get the latest record id
        record = list(self.cursor.execute(
            select([func.max(result.c.record_set_id)])))[0][0]

        record_id = (int(record) if record else 0) + 1

        self.latest_record_id = record_id
        return record_id