from datetime import date, timedelta, datetime

from sqlalchemy import Table, MetaData
from sqlalchemy.sql.expression import bindparam

from ..sources.connection import DBConnection


class WebDataTransformation(DBConnection):
    """Class to handle web data transformation
    """

    web_activity_type_ids = [1, 2, 3]

    def __init__(self, **kwargs):
        self.db = kwargs.get('database')
        super(WebDataTransformation, self).__init__(database=self.db)

    # SECTION A.

    def get_vs_web_lnd_tbl(self):
        """Return data from vs_web_lnd_tbl table
        """

        raw_sql = """SELECT
                        LND.actionDetails_pageTitle,
                        conv(hex(LND.visitorId), 16, 10) AS visitorId,
                        LND.record_set_id,
                        LND.matched_status,
                        LND.serverDate,
                        LND.serverTimestamp,
                        LND.visitorType,
                        LND.visitCount,
                        LOG.type
                        FROM vs_web_lnd_tbl LND
                        JOIN vsweb_log_action LOG ON LND.actionDetails_pageTitle=LOG.name
                """
        return self.session.execute(raw_sql).fetchall()

    def get_client_config_and_inter_input(self):
        """Populate the remaining columns of segment_id , web_activity_type_id by using the below sql
        """
        raw_sql = """SELECT
                        INTER_INPUT.vs_id as vs_id,
                        CONFIG.segment_id AS segment_id,
                        CONFIG.web_activity_type_id AS web_activity_type_id
                    FROM
                        vs_web_client_config CONFIG,
                        vs_web_inter_input INTER_INPUT
                    WHERE
                        CONFIG.action_type_id=INTER_INPUT.action_type_id
                """
        # FIXME (ilia): not sure, but it doesn't work on my machine (returns zero records)
        # FIXME (ilia): so it was replaced with cursor query
        # return self.session.execute(raw_sql).fetchall()
        return self.cursor.execute(raw_sql).fetchall()

    def insert_web_inter_input(self):
        """insert get_vs_web_lnd_tbl result t vs_web_inter_input table
        """

        print ('Querying...')
        web_lnd_tbl_qry = self.get_vs_web_lnd_tbl()
        counter = 0
        print ('Start Inserting...')

        rows = [{
                'actionDetails_pageTitle': instance.actionDetails_pageTitle,
                'visitorId': instance.visitorId,
                'record_set_id': instance.record_set_id,
                'matched_status': instance.matched_status,
                'server_date': instance.serverDate,
                'server_timestamp': instance.serverTimestamp,
                'visitortype': instance.visitorType,
                'visitcount': instance.visitCount,
                'action_type_id': instance.type
                } for instance in web_lnd_tbl_qry]

        self.insert('vs_web_inter_input', rows)
        print('Total data inserted: {}'.format(len(rows)))


    def update_web_inter_input(self):
        """Update segment_id and web_activity_type_id at vs_web_inter_input table
        """

        # TODO (ilia): needs verification

        print ('Start Updating...')

        raw_sql = """
            UPDATE
              vs_web_inter_input as inter
            INNER JOIN
              (SELECT
                  INTER_INPUT.vs_id as vs_id,
                  CONFIG.segment_id AS segment_id,
                  CONFIG.web_activity_type_id AS web_activity_type_id
              FROM
                  vs_web_client_config CONFIG,
                  vs_web_inter_input INTER_INPUT
              WHERE
                  CONFIG.action_type_id=INTER_INPUT.action_type_id)
              AS mapping
            ON
              inter.vs_id = mapping.vs_id
            SET
              inter.segment_id = mapping.segment_id,
              inter.web_activity_type_id = mapping.web_activity_type_id;
        """

        self.cursor.execute(raw_sql)
        print ('Updated {} rows'.format(len(rows)))

    # SECTION B.

    def substract_date(self, days_to_subtract):
        """Get the date minus by days_to_subtract
        """
        get_date = self.get_latest_server_date()
        d = get_date - timedelta(days=days_to_subtract)
        return datetime.strftime(d, "%Y-%m-%d")

    def get_bucket_duration(self, time_bucket_percent, time_scale_in_days):
        """Get the time duration for bucket
        """

        # if time_bucket_percent is blank set 20 as default value
        if not time_bucket_percent:
            time_bucket_percent = 20

        time_bucket_percent = time_bucket_percent / 100

        return time_bucket_percent * time_scale_in_days

    def get_latest_server_date(self):
        """Get latest server date
        """
        raw_sql = """SELECT
                        COALESCE(max(i.server_date), CURDATE()) as server_date
                    FROM vs_web_inter_input i
        """
        get_dates = self.session.execute(raw_sql).fetchall()

        for row in get_dates:
            return row.server_date

    def time_bucket_dates(self, T1, T2, T3, T4, T5):
        """Get the start and end dates of T values
        """
        web_activities = []

        T_val_start = 0
        T_val_end = 0

        T_val_start += T1
        T1_start = self.substract_date(days_to_subtract=T1)
        T1_end = self.get_latest_server_date()

        T_val_start += T2
        T_val_end = T_val_start - T2
        T2_start = self.substract_date(days_to_subtract=T_val_start)
        T2_end = self.substract_date(days_to_subtract=T_val_end)

        T_val_start += T3
        T_val_end = T_val_start - T3
        T3_start = self.substract_date(days_to_subtract=T_val_start)
        T3_end = self.substract_date(days_to_subtract=T_val_end)

        T_val_start += T4
        T_val_end = T_val_start - T4
        T4_start = self.substract_date(days_to_subtract=T_val_start)
        T4_end = self.substract_date(days_to_subtract=T_val_end)

        T_val_start += T5
        T_val_end = T_val_start - T5
        T5_start = self.substract_date(days_to_subtract=T_val_start)
        T5_end = self.substract_date(days_to_subtract=T_val_end)

        Ts = {
            'T1': {
                'start': T1_start,
                'end': T1_end
            },
            'T2': {
                'start': T2_start,
                'end': T2_end
            },
            'T3': {
                'start': T3_start,
                'end': T3_end
            },
            'T4': {
                'start': T4_start,
                'end': T4_end
            },
            'T5': {
                'start': T5_start,
                'end': T5_end
            }
        }

        for id in self.web_activity_type_ids:
            T_counter = 0

            for T in range(5):
                T_counter += 1
                TKey = 'T{}'.format(T_counter)
                T_start = Ts[TKey]['start']
                T_end = Ts[TKey]['end']

                bucket = {
                            'start': T_start,
                            'end': T_end,
                            'bucket':T_counter,
                            'activity_type_id': id
                        }

                web_activities.append(bucket)

        return web_activities

    def get_client_timebucket(self, datasource_name):
        """Get the time duration for bucket for every datasource_id

        Note: already in ENGINE folder
        """
        raw_sql = """SELECT *
                    FROM
                        vs_client_time_config
                    WHERE
                        datasource_name='{datasource_name}'
        """.format(datasource_name=datasource_name)
        time_config_qry = self.session.execute(raw_sql).fetchall()
        bucket_durations = []

        for instance in time_config_qry:
            days = instance.time_scale_in_days
            T1 = self.get_bucket_duration(instance.time_bucket_1_in_percent, days)
            T2 = self.get_bucket_duration(instance.time_bucket_2_in_percent, days)
            T3 = self.get_bucket_duration(instance.time_bucket_3_in_percent, days)
            T4 = self.get_bucket_duration(instance.time_bucket_4_in_percent, days)
            T5 = self.get_bucket_duration(instance.time_bucket_5_in_percent, days)
            bucket_durations.append(self.time_bucket_dates(T1, T2, T3, T4, T5))

        return bucket_durations

    def get_visitcount(self, activity_id, start_date, end_date):
        """Get the Sum of visitcount from vs_web_inter_input
        """

        raw_sql = """SELECT
                        SUM(visitcount) as visitcount,
                        record_set_id,
                        visitorId
                    FROM
                        vs_web_inter_input
                    WHERE
                        web_activity_type_id={activity_id}
                        AND matched_status IN ('Match', 'MATCH', 'MATCHED')
                        AND server_date BETWEEN '{start_date}' and '{end_date}'

                    GROUP BY record_set_id, visitorId, server_date, server_timestamp

                    ORDER BY record_set_id, visitorId ASC
                """.format(activity_id=activity_id,
                       start_date=start_date,
                       end_date=end_date)

        # FIXME (ilia): same issue with session - query works only with cursor
        # visitcount_qry = self.session.execute(raw_sql).fetchall()
        visitcount_qry = self.cursor.execute(raw_sql).fetchall()
        return visitcount_qry

    def process(self):
        """Process web data aggregation
        """
        timebuckets_qry = self.get_client_timebucket('WEB')
        date_processed = self.get_latest_server_date()
        self.insert_web_inter_input()
        self.update_web_inter_input()
        print (date_processed)

        for buckets in timebuckets_qry:
            for bucket in buckets:
                print (bucket)
                activity_id = bucket['activity_type_id']
                start_date = bucket['start']
                end_date = bucket['end']
                bckt = bucket['bucket']

                web_act_keys = 'web_act{}_time_bckt{}'.format(activity_id, bckt)

                visits = self.get_visitcount(activity_id, start_date, end_date)
                for visit in visits:
                    record_set_id = visit.record_set_id
                    visitorId = visit.visitorId

                    sql = """SELECT *
                            FROM vs_web_final_input
                            WHERE record_set_id='{record_set_id}'
                            AND visitorId='{visitorId}'
                            AND date_processed='{date_processed}'
                    """.format(record_set_id=record_set_id,
                               visitorId=visitorId,
                               date_processed=date_processed)
                    web_final = self.session.execute(sql).fetchall()
                    row = {
                        web_act_keys: visit.visitcount,
                        'record_set_id': record_set_id,
                        'visitorId': visitorId,
                        'date_processed': date_processed
                    }

                    if web_final:
                        sql_update = """UPDATE vs_web_final_input
                                        SET
                                            {field}={value}
                                        WHERE record_set_id='{record_set_id}'
                                        AND visitorId='{visitorId}'
                                        AND date_processed='{date_processed}'
                        """.format(field=web_act_keys,
                                   value=visit.visitcount,
                                   record_set_id=record_set_id,
                                   visitorId=visitorId,
                                   date_processed=date_processed
                                )
                        self.cursor.execute(sql_update)
                    else:
                        self.insert('vs_web_final_input', [row,])
