"""
Hotfix to merge CRM and WEB tables using RL algorithm results and insert
processed records into main clustering table.
"""


from ..sources.connection import DBConnection


class WebCrmDataMerger(DBConnection):

    def __init__(self, **kwargs):
        self.db = kwargs.get('database')
        super(WebCrmDataMerger, self).__init__(database=self.db)

    def merge_web_final_input_and_crm_structured_to_create_master_table(self):
        """
        Creates records for VS_MERGED_WEB_CRM_MASTER table by merging
        VS_WEB_FINAL_INPUT and CRM_STRUCTURED_LND_TBL tables.
        """
        insert_sql = """
        INSERT INTO VS_MERGED_WEB_CRM_MASTER (
            # vs_id,
            opportunity_id
            ,customer_id
            ,customer_name
            ,Opportunity_name
            ,Product
            ,Product_id
            ,qty
            ,price
            ,earliest_year
            ,latest_year
            ,Revenue
            ,num_days_ss1
            ,num_days_ss2
            ,num_days_ss3
            ,num_days_ss4
            ,num_days_ss5
            ,num_days_ss6
            ,num_days_ss7
            ,num_days_ss8
            ,num_days_ss9
            ,num_days_ss10
            ,current_ss
            ,days_in_curr_ss
            ,record_set_id_crm
            ,customer_billing_address
            ,domain
            ,matching_key
            ,matched_status
            ,Customer_billing_addr_latitude
            ,Customer_billing_addr_longitude
            ,CRM_master_Refreshed_date)
        SELECT
            # <auto id>,
            opportunity_id
            ,customer_id
            ,customer_name
            ,Opportunity_name
            ,Product
            ,Product_id
            ,qty
            ,price
            ,EarliestYear
            ,LatestYear
            ,Revenue
            ,num_days_ss1
            ,num_days_ss2
            ,num_days_ss3
            ,num_days_ss4
            ,num_days_ss5
            ,num_days_ss6
            ,num_days_ss7
            ,num_days_ss8
            ,num_days_ss9
            ,num_days_ss10
            ,current_ss
            ,days_in_curr_ss
            ,record_set_id
            ,customer_billing_address
            ,domain
            ,matching_key
            ,matched_status
            ,Customer_billing_addr_latitude
            ,Customer_billing_addr_longitude
            ,Last_Refreshed_date
        FROM
            CRM_structured_lnd_tbl CRM

        WHERE CRM.Last_Refreshed_date = (
            SELECT MAX(Last_Refreshed_date) FROM CRM_structured_lnd_tbl);
        """

        update_sql = """
        UPDATE VS_MERGED_WEB_CRM_MASTER vs
        INNER JOIN
            (SELECT
                 CRM_STRUCTURED.customer_id
                ,CRM_STRUCTURED.opportunity_id
                ,CRM_STRUCTURED.customer_name
                ,CRM_STRUCTURED.customer_billing_address
                ,CRM_STRUCTURED.record_set_id "CRM_RECORD_SET_ID"
                ,RL.record_set_id_web "RL_WEB"
                ,RL.record_set_id_CRM "RL_CRM"
                ,CRM_STRUCTURED.domain
                ,RL.matched_status
                ,RL.matching_key
                ,WEB.visitorId
                ,WEB.web_act1_time_bckt1
                ,WEB.web_act1_time_bckt2
                ,WEB.web_act1_time_bckt3
                ,WEB.web_act1_time_bckt4
                ,WEB.web_act1_time_bckt5
                ,WEB.web_act2_time_bckt1
                ,WEB.web_act2_time_bckt2
                ,WEB.web_act2_time_bckt3
                ,WEB.web_act2_time_bckt4
                ,WEB.web_act2_time_bckt5
                ,WEB.web_act3_time_bckt1
                ,WEB.web_act3_time_bckt2
                ,WEB.web_act3_time_bckt3
                ,WEB.web_act3_time_bckt4
                ,WEB.web_act3_time_bckt5
                ,WEB.date_processed
                ,MAX(CRM_STRUCTURED.Last_Refreshed_date) "CRM_REFRESHED_DATE"
            FROM
                CRM_structured_lnd_tbl CRM_STRUCTURED

            LEFT JOIN vs_web_crm_rl RL
                ON CRM_STRUCTURED.record_set_id = RL.record_set_id_CRM

            LEFT JOIN (SELECT * FROM vs_web_final_input
                       GROUP BY record_set_id, visitorId) WEB
                ON RL.record_set_id_web = WEB.record_set_id

            WHERE RL.matched_status = 'Match'
            GROUP BY CRM_STRUCTURED.customer_id, CRM_STRUCTURED.opportunity_id
            ) AS a
        ON vs.customer_id = a.customer_id
        AND vs.opportunity_id = a.opportunity_id
        AND vs.matching_key = a.matching_key
        AND vs.record_set_id_crm = a.RL_CRM
        AND vs.crm_master_refreshed_date = a.CRM_REFRESHED_DATE
        SET
             vs.visitor_id          = a.visitorId
            ,vs.web_act1_time_bckt1 = a.web_act1_time_bckt1
            ,vs.web_act1_time_bckt2 = a.web_act1_time_bckt2
            ,vs.web_act1_time_bckt3 = a.web_act1_time_bckt3
            ,vs.web_act1_time_bckt4 = a.web_act1_time_bckt4
            ,vs.web_act1_time_bckt5 = a.web_act1_time_bckt5
            ,vs.web_act2_time_bckt1 = a.web_act2_time_bckt1
            ,vs.web_act2_time_bckt2 = a.web_act2_time_bckt2
            ,vs.web_act2_time_bckt3 = a.web_act2_time_bckt3
            ,vs.web_act2_time_bckt4 = a.web_act2_time_bckt4
            ,vs.web_act2_time_bckt5 = a.web_act2_time_bckt5
            ,vs.web_act3_time_bckt1 = a.web_act3_time_bckt1
            ,vs.web_act3_time_bckt2 = a.web_act3_time_bckt2
            ,vs.web_act3_time_bckt3 = a.web_act3_time_bckt3
            ,vs.web_act3_time_bckt4 = a.web_act3_time_bckt4
            ,vs.web_act3_time_bckt5 = a.web_act3_time_bckt5
            ,vs.web_processed_date  = a.date_processed
            ,vs.record_set_id_web   = a.RL_WEB
        """

        self.cursor.execute(insert_sql)
        self.cursor.execute(update_sql)

    def merge_master_table_and_unstructured_input_to_create_clustered_table(self):
        """
        Creates records for CLUSTERED_FILE_NEW_PROS table by merging
        VS_MERGED_WEB_CRM_MASTER and CRM_UNSTRUCTURED_INPUT tables.
        """
        insert_sql = """
        INSERT INTO clustered_file_new_pros
            (#vs_id,
            web_act1_time_bckt1
            ,web_act1_time_bckt2
            ,web_act1_time_bckt3
            ,web_act1_time_bckt4
            ,web_act1_time_bckt5
            ,web_act2_time_bckt1
            ,web_act2_time_bckt2
            ,web_act2_time_bckt3
            ,web_act2_time_bckt4
            ,web_act2_time_bckt5
            ,web_act3_time_bckt1
            ,web_act3_time_bckt2
            ,web_act3_time_bckt3
            ,web_act3_time_bckt4
            ,web_act3_time_bckt5
            ,revenue
            ,num_days_ss1
            ,num_days_ss2
            ,num_days_ss3
            ,num_days_ss4
            ,num_days_ss5
            ,num_days_ss6
            ,current_ss
            ,days_in_curr_ss
            ,customer_id
            ,customer_name
            ,opportunity_id
            ,opportunity_value
            ,product
            ,qty
            ,price
            ,earliest_year
            ,latest_year
            ,num_days_ss7
            ,num_days_ss8
            ,num_days_ss9
            ,num_days_ss10
            ,record_set_id
            ,matching_key
            ,matched_status)
        SELECT
            #<auto id>,
            web_act1_time_bckt1
            ,web_act1_time_bckt2
            ,web_act1_time_bckt3
            ,web_act1_time_bckt4
            ,web_act1_time_bckt5
            ,web_act2_time_bckt1
            ,web_act2_time_bckt2
            ,web_act2_time_bckt3
            ,web_act2_time_bckt4
            ,web_act2_time_bckt5
            ,web_act3_time_bckt1
            ,web_act3_time_bckt2
            ,web_act3_time_bckt3
            ,web_act3_time_bckt4
            ,web_act3_time_bckt5
            ,revenue
            ,num_days_ss1
            ,num_days_ss2
            ,num_days_ss3
            ,num_days_ss4
            ,num_days_ss5
            ,num_days_ss6
            ,current_ss
            ,days_in_curr_ss
            ,customer_id
            ,customer_name
            ,opportunity_id
            ,opportunity_value
            ,product
            ,qty
            ,price
            ,earliest_year
            ,latest_year
            ,num_days_ss7
            ,num_days_ss8
            ,num_days_ss9
            ,num_days_ss10
            ,record_set_id_crm
            ,matching_key
            ,matched_status
        FROM
            VS_MERGED_WEB_CRM_MASTER;
        """

        update_sql = """
        UPDATE clustered_file_new_pros vs
        INNER JOIN
            CRM_unstructured_input crm
            ON vs.customer_id = crm.account_id
            AND vs.opportunity_id = crm.opportunity_id
        SET
            vs.num_support_req_time_bckt1  = crm.num_support_req_time_bckt1
            ,vs.num_support_req_time_bckt2 = crm.num_support_req_time_bckt2
            ,vs.num_support_req_time_bckt3 = crm.num_support_req_time_bckt3
            ,vs.num_support_req_time_bckt4 = crm.num_support_req_time_bckt4
            ,vs.num_support_req_time_bckt5 = crm.num_support_req_time_bckt5
            ,vs.num_pos_chat_time_bckt1    = crm.num_pos_chat_time_bckt1
            ,vs.num_pos_chat_time_bckt2    = crm.num_pos_chat_time_bckt2
            ,vs.num_pos_chat_time_bckt3    = crm.num_pos_chat_time_bckt3
            ,vs.num_pos_chat_time_bckt4    = crm.num_pos_chat_time_bckt4
            ,vs.num_pos_chat_time_bckt5    = crm.num_pos_chat_time_bckt5
            ,vs.num_neg_chat_time_bckt1    = crm.num_neg_chat_time_bckt1
            ,vs.num_neg_chat_time_bckt2    = crm.num_neg_chat_time_bckt2
            ,vs.num_neg_chat_time_bckt3    = crm.num_neg_chat_time_bckt3
            ,vs.num_neg_chat_time_bckt4    = crm.num_neg_chat_time_bckt4
            ,vs.num_neg_chat_time_bckt5    = crm.num_neg_chat_time_bckt5
            ,vs.num_pos_email_time_bckt1   = crm.num_pos_email_time_bckt1
            ,vs.num_pos_email_time_bckt2   = crm.num_pos_email_time_bckt2
            ,vs.num_pos_email_time_bckt3   = crm.num_pos_email_time_bckt3
            ,vs.num_pos_email_time_bckt4   = crm.num_pos_email_time_bckt4
            ,vs.num_pos_email_time_bckt5   = crm.num_pos_email_time_bckt5
            ,vs.num_neg_email_time_bckt1   = crm.num_neg_email_time_bckt1
            ,vs.num_neg_email_time_bckt2   = crm.num_neg_email_time_bckt2
            ,vs.num_neg_email_time_bckt3   = crm.num_neg_email_time_bckt3
            ,vs.num_neg_email_time_bckt4   = crm.num_neg_email_time_bckt4
            ,vs.num_neg_email_time_bckt5   = crm.num_neg_email_time_bckt5
            ,vs.num_pos_tweets_time_bckt1  = crm.num_pos_tweets_time_bckt1
            ,vs.num_pos_tweets_time_bckt2  = crm.num_pos_tweets_time_bckt2
            ,vs.num_pos_tweets_time_bckt3  = crm.num_pos_tweets_time_bckt3
            ,vs.num_pos_tweets_time_bckt4  = crm.num_pos_tweets_time_bckt4
            ,vs.num_pos_tweets_time_bckt5  = crm.num_pos_tweets_time_bckt5
            ,vs.num_neg_tweets_time_bckt1  = crm.num_neg_tweets_time_bckt1
            ,vs.num_neg_tweets_time_bckt2  = crm.num_neg_tweets_time_bckt2
            ,vs.num_neg_tweets_time_bckt3  = crm.num_neg_tweets_time_bckt3
            ,vs.num_neg_tweets_time_bckt4  = crm.num_neg_tweets_time_bckt4
            ,vs.num_neg_tweets_time_bckt5  = crm.num_neg_tweets_time_bckt5
            ,vs.opentasks_time_bckt1       = crm.opentasks_time_bckt1
            ,vs.opentasks_time_bckt2       = crm.opentasks_time_bckt2
            ,vs.opentasks_time_bckt3       = crm.opentasks_time_bckt3
            ,vs.opentasks_time_bckt4       = crm.opentasks_time_bckt4
            ,vs.opentasks_time_bckt5       = crm.opentasks_time_bckt5
            ,vs.num_closed_tsk_time_bckt1  = crm.num_closed_tsk_time_bckt1
            ,vs.num_closed_tsk_time_bckt2  = crm.num_closed_tsk_time_bckt2
            ,vs.num_closed_tsk_time_bckt3  = crm.num_closed_tsk_time_bckt3
            ,vs.num_closed_tsk_time_bckt4  = crm.num_closed_tsk_time_bckt4
            ,vs.num_closed_tsk_time_bckt5  = crm.num_closed_tsk_time_bckt5
            ,vs.num_sp_pos_com_time_bckt1  = crm.num_sp_pos_com_time_bckt1
            ,vs.num_sp_pos_com_time_bckt2  = crm.num_sp_pos_com_time_bckt2
            ,vs.num_sp_pos_com_time_bckt3  = crm.num_sp_pos_com_time_bckt3
            ,vs.num_sp_pos_com_time_bckt4  = crm.num_sp_pos_com_time_bckt4
            ,vs.num_sp_pos_com_time_bckt5  = crm.num_sp_pos_com_time_bckt5
            ,vs.num_sp_neg_com_time_bckt1  = crm.num_sp_neg_com_time_bckt1
            ,vs.num_sp_neg_com_time_bckt2  = crm.num_sp_neg_com_time_bckt2
            ,vs.num_sp_neg_com_time_bckt3  = crm.num_sp_neg_com_time_bckt3
            ,vs.num_sp_neg_com_time_bckt4  = crm.num_sp_neg_com_time_bckt4
            ,vs.num_sp_neg_com_time_bckt5  = crm.num_sp_neg_com_time_bckt5
        WHERE vs.pred_run_date IS NULL;
        """

        self.cursor.execute(insert_sql)
        self.cursor.execute(update_sql)


if __name__ == '__main__':
    merger = WebCrmDataMerger(database='cl1_clone')
    merger.merge_web_final_input_and_crm_structured_to_create_master_table()
    merger.merge_master_table_and_unstructured_input_to_create_clustered_table()
