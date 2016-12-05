from sqlalchemy import (
                        Column,
                        Integer,
                        String,
                        Text,
                        Float,
                        ForeignKey,
                        Date,
                        DateTime,
                        Boolean,
                        Text,
                        UniqueConstraint
                    )

from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()

#####################################################################
# PREDICTION TABLES
#####################################################################


class ClusterClass(Base):
    __tablename__ = 'cluster_class'
    id = Column(Integer, primary_key=True)
    cluster_class = Column(Integer)
    cluster_class_name = Column(String(50))


class CommonBaseMixin(object):
    id = Column(Integer, primary_key=True)

    # Web
    web_act1_time_bckt1 = Column(Float)
    web_act1_time_bckt2 = Column(Float)
    web_act1_time_bckt3 = Column(Float)
    web_act1_time_bckt4 = Column(Float)
    web_act1_time_bckt5 = Column(Float)

    web_act2_time_bckt1 = Column(Float)
    web_act2_time_bckt2 = Column(Float)
    web_act2_time_bckt3 = Column(Float)
    web_act2_time_bckt4 = Column(Float)
    web_act2_time_bckt5 = Column(Float)

    web_act3_time_bckt1 = Column(Float)
    web_act3_time_bckt2 = Column(Float)
    web_act3_time_bckt3 = Column(Float)
    web_act3_time_bckt4 = Column(Float)
    web_act3_time_bckt5 = Column(Float)

    # Helpdesk
    num_support_req_time_bckt1 = Column(Float)
    num_support_req_time_bckt2 = Column(Float)
    num_support_req_time_bckt3 = Column(Float)
    num_support_req_time_bckt4 = Column(Float)
    num_support_req_time_bckt5 = Column(Float)

    num_pos_chat_time_bckt1 = Column(Float)
    num_pos_chat_time_bckt2 = Column(Float)
    num_pos_chat_time_bckt3 = Column(Float)
    num_pos_chat_time_bckt4 = Column(Float)
    num_pos_chat_time_bckt5 = Column(Float)

    num_neg_chat_time_bckt1 = Column(Float)
    num_neg_chat_time_bckt2 = Column(Float)
    num_neg_chat_time_bckt3 = Column(Float)
    num_neg_chat_time_bckt4 = Column(Float)
    num_neg_chat_time_bckt5 = Column(Float)

    # CRM
    num_pos_email_time_bckt1 = Column(Float)
    num_pos_email_time_bckt2 = Column(Float)
    num_pos_email_time_bckt3 = Column(Float)
    num_pos_email_time_bckt4 = Column(Float)
    num_pos_email_time_bckt5 = Column(Float)

    num_neg_email_time_bckt1 = Column(Float)
    num_neg_email_time_bckt2 = Column(Float)
    num_neg_email_time_bckt3 = Column(Float)
    num_neg_email_time_bckt4 = Column(Float)
    num_neg_email_time_bckt5 = Column(Float)

    # SM
    num_pos_tweets_time_bckt1 = Column(Float)
    num_pos_tweets_time_bckt2 = Column(Float)
    num_pos_tweets_time_bckt3 = Column(Float)
    num_pos_tweets_time_bckt4 = Column(Float)
    num_pos_tweets_time_bckt5 = Column(Float)

    num_neg_tweets_time_bckt1 = Column(Float)
    num_neg_tweets_time_bckt2 = Column(Float)
    num_neg_tweets_time_bckt3 = Column(Float)
    num_neg_tweets_time_bckt4 = Column(Float)
    num_neg_tweets_time_bckt5 = Column(Float)

    # CRM
    revenue = Column(Float)
    num_days_ss1 = Column(Float)
    num_days_ss2 = Column(Float)
    num_days_ss3 = Column(Float)
    num_days_ss4 = Column(Float)
    num_days_ss5 = Column(Float)
    num_days_ss6 = Column(Float)
    num_days_ss7 = Column(Float)
    num_days_ss8 = Column(Float)
    num_days_ss9 = Column(Float)
    num_days_ss10 = Column(Float)

    current_ss = Column(Float)
    days_in_curr_ss = Column(Float)

    # Helpdesk
    opentasks_time_bckt1 = Column(Float)
    opentasks_time_bckt2 = Column(Float)
    opentasks_time_bckt3 = Column(Float)
    opentasks_time_bckt4 = Column(Float)
    opentasks_time_bckt5 = Column(Float)

    num_closed_tsk_time_bckt1 = Column(Float)
    num_closed_tsk_time_bckt2 = Column(Float)
    num_closed_tsk_time_bckt3 = Column(Float)
    num_closed_tsk_time_bckt4 = Column(Float)
    num_closed_tsk_time_bckt5 = Column(Float)

    # CRM
    num_sp_pos_com_time_bckt1 = Column(Float)
    num_sp_pos_com_time_bckt2 = Column(Float)
    num_sp_pos_com_time_bckt3 = Column(Float)
    num_sp_pos_com_time_bckt4 = Column(Float)
    num_sp_pos_com_time_bckt5 = Column(Float)

    num_sp_neg_com_time_bckt1 = Column(Float)
    num_sp_neg_com_time_bckt2 = Column(Float)
    num_sp_neg_com_time_bckt3 = Column(Float)
    num_sp_neg_com_time_bckt4 = Column(Float)
    num_sp_neg_com_time_bckt5 = Column(Float)

    # Make a separate table
    cluster_class_by_stage = Column(Integer)
    cluster_class_name_by_stage = Column(String(50))

    @declared_attr
    def cluster_class(Base):
        return Column(Integer, ForeignKey('cluster_class.id'))


class ClusteredFileBaseMixin(CommonBaseMixin):
    customer_id = Column(String(200))
    customer_name = Column(String(200))
    opportunity_id = Column(String(200))
    opportunity_value = Column(String(200))
    product = Column(String(200))
    qty = Column(Integer)
    price = Column(Float)
    # Just added
    record_set_id = Column(Integer)
    matching_key = Column(Integer)
    matched_status = Column(String(200))

    earliest_year = Column(Integer)
    latest_year = Column(Integer)
    pred_run_date = Column(Date, nullable=True)


class QFIBaseMixin(object):
    # QFI
    distance_to_centroid = Column(Float, default=0)
    qfi_overall = Column(Float, default=0)
    qfi_web_intra = Column(Float, default=0)
    qfi_hd_intra = Column(Float, default=0)
    qfi_crm_intra = Column(Float, default=0)
    qfi_sh_intra = Column(Float, default=0)
    qfi_sm_intra = Column(Float, default=0)
    qfi_web_inter = Column(Float, default=0)
    qfi_hd_inter = Column(Float, default=0)
    qfi_crm_inter = Column(Float, default=0)
    qfi_sh_inter = Column(Float, default=0)
    qfi_sm_inter = Column(Float, default=0)


class SumBaseMixin(object):
    overall_sum_of_shipped_qty = Column(Float)
    overall_sum_of_tot_net_amt = Column(Float)
    number_of_periods_sold = Column(Float)
    total_periods_of_activity = Column(Float)
    percent_of_time_product_sold = Column(Float)


class ClusteredFileNewPros(ClusteredFileBaseMixin, QFIBaseMixin, Base):
    __tablename__ = 'clustered_file_new_pros'


class ClusteredFileExisPros(SumBaseMixin, ClusteredFileBaseMixin, QFIBaseMixin, Base):
    __tablename__ = 'clustered_file_exis_pros'


class ClusteredFileNewProsNorm(ClusteredFileBaseMixin, Base):
    __tablename__ = 'clustered_file_new_pros_norm'


class ClusteredFileExisProsNorm(SumBaseMixin, ClusteredFileBaseMixin, Base):
    __tablename__ = 'clustered_file_exis_pros_norm'


class CentroidsNewPros(CommonBaseMixin, Base):
    __tablename__ = 'centroids_new_pros'
    record_id = Column(Integer)
    opportunity_value = Column(Float)
    pred_run_date = Column(Date, nullable=True)


class CentroidsExisPros(SumBaseMixin, CommonBaseMixin, Base):
    __tablename__ = 'centroids_exis_pros'
    record_id = Column(Integer)
    opportunity_value = Column(Float)
    pred_run_date = Column(Date, nullable=True)


class MasterPredictorsLookUp(Base):
    __tablename__ = 'master_predictors_lookup'
    id = Column(Integer, primary_key=True)
    feature_id = Column(Integer, default=0)
    feature_name = Column(String(100))
    feature_type = Column(String(100))
    feature_sub_type = Column(String(100))
    data_source_cat = Column(String(100))
    time_bucket = Column(Integer, default=0)
