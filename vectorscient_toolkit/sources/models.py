from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Date,
    Float
)
from sqlalchemy.ext.declarative import declarative_base

ModelBase = declarative_base()


class WebLanding(ModelBase):

    __tablename__ = 'vs_web_lnd_tbl'

    id = Column(Integer, primary_key=True)
    actionDetails_generationTime = Column(String(50), nullable=True)
    actionDetails_icon = Column(String(45), nullable=True)
    actionDetails_pageTitle = Column(String(500), nullable=False)
    actionDetails_type = Column(String(45))
    actionDetails_url = Column(String(500), nullable=True)
    
    matching_key = Column(String(45), nullable=True)
    record_set_id = Column(Integer, nullable=False)
    web_address = Column(String(250), nullable=True)
    Domain = Column(String(45), nullable=False)
    Isp = Column(String(45), nullable=False)
    Latitude_derived = Column(String(45), nullable=True)
    Longitude_derived = Column(String(45), nullable=True)
    matched_status = Column(String(20), nullable=True)
    serverDate = Column(String(45), nullable=True)
    serverTimestamp = Column(String(45), nullable=True)
    visitorId = Column(String(45), nullable=True)
    visitorType = Column(String(45), nullable=True)
    web_city = Column(String(45), nullable=True)
    web_country = Column(String(45), nullable=True)
    ip = Column(String(45), nullable=True)
    insert_by = Column(String(45), nullable=True)
    insert_date = Column(String(45), nullable=True)
    
    actionDetails_pageIdAction = Column(String(45), nullable=True)
    actionDetails_pageId = Column(String(45), nullable=True)
    actionDetails_serverTimePretty = Column(String(45), nullable=True)
    actionDetails_siteSearchKeyword = Column(String(45), nullable=True)
    actionDetails_timeSpent = Column(String(45), nullable=True)
    actionDetails_timeSpentPretty = Column(String(45), nullable=True)
    actionDetails_timestamp = Column(String(45), nullable=True)
    actions = Column(String(45), nullable=True)
    
    browser = Column(String(45), nullable=True)
    browserCode = Column(String(45), nullable=True)
    browserFamily = Column(String(45), nullable=True)
    browserFamilyDescription = Column(String(45), nullable=True)
    browserIcon = Column(String(500), nullable=True)
    browserName = Column(String(45), nullable=True)
    browserVersion = Column(String(45), nullable=True)
    
    city = Column(String(45), nullable=True)
    continent = Column(String(45), nullable=True)
    continentCode = Column(String(45), nullable=True)
    country = Column(String(45), nullable=True)
    countryCode = Column(String(45), nullable=True)
    countryFlag = Column(String(45), nullable=True)
    daysSinceFirstVisit = Column(String(45), nullable=True)
    daysSinceLastEcommerceOrder = Column(String(45), nullable=True)
    daysSinceLastVisit = Column(String(45), nullable=True)
    
    deviceBrand = Column(String(45), nullable=True)
    deviceModel = Column(String(45), nullable=True)
    deviceType = Column(String(45), nullable=True)
    deviceTypeIcon = Column(String(500), nullable=True)
    
    events = Column(String(45), nullable=True)
    firstActionTimestamp = Column(String(45), nullable=True)
    goalConversions = Column(String(45), nullable=True)
    idSite = Column(String(45), nullable=True)
    idVisit = Column(String(45), nullable=True)
    language = Column(String(45), nullable=True)
    languageCode = Column(String(45), nullable=True)
    lastActionDateTime = Column(String(45), nullable=True)
    lastActionTimestamp = Column(String(45), nullable=True)
    latitude = Column(String(45), nullable=True)
    location = Column(String(45), nullable=True)
    longitude = Column(String(45), nullable=True)
    
    operatingSystem = Column(String(45), nullable=True)
    operatingSystemCode = Column(String(45), nullable=True)
    operatingSystemIcon = Column(String(45), nullable=True)
    operatingSystemName = Column(String(45), nullable=True)
    operatingSystemVersion = Column(String(45), nullable=True)
    
    plugins = Column(String(500), nullable=True)
    pluginsIcons = Column(String(45), nullable=True)
    pluginsIcons_pluginIcon = Column(String(500), nullable=True)
    pluginsIcons_pluginName = Column(String(45), nullable=True)
    
    referrerKeyword = Column(String(500), nullable=True)
    referrerKeywordPosition = Column(String(45), nullable=True)
    referrerName = Column(String(200), nullable=True)
    referrerSearchEngineIcon = Column(String(500), nullable=True)
    referrerSearchEngineUrl = Column(String(45), nullable=True)
    referrerType = Column(String(45), nullable=True)
    referrerTypeName = Column(String(45), nullable=True)
    referrerUrl = Column(Text(), nullable=True)
    
    region = Column(String(45), nullable=True)
    regionCode = Column(String(45), nullable=True)
    resolution = Column(String(45), nullable=True)
    searches = Column(String(45), nullable=True)
    
    serverDatePretty = Column(String(45), nullable=True)
    serverDatePrettyFirstAction = Column(String(45), nullable=True)
    serverTimePretty = Column(String(45), nullable=True)
    serverTimePrettyFirstAction = Column(String(45), nullable=True)
    
    siteCurrency = Column(String(45), nullable=True)
    siteCurrencySymbol = Column(String(45), nullable=True)
    
    userId = Column(String(45), nullable=True)
    visitConverted = Column(String(45), nullable=True)
    visitCount = Column(String(45), nullable=True)
    visitDuration = Column(String(45), nullable=True)
    visitDurationPretty = Column(String(45), nullable=True)
    visitEcommerceStatus = Column(String(45), nullable=True)
    visitEcommerceStatusIcon = Column(String(45), nullable=True)
    visitIp = Column(String(45), nullable=True)
    visitLocalHour = Column(String(45), nullable=True)
    visitLocalTime = Column(String(45), nullable=True)
    visitServerHour = Column(String(45), nullable=True)
    visitorTypeIcon = Column(String(45), nullable=True)
    
    VS_WEB_FINAL_STG1_crm_acct_name = Column(String(400), nullable=False)
    VS_WEB_FINAL_STG1_crm_opportunity_id = Column(String(45), nullable=False)
    VS_WEB_FINAL_STG1_serverDate = Column(String(45), nullable=False)
    VS_WEB_FINAL_STG1_serverTimestamp = Column(String(45), nullable=False)


class CrmLanding(ModelBase):
    __tablename__ = 'CRM_structured_lnd_tbl'

    vs_id = Column(Integer, primary_key=True)
    customer_name = Column(String(300), nullable=True)
    Customer_source = Column(String(300), nullable=True)
    Customer_industry = Column(String(300), nullable=True)
    Customer_last_activity_date = Column(DateTime, nullable=True)
    NaicsCode = Column(String(300), nullable=True)
    NaicsDesc = Column(String(300), nullable=True)
    Customer_num_employees = Column(String(300), nullable=True)
    Customer_account_ownership = Column(String(300), nullable=True)
    customer_rating = Column(String(300), nullable=True)
    Sic = Column(String(300), nullable=True)
    sicDesc = Column(String(300), nullable=True)
    TickerSymbol = Column(String(300), nullable=True)
    customer_type = Column(String(300), nullable=True)
    opportunity_id = Column(String(300), nullable=True)
    customer_id = Column(String(300), nullable=True)
    Opportunity_name = Column(String(300), nullable=True)
    Opportunity_type = Column(String(300), nullable=True)
    Opportunity_LeadSource = Column(String(300), nullable=True)
    Opportunity_CloseDate = Column(DateTime, nullable=True)
    Product = Column(String(300), nullable=True)
    Product_id = Column(String(300), nullable=True)
    Product_description = Column(String(300), nullable=True)
    Product_family = Column(String(300), nullable=True)
    Product_name = Column(String(300), nullable=True)
    qty = Column(String(300), nullable=True)
    price = Column(String(300), nullable=True)
    EarliestYear = Column(String(300), nullable=True)
    LatestYear = Column(String(300), nullable=True)
    Revenue = Column(String(300), nullable=True)

    num_days_ss1 = Column(String(300), nullable=True)
    num_days_ss2 = Column(String(300), nullable=True)
    num_days_ss3 = Column(String(300), nullable=True)
    num_days_ss4 = Column(String(300), nullable=True)
    num_days_ss5 = Column(String(300), nullable=True)
    num_days_ss6 = Column(String(300), nullable=True)
    num_days_ss7 = Column(String(300), nullable=True)
    num_days_ss8 = Column(String(300), nullable=True)
    num_days_ss9 = Column(String(300), nullable=True)
    num_days_ss10 = Column(String(300), nullable=True)
    current_ss = Column(String(300), nullable=True)
    days_in_curr_ss = Column(String(300), nullable=True)

    opportunity_value = Column(String(300), nullable=True)
    record_set_id = Column(Integer, nullable=True)

    customer_billing_address = Column(String(300), nullable=True)
    domain = Column(String(300), nullable=True)

    matching_key = Column(String(300), nullable=True)
    matched_status = Column(String(300), nullable=True)
    Customer_billing_addr_latitude = Column(Float, nullable=True)
    Customer_billing_addr_longitude = Column(Float, nullable=True)

    Last_Refreshed_date = Column(DateTime, nullable=True)


class WebCrmRL(ModelBase):

    __tablename__ = 'vs_web_crm_rl'

    matching_key = Column(String(45), primary_key=True)
    web_domain = Column(String(45), nullable=True)
    crm_domain = Column(String(45), nullable=True)
    web_isp = Column(String(45), nullable=True)
    crm_account_name = Column(String(45), nullable=True)
    web_address = Column(String(500), nullable=True)
    crm_address = Column(String(45), nullable=True)
    matched_status = Column(String(45), nullable=True)
    crm_acct_id = Column(String(45), nullable=True)
    web_visit_date = Column(DateTime(), nullable=True)
    record_set_id_web = Column(Integer())
    record_set_id_CRM = Column(Integer())
    vs_user_id = Column(String(45), nullable=True)
    username = Column(String(45), nullable=True)
    last_modified_by = Column(String(45), nullable=True)
    last_modified_date = Column(DateTime(), nullable=True)
    active_flag = Column(String(45), nullable=True)
    effective_start_date = Column(DateTime(), nullable=True)
    effective_end_date = Column(DateTime(), nullable=True)


class WebFinalInput(ModelBase):
    __tablename__ = 'vs_web_final_input'
    vs_id = Column(Integer, primary_key=True)
    record_set_id = Column(Integer, nullable=True)
    visitorId = Column(String(200), nullable=False)
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

    date_processed = Column(Date, nullable=True)
