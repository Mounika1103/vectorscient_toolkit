import pytest

from ..sources.data import DatabaseTableSource
from ..sources.data import WebAnalyticsSource
from ..sources.data import CrmReportSource
from ..sources.data import DBConnection
from ..exceptions.exc import DataSourceError
from ..utils import data_path


def assert_source_is_valid(source):
    not_ready_before_preparation = not source.ready
    source.prepare()
    ready_after_preparation = source.ready and source.data is not None

    assert not_ready_before_preparation
    assert ready_after_preparation
    assert not source.data.empty


def test_web_analytics_data_source():
    data_url = (
        "http://client-1.vantena.com/analytics/?"
        "module=API&"
        "method=Live.getLastVisitsDetails&"
        "idSite=1&"
        "period=day&"
        "format=JSON&"
        "token_auth=ead43a5aa62a8cea302aa9e32943db99"
    )
    ip_data_local_path = data_path("ipdata.bin")
    valid_source = WebAnalyticsSource(
        data_url, database_path=ip_data_local_path)

    assert_source_is_valid(valid_source)

    non_existent_url = "http://127.0.0.1/"
    invalid_source = WebAnalyticsSource(non_existent_url, database_path="")

    with pytest.raises(DataSourceError):
        invalid_source.prepare()


def test_crm_report_data_source():
    crm_report_file_path = data_path("crm.xlsx")
    valid_source = CrmReportSource(crm_report_file_path)

    assert_source_is_valid(valid_source)

    invalid_source = CrmReportSource("there_is_no_such_place")

    with pytest.raises(DataSourceError):
        invalid_source.prepare()


def test_database_table_data_source():
    conn = DBConnection(database='cl1_clone')
    table_name = 'VS_MERGED_WEB_CRM_MASTER'

    valid_source = DatabaseTableSource(table_name, conn,
                                       columns=['product', 'revenue'])

    assert_source_is_valid(valid_source)

    invalid_source = DatabaseTableSource("non_existent_table", conn)

    with pytest.raises(DataSourceError):
        invalid_source.prepare()
