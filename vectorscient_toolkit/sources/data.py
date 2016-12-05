"""
Contains data provides to be used in records linkage process.
"""

from collections import Sequence
import itertools
import logging
import abc

import pandas as pd
import numpy as np
import IP2Location

from ..utils import try_to_request_json_data, try_to_read_local_data, validate_ip
from ..exceptions.exc import IpInfoError, DataSourceError
from ..utils import shallow_flatten, resolve_lat_long
from ..sources.connection import DBConnection


log = logging.getLogger(__name__)
log.setLevel(level=logging.DEBUG)


class IpInfoProvider(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_info(self, ip_address) -> dict:
        if validate_ip(ip_address):
            return {"IP": ip_address}
        raise IpInfoError("Bad IP address: %s" % ip_address)


class LocalFileIpInfoProvider(IpInfoProvider):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.database = IP2Location.IP2Location()
        self.database.open(self.file_path)

    def get_info(self, ip_address: str) -> dict:
        """
        Retrieves geographic information about IP address.
        """
        info = super().get_info(ip_address)
        record = self.database.get_all(ip_address)
        info.update({
            "Domain": record.domain,
            "Isp": record.isp,
            "City": record.city,
            "Country": record.country_long,
            "Longitude": record.longitude,
            "Latitude": record.latitude
        })
        return info


class DataSource(metaclass=abc.ABCMeta):
    """
    Abstract data source class. Each data source should inherit from it and
    provide actual implementation of source extraction procedure.

    Basically, any data used in linkage process should conform data source
    protocol, i.e. implement data retrieving and provide access to extracted
    data.
    """

    ERROR_PREFIX = "Data source error"

    def __init__(self, **params):
        self._name = params.get("name", "")

    @abc.abstractproperty
    def data(self):
        """
        Provides access to retrieved data.
        """
        return

    @abc.abstractmethod
    def prepare(self):
        """
        Prepares data source for following data access. Should be implemented in
        subclasses
        """
        return

    @property
    def ready(self):
        """
        Checks if data was successfully retrieved or not.
        """
        return self.data is not None

    @property
    def name(self):
        """
        Data source name. Can be used for logging purposes or as a name of
        file if data source is about to be saved into file.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        """
        self._name = value

    def format_error(self, action: str, error):
        """
        Formats source error using source type prefix and description of inner
        exception occurred.

        Args:
            action (str): An action that was taken when error occurred.
            error (str or exception): An internal error occurred during action.
        """
        message_parts = (
            "\n",
            "{prefix} ({resource_name}) - {action} error:\n",
            "{error}\n"
        )
        error_message = "".join(["\t" + p for p in message_parts])
        msg = error_message.format(
            prefix=self.ERROR_PREFIX,
            resource_name=self.name,
            action=action,
            error=str(error))
        return msg


class CsvDataSource(DataSource):
    ERROR_PREFIX = "CSV data source"
    FILE_READING = "CSV file reading"

    def __init__(self, file_name, **params):
        super().__init__(**params)
        self._data = None
        self._file_name = file_name
        self._name = self._file_name
        self._reader_config = params.get("reader_config", {})

    @property
    def data(self):
        return self._data

    def prepare(self):
        log.info("Trying to read CSV file '%s'..." % self._file_name)
        error, data = try_to_read_local_data(
            lambda: pd.read_csv(self._file_name, encoding='ISO-8859-1',**self._reader_config))

        if not error:
            self._data = data
            log.info("CSV file data prepared")

        else:
            msg = self.format_error(self.FILE_READING, error)
            log.error(msg)
            raise DataSourceError(msg)


class SpreadsheetDataSource(DataSource):
    """
    Wraps spreadsheet file with data source access interface.

    Attributes:
        file_name (str): A path to file with spreadsheet.
        sheet (str): A sheet in specified spreadsheet to be retrieved.
    """

    ERROR_PREFIX = "Spreadsheet data source"
    TABLE_READING = "table reading"

    def __init__(self, file_name: str, sheet, **params):
        super().__init__(**params)
        if not isinstance(sheet, (str, int)):
            msg = "invalid 'sheet' argument type: " \
                  "'str' or 'int' expected, got '%s'"
            raise TypeError(msg % type(sheet))
        self.file_name = file_name
        self.sheet = sheet
        self._data = None
        self._name = file_name + "_" + sheet

    @property
    def data(self):
        return self._data

    def prepare(self):
        log.info("Trying to read sheet '%s' from file '%s'...",
                 self.file_name, self.sheet)

        error, data = try_to_read_local_data(
            lambda: pd.read_excel(self.file_name, self.sheet))

        if not error:
            self._data = data
            log.info("Spreadsheet data prepared")

        else:
            msg = self.format_error(self.TABLE_READING, error)
            log.error(msg)
            raise DataSourceError(msg)

    @staticmethod
    def create_sources(file_name):
        all_sheets = pd.read_excel(file_name, sheetname=None)
        return all_sheets


class WebAnalyticsSource(DataSource):
    """
    Implements web analytics data retrieving from remote host.

    Data retrieving is extended with IP geo information resolution.

    Attributes:
        url (str): Remote resource URL.
        database_path (str): Local database with IP information.
    """

    ERROR_PREFIX = "Web analytics source"
    JSON_OBJECT_RETRIEVING = "JSON object retrieving"

    def __init__(self, url: str, **params):
        super().__init__(**params)
        self.url = url
        self.database_path = params.get("database_path", "")
        self.timeout = params.get("timeout", 5.0)
        self._data = None
        self._name = url

    @property
    def data(self):
        return self._data

    def prepare(self):
        log.info("Trying to request JSON object from %s" % self.url)
        error, data = try_to_request_json_data(self.url, self.timeout)

        if not error:
            log.info("JSON object successfully retrieved")
            log.info("Convert nested JSON into plain data frame...")
            df = self._create_dataframe_from_json(data)

            log.info("IP geo information retrieving...")
            self._data = self._fill_with_ip_geo_info(df)
            print("retrieved geo information")

            log.info("Analytics data prepared")

        else:
            msg = self.format_error(self.JSON_OBJECT_RETRIEVING, error)
            log.error(msg)
            raise DataSourceError(msg)

    @staticmethod
    def _create_dataframe_from_json(data):
        rows = [shallow_flatten(item) for item in data]
        flatten = list(itertools.chain(*rows))
        df = pd.DataFrame(flatten)
        return df

    def _fill_with_ip_geo_info(self, df):
        ip_info_provider = LocalFileIpInfoProvider(self.database_path)
        geo_info = [ip_info_provider.get_info(ip) for ip in df.visitIp.values]
        geo_df = pd.DataFrame(geo_info)
        processed = pd.concat([df, geo_df], axis=1)
        return processed

    def save_to_csv(self, file_name: str):
        if not self.ready:
            raise DataSourceError("Data source is not ready")
        self.data.to_csv(file_name)


class CrmReportSource(DataSource):
    """
    Wrapper over CRM report file.

    Extends CRM report with additional columns.

    Attributes:
        report_path (str): A path to report file.
        reader (callable): A callable object accepting path to CRM report and
            returning data frame with its contents.
        input_columns (Dict[str, str]): Column names in report spreadsheet
            containing mailing address and email.
        output_columns (Dict[str, str]): Column names in output file
            used to store extracted domain, latitude and longitude.
    """

    ERROR_PREFIX = "CRM report source"
    REPORT_READING = "CRM report reading"
    LAT_LONG_INFO_RETRIEVING = "latitude/longitude info retrieving"

    def __init__(self, report_path: str, reader=pd.read_excel, **params):
        super().__init__(**params)
        self.report_path = report_path

        input_columns = params.get("input", {})
        output_columns = params.get("output", {})

        self._input_address = input_columns.get("address", "Mailing Street")
        self._input_email = input_columns.get("email", "Email")
        self._output_domain = output_columns.get("domain", "Domain")
        self._output_lat = output_columns.get("lat", "Latitude")
        self._output_lon = output_columns.get("lon", "Longitude")
        self._reader = reader
        self._data = None
        self._name = report_path

    @property
    def data(self):
        """
        Returns data frame with processed CRM report.
        """
        return self._data

    def prepare(self):
        """
        Processes CRM report contents, adding domain, latitude and longitude
        columns.
        """
        log.info("Read data from CRM report \"%s\"..." % self.report_path)
        reader = lambda: self._reader(self.report_path)
        error, raw = try_to_read_local_data(reader)

        if error:
            msg = self.format_error(self.REPORT_READING, error)
            log.error(msg)
            raise DataSourceError(msg)

        log.info("Remove invalid (without address) rows...")
        valid_records = self._filter_out_rows_without_address(raw)

        log.info("Extract (lat, lon) coordinates...")
        error, lat_and_long_df = \
            self._try_to_fill_with_lat_lon_data(valid_records)

        if error:
            msg = self.format_error(self.LAT_LONG_INFO_RETRIEVING, error)
            log.error(msg)
            raise DataSourceError(msg)

        log.info("Extract domains from emails...")
        prepared_df = self._extract_domain_from_email(lat_and_long_df)

        log.info("CRM data prepared")
        self._data = prepared_df

    def _filter_out_rows_without_address(self, raw):
        indexer = pd.notnull(raw[self._input_address])
        valid_records = pd.DataFrame(raw[indexer])
        return valid_records

    def _try_to_fill_with_lat_lon_data(self, valid_records):
        try:
            locations = \
                valid_records[self._input_address].apply(resolve_lat_long)
        except Exception as e:
            error = str(e)
            return error, None

        lats, longs = zip(*[(loc.lat, loc.lon) for loc in locations.values])
        lat_lon_records = pd.DataFrame(valid_records)
        lat_lon_records[self._output_lat] = lats
        lat_lon_records[self._output_lon] = longs
        return None, lat_lon_records

    def _extract_domain_from_email(self, lat_and_long_df):

        def get_domain_from_email(email) -> str:
            if pd.isnull(email) or '@' not in email:
                return np.NaN
            return email.split('@')[-1]

        emails = lat_and_long_df[self._input_email]
        prepared_df = pd.DataFrame(lat_and_long_df)
        prepared_df[self._output_domain] = emails.apply(get_domain_from_email)
        return prepared_df

    def save_to_csv(self, file_name: str):
        """
        Saves processed CRM report into CSV file.

        Args:
            file_name (str): A path to the output file.
        """
        if not self.ready:
            raise DataSourceError("Data source is not ready")
        self.data.to_csv(file_name)


class DatabaseTableSource(DataSource):
    """
    Retrieves data from specified database table using provided connector
    and puts received records into data frame.
    """

    ERROR_PREFIX = "Database table source"
    INITIALIZATION = "Table source initialization"
    TABLE_QUERY = "Querying database"

    def __init__(self, table_name: str, conn: DBConnection, **params):
        super(DatabaseTableSource, self).__init__(**params)
        self._columns = params.get('columns', [])
        self._table_name = table_name
        self._name = table_name
        self._conn = conn
        self._data = None

        if not isinstance(self._columns, Sequence):
            msg = self.format_error(
                self.INITIALIZATION,
                "columns argument should be a sequence, "
                "not a {}".format(type(self._columns)))
            log.error(msg)
            raise DataSourceError(msg)

    @property
    def name(self):
        return self._table_name

    @property
    def data(self):
        return self._data

    def prepare(self):
        try:
            log.info(self.TABLE_QUERY + "...")
            if not self._columns:
                self._columns = self._conn.columns(self._table_name)
            rows = self._conn.select(self._table_name, self._columns)

        except Exception as e:
            msg = self.format_error(self.TABLE_QUERY, str(e))
            log.error(msg)
            raise DataSourceError(msg)

        else:
            df = pd.DataFrame([dict(zip(self._columns, list(row)))
                               for row in rows])
            log.info("Database table data prepared")
            self._data = df
