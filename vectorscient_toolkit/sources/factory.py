from ..exceptions.exc import ConfigurationError
from .data import *


def create_data_source_from_config(source_config):
    """
    Reads data that should be clustered using provided configuration.

    Args:
        source_config (dict): The dictionary with data source parameters.

    Returns:
        DataSource: The requested data source.
    """
    if source_config["source_type"].lower() == "csv":
        records_file = source_config["source_file"]
        reader_config = source_config["reader_config"]
        source = CsvDataSource(records_file, reader_config=reader_config)
        return source

    elif source_config["source_type"].lower() == "db":
        table_name = source_config["table_name"]
        database = source_config["database"]
        conn = DBConnection(database=database)
        source = DatabaseTableSource(table_name, conn)
        return source

    err = "Unexpected data source type: " + source_config["source_type"]
    raise ConfigurationError(err)
