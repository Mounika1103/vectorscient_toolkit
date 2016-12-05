from sqlalchemy import create_engine
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import select

from .utils.db import get_engine


class DBConnection(object):
    """ Connect to the SQL database
    """
    def __init__(self, database, **kwargs):
        # sql engine
        #engine = get_engine(database)
        self.username = username
        self.password = password
        self.host = host
        self.database = database
        self.cursor = create_engine(('mysql+pymysql://{}:{}@{}/{}?charset=utf8mb4').format(self.username,self.password,self.host,self.database))

    def table(self, table_name):
        """
        Creates an SQLAlchemy table object to make queries directly.
        """
        return self._create_table(table_name)

    def select(self, table_name, columns=None):
        """
        Queries specified database table.

        Args:
            table_name: The table to be queried.
            columns: The table columns to be selected; if empty, then all
                columns will be selected.

        Returns:
            (list): A collection of fetched rows
        """
        table = self._create_table(table_name)
        if columns:
            expressions = [table.c[name] for name in columns]
            query = select(expressions)
        else:
            query = select(table.c)
        return self.cursor.execute(query).fetchall()

    def columns(self, table_name):
        """Returns set of column names for a given table."""
        table = self._create_table(table_name)
        return [c.name for c in table.c]

    def insert(self, table_name, rows, bulk=True):
        """
        Inserts rows into specified table.

        Args:
            table_name: The table to insert rows.
            rows: The list of rows to be inserted.
            bulk: If true, then rows will be inserted using the single insert
                query; otherwise, each row will have its own query.

        Returns:
            results (list): The list of errors if any.
        """
        table = self._create_table(table_name)
        return self._perform_query(table.insert(), rows, bulk)

    def update(self, table_name, data, id_column_name='id'):
        """ Update a row of the specified database table based on
            the ID.

            FORMATS:
                table_name : string
                data       : {'id': value1, 'name': new_value, ...}
        """
        table = self._create_table(table_name)
        for row in data:
            try:
                statement = table.update() \
                        .where(table.c[id_column_name] == row[id_column_name]) \
                        .values(**row)
                self.cursor.execute(statement)
            except Exception as e:
                print (e)

    def clean(self, table_name, column, value=None):
        """
        Resets specified table's column with specific value.

        Args:
             table_name: The table which column should be cleaned.
             column: The column to be cleaned.
             value: The value to be set instead of removed ones.
        """
        table = self._create_table(table_name)
        row = {column: value}
        statement = table.update().values(**row)
        return self._try_to_execute_statement(statement)

    def _perform_query(self, query, rows, bulk=True):
        results = []

        if bulk and len(rows) > 1:
            results.append(self._try_to_execute_statement(query, rows))

        else:
            for index, row in enumerate(rows):
                statement = query.values(**row)
                result = self._try_to_execute_statement(
                    statement, raw_index=index)
                results.append(result)

        return results

    def _try_to_execute_statement(self, statement, values=None, **info):
        try:
            self.cursor.execute(statement, values if values else ())
            return {'saved': True, 'exc': None}

        except Exception as e:
            index = info.get('row_index', None)
            print(e)
            print(index)
            return {'saved': False, 'exc': e}

    def _create_table(self, table_name, autoload=True):
        return Table(table_name, MetaData(bind=self.cursor), autoload=autoload)