"""
This module contains the code to extend and improve DatabaseToolSpec
Specifically adding load_sample_data and load_unique_values methods, as well as
making sure the load_data method returns a list of text values from the database, not Document[] objects.
"""
from abc import ABC
from typing import Callable, Any

#
# Additional database tool
#
class DBTool(ABC):
    """
    A base class for vectara-agentic database tools extensions
    """
    def __init__(self, load_data_fn: Callable):
        self.load_data_fn = load_data_fn

class DBLoadData(DBTool):
    """
    A tool to Run SQL query on the database and return the result.
    """
    def __call__(self, query: str) -> Any:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): an SQL query to filter tables and rows.

        Returns:
            List[text]: a list of text values from the database.
        """
        res = self.load_data_fn(query)
        return [d.text for d in res]

class DBLoadSampleData(DBTool):
    """
    A tool to load a sample of data from the specified database table.

    This tool fetches the first num_rows (default 25) rows from the given table
    using a provided database query function.
    """
    def __call__(self, table_name: str, num_rows: int = 25) -> Any:
        """
        Fetches the first num_rows rows from the specified database table.

        Args:
            table_name (str): The name of the database table.

        Returns:
            Any: The result of the database query.
        """
        return self.load_data_fn(f"SELECT * FROM {table_name} LIMIT {num_rows}")

class DBLoadUniqueValues(DBTool):
    """
    A tool to list all unique values for each column in a set of columns of a database table.
    """
    def __call__(self, table_name: str, columns: list[str], num_vals: int = 200) -> dict:
        """
        Fetches the first num_vals unique values from the specified columns of the database table.

        Args:
            table_name (str): The name of the database table.
            columns (list[str]): The list of columns to fetch unique values from.
            num_vals (int): The number of unique values to fetch for each column. Default is 200.

        Returns:
            dict: A dictionary containing the unique values for each column.
        """
        res = {}
        for column in columns:
            unique_vals = self.load_data_fn(f'SELECT DISTINCT "{column}" FROM {table_name} LIMIT {num_vals}')
            res[column] = [d.text for d in unique_vals]
        return res