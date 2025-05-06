"""
This module contains the code adapted from DatabaseToolSpec
It makes the following adjustments:
* Adds load_sample_data and load_unique_values methods.
* Fixes serialization.
* Makes sure the load_data method returns a list of text values from the database (and not Document[] objects).
* Limits the returned rows to self.max_rows.
"""
from typing import Any, Optional, List, Awaitable, Callable
import asyncio
from inspect import signature

from sqlalchemy import MetaData, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.schema import CreateTable

from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.schema import Document
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.tools.utils import create_schema_from_function


AsyncCallable = Callable[..., Awaitable[Any]]

class DatabaseTools:
    """Database tools for vectara-agentic
    This class provides a set of tools to interact with a database.
    It allows you to load data, list tables, describe tables, and load unique values.
    It also provides a method to load sample data from a specified table.
    """
    spec_functions = [
        "load_data", "load_sample_data", "list_tables",
        "describe_tables", "load_unique_values",
    ]

    def __init__(
        self,
        *args: Any,
        max_rows: int = 1000,
        sql_database: Optional[SQLDatabase] = None,
        engine: Optional[Engine] = None,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        tool_name_prefix: str = "db",
        **kwargs: Any,
    ) -> None:
        self.max_rows = max_rows
        self.tool_name_prefix = tool_name_prefix

        if sql_database:
            self.sql_database = sql_database
        elif engine:
            self.sql_database = SQLDatabase(engine, *args, **kwargs)
        elif uri:
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        elif (scheme and host and port and user and password and dbname):
            uri = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        else:
            raise ValueError(
                "You must provide either a SQLDatabase, "
                "a SQL Alchemy Engine, a valid connection URI, or a valid "
                "set of credentials."
            )
        self._uri = getattr(self, "uri", None) or str(self.sql_database.engine.url)
        self._metadata = MetaData()
        self._metadata.reflect(bind=self.sql_database.engine)

    def _get_metadata_from_fn_name(
        self, fn_name: Callable,
    ) -> Optional[ToolMetadata]:
        """Return map from function name.

        Return type is Optional, meaning that the schema can be None.
        In this case, it's up to the downstream tool implementation to infer the schema.
        """
        try:
            func = getattr(self, fn_name)
        except AttributeError:
            return None
        name = self.tool_name_prefix + "_" + fn_name if self.tool_name_prefix else fn_name
        docstring = func.__doc__ or ""
        description = f"{name}{signature(func)}\n{docstring}"
        fn_schema = create_schema_from_function(fn_name, getattr(self, fn_name))
        return ToolMetadata(name=name, description=description, fn_schema=fn_schema)

    def _load_data(self, sql_query: str) -> List[Document]:
        documents = []
        with self.sql_database.engine.connect() as connection:
            if sql_query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            result = connection.execute(text(sql_query))
            for item in result.fetchall():
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(Document(text=doc_str))
        return documents

    def load_data(self, sql_query: str) -> List[str]:
        """Query and load data from the Database, returning a list of Documents.
        Args:
            sql_query (str): an SQL query to filter tables and rows.
        Returns:
            List[str]: a list of Document objects from the database.
        """
        if sql_query is None:
            raise ValueError("A query parameter is necessary to filter the data.")

        count_query = f"SELECT COUNT(*) FROM ({sql_query})"
        try:
            count_rows = self._load_data(count_query)
        except Exception as e:
            return [f"Error ({str(e)}) occurred while counting number of rows, check your query."]
        num_rows = int(count_rows[0].text)
        if num_rows > self.max_rows:
            return [
                f"The query is expected to return more than {self.max_rows} rows. "
                "Please refactor your query to make it return less rows and try again. "
            ]
        try:
            res = self._load_data(sql_query)
        except Exception as e:
            return [f"Error ({str(e)}) occurred while executing the query {sql_query}, check your query."]
        return [d.text for d in res]

    def load_sample_data(self, table_name: str, num_rows: int = 25) -> Any:
        """
        Fetches the first num_rows rows from the specified database table.

        Args:
            table_name (str): The name of the database table.

        Returns:
            Any: The result of the database query.
        """
        if table_name not in self.list_tables():
            return (
                f"Table {table_name} does not exist in the database."
                f"Valid table names are: {self.list_tables()}"
            )
        try:
            res = self._load_data(f"SELECT * FROM {table_name} LIMIT {num_rows}")
        except Exception as e:
            return [f"Error ({str(e)}) occurred while loading sample data for table {table_name}"]
        return [d.text for d in res]

    def list_tables(self) -> List[str]:
        """List all tables in the database.
        Returns:
            List[str]: A list of table names in the database.
        """
        return [x.name for x in self._metadata.sorted_tables]

    def describe_tables(self, tables: Optional[List[str]] = None) -> str:
        """Describe the tables in the database.
        Args:
            tables (Optional[List[str]]): A list of table names to describe. If None, all tables are described.
        Returns:
            str: A string representation of the table schemas.
        """
        table_names = tables or [table.name for table in self._metadata.sorted_tables]
        if len(table_names) == 0:
            return "You must specify at least one table name to describe."
        for table_name in table_names:
            if table_name not in self.list_tables():
                return (
                    f"Table {table_name} does not exist in the database."
                    f"Valid table names are: {self.list_tables()}"
                )

        table_schemas = []
        for table_name in table_names:
            table = next(
                (table for table in self._metadata.sorted_tables if table.name == table_name),
                None,
            )
            if table is None:
                raise NoSuchTableError(f"Table '{table_name}' does not exist.")
            schema = str(CreateTable(table).compile(self.sql_database.engine))
            table_schemas.append(f"{schema}\n")
        return "\n".join(table_schemas)

    def load_unique_values(self, table_name: str, columns: list[str], num_vals: int = 200) -> Any:
        """
        Fetches the first num_vals unique values from the specified columns of the database table.

        Args:
            table_name (str): The name of the database table.
            columns (list[str]): The list of columns to fetch unique values from.
            num_vals (int): The number of unique values to fetch for each column. Default is 200.

        Returns:
            Any: the result of the database query
        """
        if table_name not in self.list_tables():
            return (
                f"Table {table_name} does not exist in the database."
                f"Valid table names are: {self.list_tables()}"
            )

        res = {}
        try:
            for column in columns:
                unique_vals = self._load_data(f'SELECT DISTINCT "{column}" FROM {table_name} LIMIT {num_vals}')
                res[column] = [d.text for d in unique_vals]
        except Exception as e:
            return {f"Error ({str(e)}) occurred while loading unique values for table {table_name}"}
        return res

    def to_tool_list(self) -> List[FunctionTool]:
        """
        Returns a list of tools available.
        """

        tool_list = []
        for tool_name in self.spec_functions:
            func_sync = None
            func_async = None
            func = getattr(self, tool_name)
            if asyncio.iscoroutinefunction(func):
                func_async = func
            else:
                func_sync = func
            metadata = self._get_metadata_from_fn_name(tool_name)

            if func_sync is None:
                if func_async is not None:
                    func_sync = patch_sync(func_async)
                else:
                    raise ValueError(
                        f"Could not retrieve a function for spec: {tool_name}"
                    )

            tool = FunctionTool.from_defaults(
                fn=func_sync,
                async_fn=func_async,
                tool_metadata=metadata,
            )
            tool_list.append(tool)
        return tool_list

    # Custom pickling: exclude unpickleable objects
    def __getstate__(self):
        state = self.__dict__.copy()
        if "sql_database" in state:
            state["sql_database_state"] = {"uri": self._uri}
            del state["sql_database"]
        if "_metadata" in state:
            del state["_metadata"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct the sql_database if it was removed
        if "sql_database_state" in state:
            uri = state["sql_database_state"].get("uri")
            if uri:
                self.sql_database = SQLDatabase.from_uri(uri)
                self._uri = uri
            else:
                raise ValueError("Cannot reconstruct SQLDatabase without URI")
            # Rebuild metadata after restoring the engine
            self._metadata = MetaData()
            self._metadata.reflect(bind=self.sql_database.engine)


def patch_sync(func_async: AsyncCallable) -> Callable:
    """Patch sync function from async function."""

    def patched_sync(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func_async(*args, **kwargs))

    return patched_sync
