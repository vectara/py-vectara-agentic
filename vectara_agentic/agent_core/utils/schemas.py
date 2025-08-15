"""
Schema and type conversion utilities for agent functionality.

This module handles JSON schema to Python type conversion,
Pydantic model reconstruction, and type mapping operations.
"""

from typing import Any, Union, List


# Type mapping constants
JSON_TYPE_TO_PYTHON = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "number": float,
    "null": type(None),
}

PY_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
}


def get_field_type(field_schema: dict) -> Any:
    """
    Convert a JSON schema field definition to a Python type.
    Handles 'type' and 'anyOf' cases.

    Args:
        field_schema: JSON schema field definition

    Returns:
        Any: Corresponding Python type
    """
    if not field_schema:  # Handles empty schema {}
        return Any

    if "anyOf" in field_schema:
        types = []
        for option_schema in field_schema["anyOf"]:
            types.append(get_field_type(option_schema))  # Recursive call
        if not types:
            return Any
        return Union[tuple(types)]

    if "type" in field_schema and isinstance(field_schema["type"], list):
        types = []
        for type_name in field_schema["type"]:
            if type_name == "array":
                item_schema = field_schema.get("items", {})
                types.append(List[get_field_type(item_schema)])
            elif type_name in JSON_TYPE_TO_PYTHON:
                types.append(JSON_TYPE_TO_PYTHON[type_name])
            else:
                types.append(Any)  # Fallback for unknown types in the list
        if not types:
            return Any
        return Union[tuple(types)]  # type: ignore

    if "type" in field_schema:
        schema_type_name = field_schema["type"]
        if schema_type_name == "array":
            item_schema = field_schema.get(
                "items", {}
            )  # Default to Any if "items" is missing
            return List[get_field_type(item_schema)]

        return JSON_TYPE_TO_PYTHON.get(schema_type_name, Any)

    # If only "items" is present (implies array by some conventions, but less standard)
    # Or if it's a schema with other keywords like 'properties' (implying object)
    # For simplicity, if no "type" or "anyOf" at this point, default to Any or add more specific handling.
    # If 'properties' in field_schema, it's likely an object.
    if "properties" in field_schema:
        # This path might need to reconstruct a nested Pydantic model if you encounter such schemas.
        # For now, treating as 'dict' or 'Any' might be a simpler placeholder.
        return dict  # Or Any, or more sophisticated object reconstruction.

    return Any
