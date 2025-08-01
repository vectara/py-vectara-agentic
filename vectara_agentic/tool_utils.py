"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re
import traceback

from typing import (
    Callable,
    List,
    Dict,
    Any,
    Optional,
    Union,
    Type,
    Tuple,
    get_origin,
    get_args,
)
from pydantic import BaseModel, create_model, Field
from pydantic_core import PydanticUndefined

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.workflow.context import Context

from .types import ToolType
from .utils import is_float


class VectaraToolMetadata(ToolMetadata):
    """
    A subclass of ToolMetadata adding the tool_type and vhc_eligible attributes.
    """

    tool_type: ToolType
    vhc_eligible: bool

    def __init__(self, tool_type: ToolType, vhc_eligible: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tool_type = tool_type
        self.vhc_eligible = vhc_eligible

    def __repr__(self) -> str:
        """
        Returns a string representation of the VectaraToolMetadata object,
        including the tool_type and vhc_eligible attributes.
        """
        base_repr = super().__repr__()
        return (
            f"{base_repr}, tool_type={self.tool_type}, vhc_eligible={self.vhc_eligible}"
        )


class VectaraTool(FunctionTool):
    """
    A subclass of FunctionTool adding the tool_type attribute.
    """

    def __init__(
        self,
        tool_type: ToolType,
        metadata: ToolMetadata,
        fn: Optional[Callable[..., Any]] = None,
        async_fn: Optional[AsyncCallable] = None,
        vhc_eligible: bool = True,
    ) -> None:
        # Use Pydantic v2 compatible method for extracting metadata
        metadata_dict = (
            metadata.model_dump()
            if hasattr(metadata, "model_dump")
            else metadata.dict() if hasattr(metadata, "dict") else metadata.__dict__
        )
        vm = VectaraToolMetadata(
            tool_type=tool_type, vhc_eligible=vhc_eligible, **metadata_dict
        )
        super().__init__(fn, vm, async_fn)

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        async_callback: Optional[AsyncCallable] = None,
        partial_params: Optional[Dict[str, Any]] = None,
        tool_type: ToolType = ToolType.QUERY,
        vhc_eligible: bool = True,
    ) -> "VectaraTool":
        tool = FunctionTool.from_defaults(
            fn,
            name,
            description,
            return_direct,
            fn_schema,
            async_fn,
            tool_metadata,
            callback,
            async_callback,
            partial_params,
        )
        vectara_tool = cls(
            tool_type=tool_type,
            fn=tool.fn,
            metadata=tool.metadata,
            async_fn=tool.async_fn,
            vhc_eligible=vhc_eligible,
        )
        return vectara_tool

    def __str__(self) -> str:
        return f"Tool(name={self.metadata.name}, " f"Tool metadata={self.metadata})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        try:
            # Try to get schema as dict if possible
            self_schema = self.metadata.fn_schema.model_json_schema()
            other_schema = other.metadata.fn_schema.model_json_schema()
        except Exception:
            return False

        is_equal = (
            isinstance(other, VectaraTool)
            and self.metadata.tool_type == other.metadata.tool_type
            and self.metadata.name == other.metadata.name
            and self_schema == other_schema
        )
        return is_equal

    def _create_tool_error_output(
        self, error: Exception, args: Any, kwargs: Any, include_traceback: bool = False
    ) -> ToolOutput:
        """Create standardized error output for tool execution failures."""
        if isinstance(error, TypeError):
            # Parameter validation error handling
            sig = inspect.signature(self.metadata.fn_schema)
            valid_parameters = list(sig.parameters.keys())
            params_str = ", ".join(valid_parameters)
            return ToolOutput(
                tool_name=self.metadata.name,
                content=(
                    f"Wrong argument used when calling {self.metadata.name}: {str(error)}. "
                    f"Valid arguments: {params_str}. Please call the tool again with the correct arguments."
                ),
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(error)},
            )
        else:
            # General execution error handling
            content = f"Tool {self.metadata.name} Malfunction: {str(error)}"
            if include_traceback:
                content += f", traceback: {traceback.format_exc()}"

            return ToolOutput(
                tool_name=self.metadata.name,
                content=content,
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(error)},
            )

    def call(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        try:
            # Only pass ctx if it's not None to avoid passing unwanted kwargs to the function
            if ctx is not None:
                result = super().call(*args, ctx=ctx, **kwargs)
            else:
                result = super().call(*args, **kwargs)
            return self._format_tool_output(result)
        except Exception as e:
            return self._create_tool_error_output(e, args, kwargs)

    async def acall(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        try:
            # Only pass ctx if it's not None to avoid passing unwanted kwargs to the function
            if ctx is not None:
                result = await super().acall(*args, ctx=ctx, **kwargs)
            else:
                result = await super().acall(*args, **kwargs)
            return self._format_tool_output(result)
        except Exception as e:
            return self._create_tool_error_output(
                e, args, kwargs, include_traceback=True
            )

    def _format_tool_output(self, result: ToolOutput) -> ToolOutput:
        """Format tool output by converting human-readable wrappers to formatted content immediately."""
        import logging

        # If the raw_output has human-readable formatting, use it for the content
        if hasattr(result, "raw_output") and _is_human_readable_output(
            result.raw_output
        ):
            try:
                formatted_content = result.raw_output.to_human_readable()
                # Replace the content with the formatted version
                result.content = formatted_content
            except Exception as e:
                logging.warning(
                    f"{self.metadata.name}: Failed to convert to human-readable: {e}"
                )

        return result


class EmptyBaseModel(BaseModel):
    """empty base model"""


def _clean_type_repr(type_repr: str) -> str:
    """Cleans the string representation of a type."""
    # Replace <class 'somename'> with somename
    match = re.match(r"<class '(\w+)'>", type_repr)
    if match:
        type_repr = match.group(1)

    type_repr = type_repr.replace("typing.", "")
    return type_repr


def _format_type(annotation) -> str:
    """
    Turn things like Union[int, str, NoneType] into 'int | str | None',
    and replace any leftover 'NoneType' → 'None'.
    """
    origin = get_origin(annotation)
    if origin is Union:
        parts = []
        for arg in get_args(annotation):
            if arg is type(None):
                parts.append("None")
            else:
                # recurse in case of nested unions
                parts.append(_format_type(arg))
        return " | ".join(parts)

    # Fallback
    type_repr = str(annotation)
    type_repr = _clean_type_repr(type_repr)
    return type_repr.replace("NoneType", "None")


def _make_docstring(
    function: Callable[..., ToolOutput],
    tool_name: str,
    tool_description: str,
    fn_schema: Type[BaseModel],
    all_params: List[inspect.Parameter],
) -> str:
    """
    Generates a docstring for a function based on its signature, description,
    and Pydantic schema, correctly handling complex type annotations.

    Args:
        function: The function for which to generate the docstring.
        tool_name: The desired name for the tool/function in the docstring.
        tool_description: The main description of the tool/function.
        fn_schema: The Pydantic model representing the function's arguments schema.
        all_params: A list of inspect.Parameter objects for the function signature.

    Returns:
        A formatted docstring string.
    """
    params_str_parts = []
    for p in all_params:
        type_repr = _format_type(p.annotation)
        params_str_parts.append(f"{p.name}: {type_repr}")

    params_str = ", ".join(params_str_parts)
    signature_line = f"{tool_name}({params_str}) -> dict[str, Any]"

    doc_lines = [signature_line, "", tool_description.strip()]

    full_schema = fn_schema.model_json_schema()
    props = full_schema.get("properties", {})

    if props:
        doc_lines.extend(["", "Args:"])
        for prop_name, schema_prop in props.items():
            desc = schema_prop.get("description", "")

            # pick up any examples you declared on the Field or via schema_extra
            examples = schema_prop.get("examples", [])
            default = schema_prop.get("default", PydanticUndefined)

            # format the type, default, description, examples
            # find the matching inspect.Parameter so you get its annotation
            param = next((p for p in all_params if p.name == prop_name), None)
            ty_str = ""
            if param:
                ty_str = _format_type(param.annotation)
            elif "type" in schema_prop:
                ty_info = schema_prop["type"]
                if isinstance(ty_info, str):
                    ty_str = _clean_type_repr(ty_info)
                elif isinstance(
                    ty_info, list
                ):  # Handle JSON schema array type e.g., ["integer", "string"]
                    ty_str = " | ".join([_clean_type_repr(t) for t in ty_info])

            # inline default if present
            default_txt = (
                f", default={default!r}" if default is not PydanticUndefined else ""
            )

            # inline examples if any
            if examples:
                examples_txt = ", ".join(repr(e) for e in examples)
                desc = f"{desc}  (e.g., {examples_txt})"

            doc_lines.append(f"  - {prop_name} ({ty_str}{default_txt}): {desc}")

    doc_lines.append("")
    doc_lines.append("Returns:")
    return_desc = getattr(
        function, "__return_description__", "A dictionary containing the result data."
    )
    doc_lines.append(f"    dict[str, Any]: {return_desc}")

    initial_docstring = "\n".join(doc_lines)
    collapsed_spaces = re.sub(r" {2,}", " ", initial_docstring)
    final_docstring = re.sub(r"\n{2,}", "\n", collapsed_spaces).strip()
    return final_docstring


def _auto_fix_field_if_needed(
    field_name: str, field_info, annotation
) -> Tuple[Any, Any]:
    """
    Auto-fix problematic Field definitions: convert non-Optional types with any default value to Optional.

    Args:
        field_name: Name of the field
        field_info: The Pydantic FieldInfo object
        annotation: The type annotation for the field

    Returns:
        Tuple of (possibly_modified_annotation, possibly_modified_field_info)
    """
    # Check for problematic pattern: non-Optional type with any default value
    if (
        field_info.default is not PydanticUndefined
        and annotation is not None
        and get_origin(annotation) is not Union
    ):

        # Convert to Optional[OriginalType] and keep the original default value
        new_annotation = Union[annotation, type(None)]
        # Create new field_info preserving the original default value
        new_field_info = Field(
            default=field_info.default,
            description=field_info.description,
            examples=getattr(field_info, "examples", None),
            title=getattr(field_info, "title", None),
            alias=getattr(field_info, "alias", None),
            json_schema_extra=getattr(field_info, "json_schema_extra", None),
        )

        # Optional: Log the auto-fix for debugging
        import logging

        logging.debug(
            f"Auto-fixed field '{field_name}': "
            f"converted {annotation} with default={field_info.default} to Optional[{annotation.__name__}]"
        )

        return new_annotation, new_field_info
    else:
        # Keep original field definition
        return annotation, field_info


def create_tool_from_dynamic_function(
    function: Callable[..., ToolOutput],
    tool_name: str,
    tool_description: str,
    base_params_model: Type[BaseModel],
    tool_args_schema: Type[BaseModel],
    return_direct: bool = False,
    vhc_eligible: bool = True,
) -> VectaraTool:
    """
    Create a VectaraTool from a dynamic function.
    Args:
        function (Callable[..., ToolOutput]): The function to wrap as a tool.
        tool_name (str): The name of the tool.
        tool_description (str): The description of the tool.
        base_params_model (Type[BaseModel]): The Pydantic model for the base parameters.
        tool_args_schema (Type[BaseModel]): The Pydantic model for the tool arguments.
        return_direct (bool): Whether to return the tool output directly.
    Returns:
        VectaraTool: The created VectaraTool.
    """
    if tool_args_schema is None:
        tool_args_schema = EmptyBaseModel

    if not isinstance(tool_args_schema, type) or not issubclass(
        tool_args_schema, BaseModel
    ):
        raise TypeError("tool_args_schema must be a Pydantic BaseModel subclass")

    fields: Dict[str, Any] = {}
    base_params = []
    for field_name, field_info in base_params_model.model_fields.items():
        # Apply auto-conversion if needed
        annotation, field_info = _auto_fix_field_if_needed(
            field_name, field_info, field_info.annotation
        )

        default = (
            Ellipsis if field_info.default is PydanticUndefined else field_info.default
        )
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default if default is not Ellipsis else inspect.Parameter.empty,
            annotation=annotation,
        )
        base_params.append(param)
        fields[field_name] = (annotation, field_info)

    # Add tool_args_schema fields to the fields dict if not already included.
    for field_name, field_info in tool_args_schema.model_fields.items():
        if field_name in fields:
            continue

        # Apply auto-conversion if needed
        annotation, field_info = _auto_fix_field_if_needed(
            field_name, field_info, field_info.annotation
        )

        default = (
            Ellipsis if field_info.default is PydanticUndefined else field_info.default
        )
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default if default is not Ellipsis else inspect.Parameter.empty,
            annotation=annotation,
        )
        base_params.append(param)
        fields[field_name] = (annotation, field_info)

    # Create the dynamic schema with both base_params_model and tool_args_schema fields (auto-fixed)
    fn_schema = create_model(f"{tool_name}_schema", **fields)

    # Combine parameters into a function signature.
    all_params = base_params[:]  # Now all_params contains parameters from both models.
    required_params = [p for p in all_params if p.default is inspect.Parameter.empty]
    optional_params = [
        p for p in all_params if p.default is not inspect.Parameter.empty
    ]
    function.__signature__ = inspect.Signature(required_params + optional_params)
    function.__annotations__["return"] = dict[str, Any]
    function.__name__ = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

    function.__doc__ = _make_docstring(
        function, tool_name, tool_description, fn_schema, all_params
    )
    tool = VectaraTool.from_defaults(
        fn=function,
        name=tool_name,
        description=function.__doc__,
        fn_schema=fn_schema,
        tool_type=ToolType.QUERY,
        return_direct=return_direct,
        vhc_eligible=vhc_eligible,
    )
    return tool


_PARSE_RANGE_REGEX = re.compile(
    r"""
    ^([\[\(])\s*            # opening bracket
    ([+-]?\d+(\.\d*)?)\s*,  # first number
    \s*([+-]?\d+(\.\d*)?)   # second number
    \s*([\]\)])$            # closing bracket
    """,
    re.VERBOSE,
)


def _parse_range(val_str: str) -> Tuple[str, str, bool, bool]:
    """
    Parses '[1,10)' or '(0.5, 5]' etc.
    Returns (start, end, start_incl, end_incl) or raises ValueError.
    """
    m = _PARSE_RANGE_REGEX.match(val_str)
    if not m:
        raise ValueError(f"Invalid range syntax: {val_str!r}")
    start_inc = m.group(1) == "["
    end_inc = m.group(6) == "]"
    start = m.group(2)
    end = m.group(4)
    if float(start) > float(end):
        raise ValueError(f"Range lower bound greater than upper bound: {val_str!r}")
    return start, end, start_inc, end_inc


def _parse_comparison(val_str: str) -> Tuple[str, Union[float, str, bool]]:
    """
    Parses '>10', '<=3.14', '!=foo', \"='bar'\" etc.
    Returns (operator, rhs) or raises ValueError.
    """
    # pick off the operator
    comparison_operators = [">=", "<=", "!=", ">", "<", "="]
    numeric_only_operators = {">", "<", ">=", "<="}
    for op in comparison_operators:
        if val_str.startswith(op):
            rhs = val_str[len(op) :].strip()
            if op in numeric_only_operators:
                try:
                    rhs_val = float(rhs)
                except ValueError as e:
                    raise ValueError(
                        f"Numeric comparison {op!r} must have a number, got {rhs!r}"
                    ) from e
                return op, rhs_val
            # = and != can be bool, numeric, or string
            low = rhs.lower()
            if low in ("true", "false"):
                return op, (low == "true")
            try:
                return op, float(rhs)
            except ValueError:
                return op, rhs
    raise ValueError(f"No valid comparison operator at start of {val_str!r}")


def build_filter_string(
    kwargs: Dict[str, Any], tool_args_type: Dict[str, dict], fixed_filter: str
) -> str:
    """
    Build filter string for Vectara from kwargs
    """
    filter_parts = []
    for key, raw in kwargs.items():
        if raw is None or raw == "":
            continue

        if raw is PydanticUndefined:
            raise ValueError(
                f"Value of argument {key!r} is undefined, and this is invalid. "
                "Please form proper arguments and try again."
            )

        tool_args_dict = tool_args_type.get(key, {"type": "doc", "is_list": False})
        prefix = tool_args_dict.get("type", "doc")
        is_list = tool_args_dict.get("is_list", False)

        # In case the tool_args_dict has a filter_name, use it, otherwise use the key
        # This is helpful in case the filter name needs to have spaces or special characters
        # not allowed in variable names.
        key = tool_args_dict.get("filter_name", key)

        # Validate prefix
        if prefix not in ("doc", "part"):
            raise ValueError(
                f'Unrecognized prefix {prefix!r}. Please make sure to use either "doc" or "part" for the prefix.'
            )

        # 1) native numeric
        if isinstance(raw, (int, float)) or is_float(str(raw)):
            val = str(raw)
            if is_list:
                filter_parts.append(f"({val} IN {prefix}.{key})")
            else:
                filter_parts.append(f"{prefix}.{key}={val}")
            continue

        # 2) native boolean
        if isinstance(raw, bool):
            val = "true" if raw else "false"
            if is_list:
                filter_parts.append(f"({val} IN {prefix}.{key})")
            else:
                filter_parts.append(f"{prefix}.{key}={val}")
            continue

        if not isinstance(raw, str):
            raise ValueError(f"Unsupported type for {key!r}: {type(raw).__name__}")

        val_str = raw.strip()

        # 3) Range operator
        if (val_str.startswith("[") or val_str.startswith("(")) and (
            val_str.endswith("]") or val_str.endswith(")")
        ):
            start, end, start_incl, end_incl = _parse_range(val_str)
            conds = []
            op1 = ">=" if start_incl else ">"
            op2 = "<=" if end_incl else "<"
            conds.append(f"{prefix}.{key} {op1} {start}")
            conds.append(f"{prefix}.{key} {op2} {end}")
            filter_parts.append("(" + " AND ".join(conds) + ")")
            continue

        # 4) comparison operator
        try:
            op, rhs = _parse_comparison(val_str)
        except ValueError:
            # no operator → treat as membership or equality-on-string
            if is_list:
                filter_parts.append(f"('{val_str}' IN {prefix}.{key})")
            else:
                filter_parts.append(f"{prefix}.{key}='{val_str}'")
        else:
            # normal comparison always binds to the field
            if isinstance(rhs, bool):
                rhs_sql = "true" if rhs else "false"
            elif isinstance(rhs, (int, float)):
                rhs_sql = str(rhs)
            else:
                rhs_sql = f"'{rhs}'"
            filter_parts.append(f"{prefix}.{key}{op}{rhs_sql}")

    joined = " AND ".join(filter_parts)
    if fixed_filter and joined:
        return f"({fixed_filter}) AND ({joined})"
    return fixed_filter or joined


def _is_human_readable_output(obj: Any) -> bool:
    """Check if an object implements the HumanReadableOutput protocol."""
    return (
        hasattr(obj, "to_human_readable")
        and hasattr(obj, "get_raw_output")
        and callable(getattr(obj, "to_human_readable", None))
        and callable(getattr(obj, "get_raw_output", None))
    )


def create_human_readable_output(
    raw_output: Any, formatter: Optional[Callable[[Any], str]] = None
) -> "HumanReadableToolOutput":
    """Create a HumanReadableToolOutput wrapper for tool outputs."""
    return HumanReadableToolOutput(raw_output, formatter)


def format_as_table(data: List[Dict[str, Any]], max_width: int = 80) -> str:
    """Format list of dictionaries as a table."""
    if not data:
        return "No data to display"

    # Get all unique keys
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    headers = list(all_keys)

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = max(
            len(header), max(len(str(item.get(header, ""))) for item in data)
        )
        # Limit column width
        col_widths[header] = min(col_widths[header], max_width // len(headers))

    # Create table
    lines = []

    # Header row
    header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Data rows
    for item in data:
        row = " | ".join(
            str(item.get(header, "")).ljust(col_widths[header])[: col_widths[header]]
            for header in headers
        )
        lines.append(row)

    return "\n".join(lines)


def format_as_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty-printed JSON."""
    import json

    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)


def format_as_markdown_list(items: List[Any], numbered: bool = False) -> str:
    """Format items as markdown list."""
    if not items:
        return "No items to display"

    if numbered:
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    else:
        return "\n".join(f"- {item}" for item in items)


class HumanReadableToolOutput:
    """Wrapper class that implements HumanReadableOutput protocol."""

    def __init__(
        self, raw_output: Any, formatter: Optional[Callable[[Any], str]] = None
    ):
        self._raw_output = raw_output
        self._formatter = formatter or str

    def to_human_readable(self) -> str:
        """Convert the output to a human-readable format."""
        try:
            return self._formatter(self._raw_output)
        except Exception as e:
            import logging

            logging.warning(f"Failed to format output with custom formatter: {e}")
            # Fallback to string representation
            try:
                return str(self._raw_output)
            except Exception:
                return f"[Error formatting output: {e}]"

    def get_raw_output(self) -> Any:
        """Get the raw output data."""
        return self._raw_output

    def __str__(self) -> str:
        return self.to_human_readable()

    def __repr__(self) -> str:
        return f"HumanReadableToolOutput({self._raw_output!r})"
