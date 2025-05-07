"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re

from typing import (
    Callable, List, Dict, Any, Optional, Union, Type, Tuple,
    Sequence, get_origin, get_args
)
from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.workflow.context import Context

from llama_index.core.tools.types import BaseTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai.utils import resolve_tool_choice

from .types import ToolType
from .utils import is_float


def _updated_openai_prepare_chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    tool_choice: Union[str, dict] = "auto",
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Predict and call the tool."""
    tool_specs = [tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools]

    # if strict is passed in, use, else default to the class-level attribute, else default to True`
    strict = strict if strict is not None else self.strict

    if self.metadata.is_function_calling_model:
        for tool_spec in tool_specs:
            if tool_spec["type"] == "function":
                tool_spec["function"]["strict"] = strict
                # in current openai 1.40.0 it is always false.
                tool_spec["function"]["parameters"]["additionalProperties"] = False

    if isinstance(user_msg, str):
        user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

    messages = chat_history or []
    if user_msg:
        messages.append(user_msg)

    return {
        "messages": messages,
        "tools": tool_specs or None,
        "tool_choice": resolve_tool_choice(tool_choice) if tool_specs else None,
        **kwargs,
    }

class VectaraToolMetadata(ToolMetadata):
    """
    A subclass of ToolMetadata adding the tool_type attribute.
    """

    tool_type: ToolType

    def __init__(self, tool_type: ToolType, **kwargs):
        super().__init__(**kwargs)
        self.tool_type = tool_type

    def __repr__(self) -> str:
        """
        Returns a string representation of the VectaraToolMetadata object, including the tool_type attribute.
        """
        base_repr = super().__repr__()
        return f"{base_repr}, tool_type={self.tool_type}"


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
    ) -> None:
        metadata_dict = (
            metadata.dict() if hasattr(metadata, "dict") else metadata.__dict__
        )
        vm = VectaraToolMetadata(tool_type=tool_type, **metadata_dict)
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
        tool_type: ToolType = ToolType.QUERY,
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
        )
        vectara_tool = cls(
            tool_type=tool_type,
            fn=tool.fn,
            metadata=tool.metadata,
            async_fn=tool.async_fn,
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

    def call(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        try:
            return super().call(*args, ctx=ctx, **kwargs)
        except TypeError as e:
            sig = inspect.signature(self.metadata.fn_schema)
            valid_parameters = list(sig.parameters.keys())
            params_str = ", ".join(valid_parameters)

            err_output = ToolOutput(
                tool_name=self.metadata.name,
                content=(
                    f"Wrong argument used when calling {self.metadata.name}: {str(e)}."
                    f"Valid arguments: {params_str}. please call the tool again with the correct arguments."
                ),
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(e)},
            )
            return err_output
        except Exception as e:
            err_output = ToolOutput(
                tool_name=self.metadata.name,
                content=f"Tool {self.metadata.name} Malfunction: {str(e)}",
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(e)},
            )
            return err_output

    async def acall(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> ToolOutput:
        try:
            return await super().acall(*args, ctx=ctx, **kwargs)
        except TypeError as e:
            sig = inspect.signature(self.metadata.fn_schema)
            valid_parameters = list(sig.parameters.keys())
            params_str = ", ".join(valid_parameters)

            err_output = ToolOutput(
                tool_name=self.metadata.name,
                content=(
                    f"Wrong argument used when calling {self.metadata.name}: {str(e)}. "
                    f"Valid arguments: {params_str}. please call the tool again with the correct arguments."
                ),
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(e)},
            )
            return err_output
        except Exception as e:
            import traceback
            err_output = ToolOutput(
                tool_name=self.metadata.name,
                content=f"Tool {self.metadata.name} Malfunction: {str(e)}, traceback: {traceback.format_exc()}",
                raw_input={"args": args, "kwargs": kwargs},
                raw_output={"response": str(e)},
            )
            return err_output


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
    compact_docstring: bool,
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
        compact_docstring: If True, omits the signature line in the main description.

    Returns:
        A formatted docstring string.
    """
    params_str_parts = []
    for p in all_params:
        type_repr = _format_type(p.annotation)
        params_str_parts.append(f"{p.name}: {type_repr}")

    params_str = ", ".join(params_str_parts)
    signature_line = f"{tool_name}({params_str}) -> dict[str, Any]"

    if compact_docstring:
        doc_lines = [tool_description.strip()]
    else:
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
                elif isinstance(ty_info, list):  # Handle JSON schema array type e.g., ["integer", "string"]
                    ty_str = " | ".join([_clean_type_repr(t) for t in ty_info])

            # inline default if present
            default_txt = f", default={default!r}" if default is not PydanticUndefined else ""

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
    collapsed_spaces = re.sub(r' {2,}', ' ', initial_docstring)
    final_docstring = re.sub(r'\n{2,}', '\n', collapsed_spaces).strip()
    return final_docstring


def create_tool_from_dynamic_function(
    function: Callable[..., ToolOutput],
    tool_name: str,
    tool_description: str,
    base_params_model: Type[BaseModel],
    tool_args_schema: Type[BaseModel],
    compact_docstring: bool = False,
    return_direct: bool = False,
) -> VectaraTool:
    """
    Create a VectaraTool from a dynamic function.
    Args:
        function (Callable[..., ToolOutput]): The function to wrap as a tool.
        tool_name (str): The name of the tool.
        tool_description (str): The description of the tool.
        base_params_model (Type[BaseModel]): The Pydantic model for the base parameters.
        tool_args_schema (Type[BaseModel]): The Pydantic model for the tool arguments.
        compact_docstring (bool): Whether to use a compact docstring format.
    Returns:
        VectaraTool: The created VectaraTool.
    """
    if tool_args_schema is None:
        tool_args_schema = EmptyBaseModel

    if not isinstance(tool_args_schema, type) or not issubclass(tool_args_schema, BaseModel):
        raise TypeError("tool_args_schema must be a Pydantic BaseModel subclass")

    fields: Dict[str, Any] = {}
    base_params = []
    for field_name, field_info in base_params_model.model_fields.items():
        default = Ellipsis if field_info.default is PydanticUndefined else field_info.default
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default if default is not Ellipsis else inspect.Parameter.empty,
            annotation=field_info.annotation,
        )
        base_params.append(param)
        fields[field_name] = (field_info.annotation, field_info)

    # Add tool_args_schema fields to the fields dict if not already included.
    for field_name, field_info in tool_args_schema.model_fields.items():
        if field_name in fields:
            continue

        default = Ellipsis if field_info.default is PydanticUndefined else field_info.default
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default if default is not Ellipsis else inspect.Parameter.empty,
            annotation=field_info.annotation,
        )
        base_params.append(param)
        fields[field_name] = (field_info.annotation, field_info)

    # Create the dynamic schema with both base_params_model and tool_args_schema fields.
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
        function,
        tool_name, tool_description, fn_schema,
        all_params, compact_docstring
    )
    tool = VectaraTool.from_defaults(
        fn=function,
        name=tool_name,
        description=function.__doc__,
        fn_schema=fn_schema,
        tool_type=ToolType.QUERY,
        return_direct=return_direct,
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
