"""Conservative strict-tool transport schemas with stronger local validation.

The application deliberately supports a smaller subset than Anthropic. Unknown
structural keywords fail before HTTP; they are never silently discarded. Bounds
we implement locally are moved to descriptions for the API. The original schema
remains authoritative for accepting a reply. See docs/ANTHROPIC_REMEDIATION.md.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import math
import re
from typing import Any, Mapping


class SchemaError(ValueError):
    """Schema definition or tool input is outside the supported local contract."""


TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
LOCAL_CONSTRAINTS = {"minLength", "maxLength", "maxItems", "minimum", "maximum",
                     "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern"}
ALLOWED = {"type", "description", "title", "default", "enum", "const", "properties",
           "required", "additionalProperties", "items", "minItems"} | LOCAL_CONSTRAINTS


def _primitive(value: Any) -> bool:
    return value is None or type(value) in {str, bool, int} or type(value) is float and math.isfinite(value)


def _json_equal(a: Any, b: Any) -> bool:
    # JSON boolean true is not the numeric value 1.
    return a == b and (isinstance(a, bool) == isinstance(b, bool))


def transport_schema(schema: Mapping[str, Any], path: str = "schema", *, _depth: int = 0) -> dict:
    """Validate the application subset and return a new API-compatible schema.

    References, combinators, free-form objects and boolean schemas are rejected
    rather than partially supported. Nullable type arrays and primitive enums
    are retained. Objects are closed in BOTH transport and local acceptance.
    """
    if _depth > 24 or not isinstance(schema, Mapping):
        raise SchemaError(f"Invalid or excessive schema nesting at {path}.")
    unknown = set(schema) - ALLOWED
    if unknown:
        raise SchemaError(f"Unsupported schema keyword at {path}: {','.join(sorted(unknown))}.")
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if not kinds or any(not isinstance(k, str) or k not in TYPES for k in kinds) or len(set(kinds)) != len(kinds):
        raise SchemaError(f"A supported explicit type is required at {path}.")
    out = {"type": deepcopy(kind)}
    for key in ("description", "title"):
        if key in schema:
            if not isinstance(schema[key], str):
                raise SchemaError(f"Invalid {key} at {path}.")
            out[key] = schema[key]
    if "default" in schema:
        try:
            json.dumps(schema["default"], allow_nan=False)
        except (ValueError, TypeError):
            raise SchemaError(f"Invalid default at {path}.") from None
        out["default"] = deepcopy(schema["default"])
    for key in ("enum", "const"):
        if key in schema:
            values = schema[key] if key == "enum" else [schema[key]]
            if not isinstance(values, list) or not values or any(not _primitive(v) for v in values):
                raise SchemaError(f"Only nonempty primitive {key} values are supported at {path}.")
            out[key] = deepcopy(schema[key])
    if "object" in kinds:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, Mapping) or any(not isinstance(k, str) for k in props):
            raise SchemaError(f"Invalid properties at {path}.")
        if (not isinstance(required, list) or any(not isinstance(k, str) for k in required)
                or len(required) != len(set(required)) or set(required) - props.keys()):
            raise SchemaError(f"Invalid required fields at {path}.")
        if schema.get("additionalProperties", False) is not False:
            raise SchemaError(f"Open-ended additionalProperties is unsupported at {path}.")
        out.update(properties={k: transport_schema(v, f"{path}.{k}", _depth=_depth + 1) for k, v in props.items()},
                   required=list(required), additionalProperties=False)
    elif set(schema) & {"properties", "required", "additionalProperties"}:
        raise SchemaError(f"Object keywords require object type at {path}.")
    if "array" in kinds:
        if "items" not in schema:
            raise SchemaError(f"Array items schema required at {path}.")
        out["items"] = transport_schema(schema["items"], f"{path}[]", _depth=_depth + 1)
    elif set(schema) & {"items", "minItems", "maxItems"}:
        raise SchemaError(f"Array keywords require array type at {path}.")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise SchemaError(f"Invalid nonnegative integer {key} at {path}.")
    for key in ("minLength", "maxLength", "pattern"):
        if key in schema and "string" not in kinds:
            raise SchemaError(f"String keywords require string type at {path}.")
    numeric = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    for key in numeric:
        if key in schema:
            if (not set(kinds) & {"integer", "number"} or type(schema[key]) not in {int, float}
                    or not math.isfinite(schema[key]) or key == "multipleOf" and schema[key] <= 0):
                raise SchemaError(f"Invalid numeric constraint {key} at {path}.")
    for minimum, maximum in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
        if minimum in schema and maximum in schema and schema[minimum] > schema[maximum]:
            raise SchemaError(f"Contradictory schema bounds at {path}.")
    if "pattern" in schema:
        try:
            if not isinstance(schema["pattern"], str):
                raise TypeError
            re.compile(schema["pattern"])
        except (re.error, TypeError):
            raise SchemaError(f"Invalid local pattern at {path}.") from None
    removed = {k: schema[k] for k in LOCAL_CONSTRAINTS if k in schema}
    if "minItems" in schema:
        if schema["minItems"] in (0, 1):
            out["minItems"] = schema["minItems"]
        else:
            removed["minItems"] = schema["minItems"]
    if removed:
        out["description"] = (out.get("description", "") + " Locally enforced constraints: "
                              + json.dumps(removed, sort_keys=True, allow_nan=False) + ".").strip()
    return out


def transport_tools(tools: list[dict]) -> list[dict]:
    """Validate every tool recursively before serialization, including raw loops."""
    if not isinstance(tools, list) or len(tools) > 20:
        raise SchemaError("At most twenty strict tools may be sent per request.")
    output, names = [], set()
    optional_count = union_count = 0

    def count(node):
        nonlocal optional_count, union_count
        union_count += int(isinstance(node.get("type"), list) and len(node["type"]) > 1)
        props = node.get("properties", {})
        optional_count += len(set(props) - set(node.get("required", [])))
        for child in props.values():
            count(child)
        if "items" in node:
            count(node["items"])

    for tool in tools:
        if not isinstance(tool, dict) or set(tool) - {"name", "description", "input_schema", "strict"}:
            raise SchemaError("Unsupported tool definition fields.")
        name = tool.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in names:
            raise SchemaError("Invalid or duplicate tool name.")
        if tool.get("strict") is not True or not isinstance(tool.get("description", ""), str):
            raise SchemaError("Every application tool must explicitly enable strict mode.")
        schema = transport_schema(tool.get("input_schema"), path=f"tool.{name}")
        if schema["type"] != "object":
            raise SchemaError("Tool root must be an object.")
        count(schema)
        output.append({**deepcopy(tool), "input_schema": schema})
        names.add(name)
    if optional_count > 24 or union_count > 16:
        raise SchemaError("Strict schema exceeds documented optional/union complexity limits.")
    return output


def validate_instance(value: Any, schema: Mapping[str, Any], path: str = "result") -> None:
    """Validate using the ORIGINAL application schema, never the relaxed copy."""
    transport_schema(schema)  # Invalid definitions cannot bypass the local gate.
    _validate(value, schema, path)


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> None:
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "number": type(value) in {int, float}, "integer": type(value) is int, "null": value is None}
    kind = schema["type"]
    kinds = kind if isinstance(kind, list) else [kind]
    if not any(valid[k] for k in kinds):
        raise SchemaError(f"Schema validation failed at {path}: expected {kind}.")
    if type(value) in {int, float} and not math.isfinite(value):
        raise SchemaError(f"Non-finite number at {path}.")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise SchemaError(f"Invalid enum at {path}.")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise SchemaError(f"Invalid const at {path}.")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise SchemaError(f"Missing required field at {path}.")
        if value.keys() - props.keys():
            raise SchemaError(f"Unexpected field at {path}.")
        for key, item in value.items():
            _validate(item, props[key], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10000):
            raise SchemaError(f"Array length outside schema at {path}.")
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 200000):
            raise SchemaError(f"String length outside schema at {path}.")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaError(f"String pattern mismatch at {path}.")
    if type(value) in {int, float}:
        if (value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf)
                or "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]
                or "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]):
            raise SchemaError(f"Number outside schema at {path}.")
        if "multipleOf" in schema and Decimal(str(value)) % Decimal(str(schema["multipleOf"])) != 0:
            raise SchemaError(f"Number is not a required multiple at {path}.")
