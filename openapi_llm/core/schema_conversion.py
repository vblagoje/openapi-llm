import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.spec import (
    MIN_REQUIRED_OPENAPI_SPEC_VERSION,
    VALID_HTTP_METHODS,
    OpenAPISpecification,
    create_operation_id,
)

logger = logging.getLogger(__name__)


@dataclass
class ConverterConfig:
    """
    Configuration for OpenAPI to LLM function conversion.

    :param filter_fn: Optional function to filter OpenAPI operations based on custom criteria.
                     Takes operation dict as input and returns bool.
    :param max_functions: Optional maximum number of functions to convert.
    """

    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    max_functions: Optional[int] = None


def openai_converter(
    schema: OpenAPISpecification,
    config: Optional[ConverterConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Converts OpenAPI specification to a list of tools suitable for OpenAI LLM tool calling.

    See https://platform.openai.com/docs/guides/function-calling for more information about OpenAI's tool schema.
    :param schema: The OpenAPI specification to convert.
    :param config: Configuration controlling the conversion process.
    :returns: A list of dictionaries, each dictionary representing an OpenAI tool definition.
    """
    tool_definitions = _openapi_to_tools(
        schema.spec_dict, "parameters", _convert_operation_to_openai_schema, config
    )
    return [{"type": "function", "function": tool} for tool in tool_definitions]


def anthropic_converter(
    schema: OpenAPISpecification,
    config: Optional[ConverterConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Converts an OpenAPI specification to a list of tool definitions for Anthropic LLM tool calling.

    See https://docs.anthropic.com/en/docs/tool-use for more information about Anthropic's tool schema.

    :param schema: The OpenAPI specification to convert.
    :param config: Configuration controlling the conversion process.
    :returns: A list of dictionaries, each dictionary representing Anthropic tool definition.
    """
    return _openapi_to_tools(
        schema.spec_dict, "input_schema", _convert_operation_to_openai_schema, config
    )


def cohere_converter(
    schema: OpenAPISpecification,
    config: Optional[ConverterConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Converts an OpenAPI specification to a list of function definitions for Cohere LLM function calling.

    See https://docs.cohere.com/docs/tool-use for more information about Cohere's function schema.

    :param schema: The OpenAPI specification to convert.
    :param config: Configuration controlling the conversion process.
    :returns: A list of dictionaries, each representing a Cohere style function definition.
    """
    return _openapi_to_tools(
        schema.spec_dict,
        "not important for cohere",
        _convert_operation_to_cohere_schema,
        config,
    )


def _openapi_to_tools(
    service_openapi_spec: Dict[str, Any],
    parameters_name: str,
    operation_converter: Callable[[Dict[str, Any], str], Dict[str, Any]],
    config: Optional[ConverterConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts operations from the OpenAPI specification, converts them into a tool schema.

    :param service_openapi_spec: The OpenAPI specification to extract operations from.
    :param parameters_name: The name of the parameters field in the tool schema.
    :param operation_converter: Function that converts an OpenAPI operation into an LLM-specific tool schema.
    :param config: Configuration controlling the conversion process.
    :returns: A list of dictionaries, each dictionary representing a tool schema.
    """

    # Doesn't enforce rigid spec validation because that would require a lot of dependencies
    # We check the version and require minimal fields to be present, so we can extract operations
    spec_version = service_openapi_spec.get("openapi")
    if not spec_version:
        raise ValueError(
            f"Invalid OpenAPI spec provided. Could not extract version from {service_openapi_spec}"
        )
    service_openapi_spec_version = int(spec_version.split(".")[0])
    # Compare the versions
    if service_openapi_spec_version < MIN_REQUIRED_OPENAPI_SPEC_VERSION:
        raise ValueError(
            f"Invalid OpenAPI spec version {service_openapi_spec_version}. Must be "
            f"at least {MIN_REQUIRED_OPENAPI_SPEC_VERSION}."
        )
    operations: List[Dict[str, Any]] = []
    for path, path_value in service_openapi_spec["paths"].items():
        for path_key, operation_spec in path_value.items():
            if path_key.lower() in VALID_HTTP_METHODS:
                if "operationId" not in operation_spec:
                    operation_spec["operationId"] = create_operation_id(path, path_key)

                # Apply the filter if configured
                if config and config.filter_fn and not config.filter_fn(operation_spec):
                    continue

                # parse (and register) this operation
                ops_dict = operation_converter(operation_spec, parameters_name)
                if ops_dict:
                    operations.append(ops_dict)
                    # Check max_functions limit if configured
                    if (
                        config
                        and config.max_functions
                        and len(operations) >= config.max_functions
                    ):
                        return operations
    return operations


def _convert_operation_to_openai_schema(
    resolved_spec: Dict[str, Any], parameters_name: str
) -> Dict[str, Any]:
    """
    Converts an OpenAPI operation into OpenAI's function schema format.

    Transforms the OpenAPI operation specification into OpenAI's expected function
    definition format, including name, description, and parameters schema.

    :param resolved_spec: The resolved OpenAPI operation specification.
    :param parameters_name: The name of the parameters field in the function schema.
    :returns: A dictionary containing the OpenAI function schema.
    """
    if not isinstance(resolved_spec, dict):
        logger.warning(
            "Invalid OpenAPI spec format provided. Could not extract function."
        )
        return {}

    function_name = resolved_spec.get("operationId")
    description = resolved_spec.get("description") or resolved_spec.get("summary", "")

    # Return valid schema even if no parameters are present
    if function_name and description:
        schema: Dict[str, Any] = {"type": "object", "properties": {}}

        # requestBody section
        req_body_schema = (
            resolved_spec.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        if "properties" in req_body_schema:
            for prop_name, prop_schema in req_body_schema["properties"].items():
                schema["properties"][prop_name] = _parse_property_attributes(prop_schema)
            if "required" in req_body_schema:
                schema.setdefault("required", []).extend(req_body_schema["required"])

        # parameters section
        for param in resolved_spec.get("parameters", []):
            if "schema" in param:
                schema_dict = _parse_property_attributes(param["schema"])
                # these attributes are not in param[schema] level but on param level
                useful_attributes = ["description", "pattern", "enum"]
                schema_dict.update(
                    {key: param[key] for key in useful_attributes if param.get(key)}
                )
                schema["properties"][param["name"]] = schema_dict
                if param.get("required", False):
                    schema.setdefault("required", []).append(param["name"])

        return {
            "name": function_name,
            "description": description,
            parameters_name: schema,
        }

    logger.warning(
        "Invalid OpenAPI spec format provided. Could not extract function from %s",
        resolved_spec,
    )
    return {}


def _parse_property_attributes(
    property_schema: Dict[str, Any], include_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Recursively parses the attributes of a property schema.

    :param property_schema: The property schema to parse.
    :param include_attributes: The attributes to include in the parsed schema.
    :returns: A dictionary containing the parsed property schema.
    """
    include_attributes = include_attributes or ["description", "pattern", "enum"]
    schema_type = property_schema.get("type")
    parsed_schema = {"type": schema_type} if schema_type else {}
    for attr in include_attributes:
        if attr in property_schema:
            parsed_schema[attr] = property_schema[attr]
    if schema_type == "object":
        properties = property_schema.get("properties", {})
        parsed_properties = {
            prop_name: _parse_property_attributes(prop, include_attributes)
            for prop_name, prop in properties.items()
        }
        parsed_schema["properties"] = parsed_properties
        if "required" in property_schema:
            parsed_schema["required"] = property_schema["required"]
    elif schema_type == "array":
        items = property_schema.get("items", {})
        parsed_schema["items"] = _parse_property_attributes(items, include_attributes)
    return parsed_schema


def _convert_operation_to_cohere_schema(
    operation: Dict[str, Any], ignored_param: str
) -> Dict[str, Any]:
    """
    Converts an OpenAPI operation into Cohere's function schema format.

    Transforms the OpenAPI operation specification into Cohere's expected function
    definition format, including name, description, and parameter definitions.

    :param operation: The operation specification to convert.
    :param ignored_param: Ignored parameter, maintained for API compatibility with OpenAI converter.
    :returns: A dictionary containing the Cohere function schema.
    """
    function_name = operation.get("operationId")
    description = operation.get("description") or operation.get("summary", "")
    parameter_definitions = _parse_parameters(operation)
    if function_name:
        return {
            "name": function_name,
            "description": description,
            "parameter_definitions": parameter_definitions,
        }
    logger.warning("Operation missing operationId, cannot create function definition.")
    return {}


def _parse_parameters(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the parameters from an operation specification.

    :param operation: The operation specification to parse.
    :returns: A dictionary containing the parsed parameters.
    """
    parameters = {}
    for param in operation.get("parameters", []):
        if "schema" in param:
            parameters[param["name"]] = _parse_schema(
                param["schema"],
                param.get("required", False),
                param.get("description", ""),
            )
    if "requestBody" in operation:
        content = (
            operation["requestBody"].get("content", {}).get("application/json", {})
        )
        if "schema" in content:
            schema_properties = content["schema"].get("properties", {})
            required_properties = content["schema"].get("required", [])
            for name, schema in schema_properties.items():
                parameters[name] = _parse_schema(
                    schema, name in required_properties, schema.get("description", "")
                )
    return parameters


def _parse_schema(
    schema: Dict[str, Any], required: bool, description: str
) -> Dict[str, Any]:  # noqa: FBT001
    """
    Parses a schema part of an operation specification.

    :param schema: The schema to parse.
    :param required: Whether the schema is required.
    :param description: The description of the schema.
    :returns: A dictionary containing the parsed schema.
    """
    schema_type = _get_type(schema)
    if schema_type == "object":
        # Recursive call for complex types
        properties = schema.get("properties", {})
        nested_parameters = {
            name: _parse_schema(
                schema=prop_schema,
                required=bool(name in schema.get("required", [])),
                description=prop_schema.get("description", ""),
            )
            for name, prop_schema in properties.items()
        }
        return {
            "type": schema_type,
            "description": description,
            "properties": nested_parameters,
            "required": required,
        }
    return {"type": schema_type, "description": description, "required": required}


def _get_type(schema: Dict[str, Any]) -> str:
    type_mapping = {
        "integer": "int",
        "string": "str",
        "boolean": "bool",
        "number": "float",
        "object": "object",
        "array": "list",
    }
    schema_type = schema.get("type", "object")
    if schema_type not in type_mapping:
        raise ValueError(f"Unsupported schema type {schema_type}")
    return type_mapping[schema_type]
