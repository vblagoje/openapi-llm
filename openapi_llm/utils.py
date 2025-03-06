import dataclasses
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import requests

if TYPE_CHECKING:
    from openapi_llm.client import ClientConfig
    from openapi_llm.core.spec import Operation

logger = logging.getLogger(__name__)


def normalize_tool_definition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes the given tool definition by adjusting its properties to LLM requirements.

    While various LLMs have slightly different requirements for tool definitions, we normalize them to a common
    format that is compatible with OpenAI, Anthropic, and Cohere LLMs:
    - tool names have to match the pattern ^[a-zA-Z0-9_]+$ and are truncated to 64 characters
    - tool/parameter descriptions are truncated to 1024 characters

    For reference on tool definition formats, see:
        - https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models#basic-concepts
        - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        - https://docs.cohere.com/docs/tool-use

    :param data: The function calling definition(s) to normalize.
    :returns: A normalized function calling definition.
    """
    normalized_data: Dict[str, Any] = {}
    for key, value in data.items():
        # all LLMs tool definitions have tool (function) name and description on the same level
        # if we find it then normalize the function name
        if key == "name" and "description" in data.keys():
            normalized_data[key] = normalize_function_name(value)
        elif key == "description":
            normalized_data[key] = value[:1024]
        elif isinstance(value, dict):
            # recursively normalize nested descriptions (e.g. tool parameters)
            normalized_data[key] = normalize_tool_definition(value)
        else:
            normalized_data[key] = value
    return normalized_data


def normalize_function_name(name: str) -> str:
    """
    Normalizes the function name to match the LLM function naming requirements.

    While various LLMs have slightly different requirements for tool (function) names, we normalize them to
    a common format that is compatible with OpenAI, Anthropic, and Cohere LLMs:
    - The function name must match the pattern ^[a-zA-Z0-9_]+$
    - The function name must be truncated to 64 characters

    :param name: The original function name.
    :returns: A normalized function name that matches the allowed pattern.
    """
    # Replace characters not allowed in the pattern with underscores
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    # Remove leading and trailing underscores and truncate to 64 characters
    return normalized.strip("_")[:64]


def create_function_payload_extractor(
    arguments_field_name: str,
) -> Callable[[Any], Dict[str, Any]]:
    """
    Extracts invocation payload from a given LLM completion containing function invocation.

    :param arguments_field_name: The name of the field containing the function arguments.
    :return: A function that extracts the function invocation details from the LLM payload.
    """

    def _extract_function_invocation(payload: Any) -> Dict[str, Any]:
        """
        Extract the function invocation details from the payload.

        :param payload: The LLM fc payload to extract the function invocation details from.
        """
        fields_and_values = _search(payload, arguments_field_name)
        if fields_and_values:
            arguments = fields_and_values.get(arguments_field_name)
            if not isinstance(arguments, (str, dict)):
                raise ValueError(
                    f"Invalid {arguments_field_name} type {type(arguments)} for function call, expected str/dict"
                )
            return {
                "name": fields_and_values.get("name"),
                "arguments": (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                ),
            }
        return {}

    return _extract_function_invocation


def _get_dict_converter(
    obj: Any, method_names: Optional[List[str]] = None
) -> Union[Callable[[], Dict[str, Any]], None]:
    method_names = method_names or [
        "model_dump",
        "dict",
    ]  # search for pydantic v2 then v1
    for attr in method_names:
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            return getattr(obj, attr)
    return None


def _is_primitive(obj) -> bool:
    return isinstance(obj, (int, float, str, bool, type(None)))


def _required_fields(arguments_field_name: str) -> List[str]:
    return ["name", arguments_field_name]


def _search(payload: Any, arguments_field_name: str) -> Dict[str, Any]:
    if _is_primitive(payload):
        return {}
    if dict_converter := _get_dict_converter(payload):
        payload = dict_converter()
    elif dataclasses.is_dataclass(payload):
        # Cast payload to Any to satisfy mypy 1.11.0
        payload = dataclasses.asdict(cast(Any, payload))
    if isinstance(payload, dict):
        if all(field in payload for field in _required_fields(arguments_field_name)):
            # this is the payload we are looking for
            return payload
        for value in payload.values():
            result = _search(value, arguments_field_name)
            if result:
                return result
    elif isinstance(payload, list):
        for item in payload:
            result = _search(item, arguments_field_name)
            if result:
                return result
    return {}


def send_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an HTTP request and return the response.

    :param request: The request to send.
    :returns: The response from the server.
    """
    url = request["url"]
    headers = {**request.get("headers", {})}
    try:
        response = requests.request(
            request["method"],
            url,
            headers=headers,
            params=request.get("params", {}),
            json=request.get("json"),
            auth=request.get("auth"),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"HTTP error occurred: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.warning("Request error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"HTTP error occurred: {e}") from e
    except Exception as e:
        logger.warning("An error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"An error occurred: {e}") from e


class HttpClientError(Exception):
    """Exception raised for errors in the HTTP client."""


def build_request(operation: "Operation", config: "ClientConfig", **kwargs) -> Dict[str, Any]:
    """
    Build an HTTP request for the operation.

    :param operation: The operation to build the request for.
    :param config: The client configuration.
    :param kwargs: The arguments to use for building the request.
    :returns: The HTTP request as a dictionary.
    :raises ValueError: If a required parameter is missing.
    :raises NotImplementedError: If the request body content type is not supported. We only support JSON payloads.
    """
    path = operation.path
    for parameter in operation.get_parameters("path"):
        param_value = kwargs.get(parameter["name"], None)
        if param_value:
            path = path.replace(f"{{{parameter['name']}}}", str(param_value))
        elif parameter.get("required", False):
            raise ValueError(f"Missing required path parameter: {parameter['name']}")
    url = config.base_url + path if isinstance(config.base_url, str) else operation.get_server(config.base_url) + path
    # method
    method = operation.method.lower()
    # headers
    headers = {}
    for parameter in operation.get_parameters("header"):
        param_value = kwargs.get(parameter["name"], None)
        if param_value:
            headers[parameter["name"]] = str(param_value)
        elif parameter.get("required", False):
            raise ValueError(f"Missing required header parameter: {parameter['name']}")
    # query params
    query_params = {}
    for parameter in operation.get_parameters("query"):
        param_value = kwargs.get(parameter["name"], None)
        if param_value:
            query_params[parameter["name"]] = param_value
        elif parameter.get("required", False):
            raise ValueError(f"Missing required query parameter: {parameter['name']}")

    json_payload = None
    request_body = operation.request_body
    if request_body:
        content = request_body.get("content", {})
        if "application/json" in content:
            json_payload = {**kwargs}
        else:
            raise NotImplementedError("Request body content type not supported")
    return {
        "url": url,
        "method": method,
        "headers": headers,
        "params": query_params,
        "json": json_payload,
    }


def apply_authentication(
    auth_strategy: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    operation: "Operation",
    request: Dict[str, Any],
):
    """
    Apply the authentication strategy to the given request.

    :param auth_strategy: The authentication strategy to apply.
    This is a function that takes a security scheme and a request as arguments (at runtime)
    and applies the authentication
    :param operation: The operation to apply the authentication to.
    :param request: The request to apply the authentication to.
    """
    security_requirements = operation.security_requirements
    security_schemes = operation.spec_dict.get("components", {}).get(
        "securitySchemes", {}
    )
    if security_requirements:
        for requirement in security_requirements:
            for scheme_name in requirement:
                if scheme_name in security_schemes:
                    security_scheme = security_schemes[scheme_name]
                    auth_strategy(security_scheme, request)
                break
