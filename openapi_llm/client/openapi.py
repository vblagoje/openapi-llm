from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from openapi_llm.client.config import ClientConfig, create_client_config
from openapi_llm.utils import apply_authentication, build_request


class OpenAPIClient:
    """
    A client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfig):
        self.client_config = client_config

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Tool definitions derived from the OpenAPI specification suitable for LLM tool calling.

        :returns: A list of tool definitions that can be used with LLM tool/function calling.
        """
        return self.client_config.get_tool_definitions()

    def invoke(self, function_payload: Any) -> Any:
        """
        Invokes a function specified in the function payload.

        :param function_payload: The function payload containing the details of the function to be invoked.
        :returns: The response from the service after invoking the function.
        :raises OpenAPIClientError: If the function invocation payload cannot be extracted from the function payload.
        :raises HttpClientError: If an error occurs while sending the request and receiving the response.
        """
        fn_invocation_payload = {}
        try:
            fn_extractor = self.client_config.get_payload_extractor()
            fn_invocation_payload = fn_extractor(function_payload)
        except Exception as e:
            raise OpenAPIClientError(
                f"Error extracting function invocation payload: {str(e)}"
            ) from e

        if (
            "name" not in fn_invocation_payload
            or "arguments" not in fn_invocation_payload
        ):
            raise OpenAPIClientError(
                f"Function invocation payload does not contain 'name' or 'arguments' keys: {fn_invocation_payload}, "
                f"the payload extraction function may be incorrect."
            )
        # fn_invocation_payload, if not empty, guaranteed to have "name" and "arguments" keys from here on
        operation = self.client_config.openapi_spec.find_operation_by_id(
            fn_invocation_payload["name"]
        )
        request = build_request(operation, **fn_invocation_payload["arguments"])
        apply_authentication(self.client_config.get_authenticator(), operation, request)
        return self.client_config.request_sender(request)

    @classmethod
    def from_spec(
        cls,
        openapi_spec: Union[str, Path],
        config_factory: Optional[Callable[[Union[str, Path]], ClientConfig]] = None,
        **kwargs
    ) -> "OpenAPIClient":
        """
        Constructs an OpenAPIClient using provided OpenAPI specification.

        This method provides an extensible mechanism for constructing clients. By default,
        it uses the `create_client_config` factory to parse the OpenAPI spec and construct
        a `ClientConfig`. Users can override the configuration logic by supplying their
        own `config_factory` callable.

        :param openapi_spec: OpenAPI spec as a file path, URL, or raw string.
        :param config_factory: A factory function for creating the ClientConfig.
            If not provided, defaults to using `create_client_config`.
        :param kwargs: Additional ClientConfig parameters (e.g., credentials).
        :returns: Configured OpenAPIClient instance.

        Examples
        --------
        Default usage:

        >>> client = OpenAPIClient.from_spec(
        ...     "https://example.com/openapi.yaml",
        ...     credentials="my_api_key"
        ... )

        Custom configuration with a factory:

        >>> def custom_factory(spec: Union[str, Path], **kwargs):
        ...     spec_obj = create_openapi_spec(spec)
        ...     validate_spec(spec_obj.spec_dict)  # Custom validation
        ...     return ClientConfig(openapi_spec=spec_obj, **kwargs)
        ...
        >>> client = OpenAPIClient.from_spec(
        ...     "https://example.com/openapi.yaml",
        ...     config_factory=custom_factory
        ... )
        """
        if config_factory:
            config = config_factory(openapi_spec, **kwargs)
        else:
            config = create_client_config(openapi_spec, **kwargs)
        return cls(config)


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""


class HttpClientError(Exception):
    """Exception raised for HTTP-related errors during service invocation."""
