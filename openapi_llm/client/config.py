from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.auth import (
    create_api_key_authenticator,
    create_bearer_token_authenticator,
)
from openapi_llm.core.provider import LLMProvider
from openapi_llm.core.schema_conversion import ConverterConfig
from openapi_llm.core.spec import OpenAPISpecification
from openapi_llm.providers.openai import OpenAIProvider
from openapi_llm.utils import normalize_tool_definition, send_request


class ClientConfig:
    """
    Configuration for OpenAPI to LLM function calling.

    Manages the configuration needed to convert OpenAPI specifications to LLM functions
    and handle authentication and request processing.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        openapi_spec: OpenAPISpecification,
        credentials: Optional[Any] = None,
        request_sender: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        llm_provider: Optional[LLMProvider] = None,
        operations_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Initialize client configuration.

        :param openapi_spec: OpenAPI specification to use.
        :param credentials: Authentication credentials.
        :param request_sender: Custom function for sending HTTP requests.
        :param llm_provider: LLM provider implementation to use.
        :param operations_filter: Deprecated. Use converter_config instead.
        :param converter_config: Configuration for OpenAPI to LLM function conversion.
        :raises ValueError: If the OpenAPI specification format is invalid.
        """
        self.openapi_spec = openapi_spec
        self.credentials = credentials
        self.request_sender = request_sender or send_request
        self.llm_provider = llm_provider or OpenAIProvider()
        self.converter_config = ConverterConfig(filter_fn=operations_filter)

    def get_authenticator(self) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        """
        Get the authentication function for request processing.

        Creates an authenticator function that applies the configured credentials
        according to the security schemes defined in the OpenAPI spec.

        :returns: Function that applies authentication to requests.
        :raises ValueError: If the credentials type is not supported.
        """
        security_schemes = self.openapi_spec.get_security_schemes()
        if not self.credentials:
            return lambda security_scheme, request: None  # No-op function
        if isinstance(self.credentials, str):
            return self._create_authenticator_from_credentials(
                self.credentials, security_schemes
            )
        raise ValueError(f"Unsupported credentials type: {type(self.credentials)}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the LLM tool/function definitions for the OpenAPI spec.

        Converts the OpenAPI specification into LLM-specific function/tool definitions
        using the configured LLM provider.

        :returns: List of tool/function definitions ready for LLM use.
        """
        tools_definitions = self.llm_provider.converter()(
            self.openapi_spec, self.converter_config
        )
        return [normalize_tool_definition(t) for t in tools_definitions]

    def get_payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the parser for LLM tool call invocation payloads.

        Returns a function that can extract the actual tool call payload
        from LLM-specific response formats.

        :returns: Function that extracts tool call payloads from LLM responses.
        """
        return self.llm_provider.payload_extractor()

    def _create_authenticator_from_credentials(
        self, credentials: str, security_schemes: Dict[str, Any]
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        """
        Create an authenticator function from provided credentials.

        :param credentials: Authentication credentials string.
        :param security_schemes: Security schemes from OpenAPI spec.
        :returns: Function that applies authentication to requests.
        :raises ValueError: If unable to create authenticator from credentials.
        """
        for scheme in security_schemes.values():
            if scheme["type"] == "apiKey":
                return create_api_key_authenticator(api_key=credentials)
            if scheme["type"] == "http":
                return create_bearer_token_authenticator(token=credentials)
            raise ValueError(
                f"Unsupported authentication type '{scheme['type']}' provided."
            )
        raise ValueError(
            f"Unable to create authentication from provided credentials: {credentials}"
        )
