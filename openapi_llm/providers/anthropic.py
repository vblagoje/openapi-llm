"""Anthropic LLM provider implementation."""

from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.provider import LLMProvider
from openapi_llm.core.schema_conversion import ConverterConfig, anthropic_converter
from openapi_llm.core.spec import OpenAPISpecification
from openapi_llm.utils import create_function_payload_extractor


class AnthropicProvider(LLMProvider):
    """
    Provider implementation for Anthropic's tool calling.

    This provider handles the conversion of OpenAPI specifications to Anthropic's tool
    calling format and extraction of function call payloads from Anthropic responses.
    """

    def converter(
        self,
    ) -> Callable[
        [OpenAPISpecification, Optional[ConverterConfig]], List[Dict[str, Any]]
    ]:
        """
        Return the Anthropic-specific schema converter.

        :returns: Function that converts OpenAPI specs to Anthropic tool definitions
        """
        return anthropic_converter

    def payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Return the payload extractor for Anthropic responses.

        The extractor looks for the 'input' field in the response and processes it
        according to Anthropic's tool calling format.

        :returns: Function that extracts tool call details from Anthropic responses
        """
        return create_function_payload_extractor("input")
