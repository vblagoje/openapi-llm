"""Cohere LLM provider implementation."""

from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.provider import LLMProvider
from openapi_llm.core.schema_conversion import ConverterConfig, cohere_converter
from openapi_llm.core.spec import OpenAPISpecification
from openapi_llm.utils import create_function_payload_extractor


class CohereProvider(LLMProvider):
    """
    Provider implementation for Cohere's tool calling.

    This provider handles the conversion of OpenAPI specifications to Cohere's tool
    calling format and extraction of function call payloads from Cohere responses.
    """

    def converter(
        self,
    ) -> Callable[
        [OpenAPISpecification, Optional[ConverterConfig]], List[Dict[str, Any]]
    ]:
        """
        Return the Cohere-specific schema converter.

        :returns: Function that converts OpenAPI specs to Cohere tool definitions
        """
        return cohere_converter

    def payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Return the payload extractor for Cohere responses.

        The extractor looks for the 'parameters' field in the response and processes it
        according to Cohere's tool calling format.

        :returns: Function that extracts tool call details from Cohere responses
        """
        return create_function_payload_extractor("parameters")
