"""OpenAI LLM provider implementation."""

from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.provider import LLMProvider
from openapi_llm.core.schema_conversion import ConverterConfig, openai_converter
from openapi_llm.core.spec import OpenAPISpecification
from openapi_llm.utils import create_function_payload_extractor


class OpenAIProvider(LLMProvider):
    """
    Provider implementation for OpenAI's tool calling.

    This provider handles the conversion of OpenAPI specifications to OpenAI's tool
    calling format and extraction of function call payloads from OpenAI responses.
    """

    def converter(
        self,
    ) -> Callable[
        [OpenAPISpecification, Optional[ConverterConfig]], List[Dict[str, Any]]
    ]:
        """
        Return the OpenAI-specific schema converter.

        :returns: Function that converts OpenAPI specs to OpenAI tool definitions
        """
        return openai_converter

    def payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Return the payload extractor for OpenAI responses.

        The extractor looks for the 'arguments' field in the LLM tool calling response
        and processes it according to OpenAI's tool calling format.

        :returns: Function that extracts tool call details from OpenAI responses
        """
        return create_function_payload_extractor("arguments")
