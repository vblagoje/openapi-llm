"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from openapi_llm.core.schema_conversion import ConverterConfig
from openapi_llm.core.spec import OpenAPISpecification


class LLMProvider(ABC):
    """
    Base interface for LLM provider implementations.

    Defines the contract that all LLM providers must implement to support
    OpenAPI to LLM function/tool conversion.
    """

    @abstractmethod
    def converter(
        self,
    ) -> Callable[
        [OpenAPISpecification, Optional[ConverterConfig]], List[Dict[str, Any]]
    ]:
        """
        Returns a function that converts OpenAPI specs to provider-specific function/tool definitions.

        The returned converter transforms OpenAPI specifications into the format expected by
        the specific LLM provider (e.g., OpenAI, Anthropic, Cohere, etc).

        :returns: Function that converts OpenAPI specs to LLM-specific function definitions.
        """

    @abstractmethod
    def payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns a function that extracts prepared tools invocation payload from LLM responses.

        The returned parser knows how to extract the function/tool call arguments from the
        specific format used by each LLM provider.

        :returns: Function that extracts function call arguments from provider-specific responses.
        """
