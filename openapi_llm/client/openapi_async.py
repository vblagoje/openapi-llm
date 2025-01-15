# pylint: disable=R0801
# the above is disabling a check for duplicate code in the file taken from openapi_client.py
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp

from openapi_llm.client.config import ClientConfig, create_client_config
from openapi_llm.utils import apply_authentication, build_request

# Ignore duplicate code in this file and in openapi.py
# pylint: disable=R0801

class AsyncOpenAPIClient:
    """
    An async client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfig):
        """
        Initialize the AsyncOpenAPIClient with a ClientConfig.

        :param client_config: The configuration for the OpenAPI client.
        """
        self.client_config = client_config
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Tool definitions derived from the OpenAPI specification suitable for LLM tool calling.

        :returns: A list of tool definitions that can be used with LLM tool/function calling.
        """
        return self.client_config.get_tool_definitions()


    async def __aenter__(self) -> "AsyncOpenAPIClient":
        """Enter the async context manager."""
        if not self._session:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        if self._session and self._owns_session:
            await self._session.close()
            self._session = None
            self._owns_session = False

    async def setup(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        """
        Set up the client with an optional session. If no session is provided, creates a new one.

        :param session: Optional aiohttp.ClientSession to use. If not provided, creates a new one.
        """
        if session:
            self._session = session
            self._owns_session = False
        else:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

    async def cleanup(self) -> None:
        """
        Clean up resources. Only closes the session if it was created by this client.
        """
        if self._session and self._owns_session:
            await self._session.close()
            self._session = None
            self._owns_session = False

    async def invoke(self, function_payload: Any) -> Any:
        """
        Invokes a remote endpoint asynchronously based on the function specification from an LLM.

        :param function_payload: A dictionary containing:
            - 'name': The OpenAPI operation ID to invoke
            - 'arguments': The parameters to pass to the operation
        :returns: The JSON response from the remote service
        :raises AsyncOpenAPIClientError: If the function payload is invalid or cannot be processed
        :raises AsyncHttpClientError: If the HTTP request fails or times out
        """
        fn_invocation_payload = {}
        try:
            fn_extractor = self.client_config.get_payload_extractor()
            fn_invocation_payload = fn_extractor(function_payload)
        except Exception as e:
            raise AsyncOpenAPIClientError(
                f"Error extracting function invocation payload: {str(e)}"
            ) from e

        if (
            "name" not in fn_invocation_payload
            or "arguments" not in fn_invocation_payload
        ):
            raise AsyncOpenAPIClientError(
                f"Function invocation payload does not contain 'name' or 'arguments' keys: {fn_invocation_payload}, "
                f"the payload extraction function may be incorrect."
            )
        # fn_invocation_payload, if not empty, guaranteed to have "name" and "arguments" keys from here on
        operation = self.client_config.openapi_spec.find_operation_by_id(
            fn_invocation_payload["name"]
        )
        request = build_request(operation, **fn_invocation_payload["arguments"])
        apply_authentication(self.client_config.get_authenticator(), operation, request)

        if not self._session:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        try:
            async with self._session.request(
                request["method"],
                request["url"],
                headers=request.get("headers", {}),
                params=request.get("params", {}),
                json=request.get("json"),
                timeout=30
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise AsyncHttpClientError(f"HTTP error occurred: {e}") from e
        except Exception as e:
            raise AsyncHttpClientError(f"An error occurred: {e}") from e

    @classmethod
    def from_spec(
        cls,
        openapi_spec: Union[str, Path],
        config_factory: Optional[Callable[[Union[str, Path]], ClientConfig]] = None,
        **kwargs
    ) -> "AsyncOpenAPIClient":
        """
        Constructs an AsyncOpenAPIClient using provided OpenAPI specification.

        This method provides an extensible mechanism for constructing clients. By default,
        it uses the `create_client_config` factory to parse the OpenAPI spec and construct
        a `ClientConfig`. Users can override the configuration logic by supplying their
        own `config_factory` callable.

        :param openapi_spec: OpenAPI spec as a file path, URL, or raw string.
        :param config_factory: A factory function for creating the ClientConfig.
            If not provided, defaults to using `create_client_config`.
        :param kwargs: Additional ClientConfig parameters (e.g., credentials).
        :returns: Configured AsyncOpenAPIClient instance.

        Examples
        --------
        Default usage:

        >>> client = AsyncOpenAPIClient.from_spec(
        ...     "https://example.com/openapi.yaml",
        ...     credentials="my_api_key"
        ... )

        Custom configuration with a factory:

        >>> def custom_factory(spec: Union[str, Path], **kwargs):
        ...     spec_obj = create_openapi_spec(spec)
        ...     validate_spec(spec_obj.spec_dict)  # Custom validation
        ...     return ClientConfig(openapi_spec=spec_obj, **kwargs)
        ...
        >>> client = AsyncOpenAPIClient.from_spec(
        ...     "https://example.com/openapi.yaml",
        ...     config_factory=custom_factory
        ... )
        """
        if config_factory:
            config = config_factory(openapi_spec, **kwargs)
        else:
            config = create_client_config(openapi_spec, **kwargs)
        return cls(config)

class AsyncOpenAPIClientError(Exception):
    """Exception raised for errors in the async OpenAPI client."""


class AsyncHttpClientError(Exception):
    """Exception raised for HTTP-related errors during async service invocation."""
