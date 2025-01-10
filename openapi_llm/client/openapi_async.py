# pylint: disable=R0801
# the above is disabling a check for duplicate code in the file taken from openapi_client.py
from typing import Any, Optional

import aiohttp

from openapi_llm.client.config import ClientConfig
from openapi_llm.utils import apply_authentication, build_request


class AsyncOpenAPIClient:
    """
    An async client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfig):
        self.client_config = client_config
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False

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
        Invokes a function specified in the function payload asynchronously.

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


class AsyncOpenAPIClientError(Exception):
    """Exception raised for errors in the async OpenAPI client."""


class AsyncHttpClientError(Exception):
    """Exception raised for HTTP-related errors during async service invocation."""
