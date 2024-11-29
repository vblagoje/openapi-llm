from typing import Any

from openapi_llm.client.config import ClientConfig
from openapi_llm.utils import apply_authentication, build_request


class OpenAPIClient:
    """
    A client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfig):
        self.client_config = client_config

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


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""


class HttpClientError(Exception):
    """Exception raised for HTTP-related errors during service invocation."""
