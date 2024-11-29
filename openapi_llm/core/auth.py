from typing import Any, Callable, Dict


def create_api_key_authenticator(
    api_key: str,
) -> Callable[[Dict[str, Any], Dict[str, Any]], None]:
    """
    Creates a function that applies API key authentication to requests.

    Generates an authenticator function that can inject an API key into different parts
    of a request (header, query, or cookie) based on the security scheme.

    :param api_key: The API key to use for authentication.
    :returns: A function that applies API key authentication according to the security scheme.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]) -> None:
        """
        Apply the API key authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["in"] == "header":
            request.setdefault("headers", {})[security_scheme["name"]] = api_key
        elif security_scheme["in"] == "query":
            request.setdefault("params", {})[security_scheme["name"]] = api_key
        elif security_scheme["in"] == "cookie":
            request.setdefault("cookies", {})[security_scheme["name"]] = api_key
        else:
            raise ValueError(
                f"Unsupported apiKey authentication location: {security_scheme['in']}, "
                f"must be one of 'header', 'query', or 'cookie'"
            )

    return apply_auth


def create_bearer_token_authenticator(
    token: str,
) -> Callable[[Dict[str, Any], Dict[str, Any]], None]:
    """
    Creates a function that applies Bearer token authentication to requests.

    Generates an authenticator function that can inject a Bearer token into request headers
    according to the HTTP authentication scheme.

    :param token: The authentication token to use.
    :returns: A function that applies Bearer token authentication according to the security scheme.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]) -> None:
        """
        Apply the HTTP authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["type"] == "http":
            # support bearer http auth, no basic support yet
            if security_scheme["scheme"].lower() == "bearer":
                if not token:
                    raise ValueError("Token must be provided for Bearer Auth.")
                request.setdefault("headers", {})["Authorization"] = f"Bearer {token}"
            else:
                raise ValueError(
                    f"Unsupported HTTP authentication scheme: {security_scheme['scheme']}"
                )
        else:
            raise ValueError(
                "HTTPAuthentication strategy received a non-HTTP security scheme."
            )

    return apply_auth
