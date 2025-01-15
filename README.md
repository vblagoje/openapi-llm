# OpenAPI-LLM

[![PyPI](https://img.shields.io/pypi/v/openapi-llm)](https://pypi.org/project/openapi-llm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openapi-llm?color=blue&logo=pypi&logoColor=gold)](https://pypi.org/project/openapi-llm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/openapi-llm?logo=python&logoColor=gold)](https://pypi.org/project/openapi-llm/)
[![Tests](https://github.com/vblagoje/openapi-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/vblagoje/openapi-llm/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/vblagoje/openapi-llm/badge.svg)](https://coveralls.io/github/vblagoje/openapi-llm)
[![GitHub](https://img.shields.io/github/license/vblagoje/openapi-llm?color=blue)](LICENSE)

A Python library that converts OpenAPI specifications into Large Language Model (LLM) tool/function definitions, enabling OpenAPI invocations through LLM generated tool calls.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Supported Python Versions](#supported-python-versions)
  - [LLM Provider Dependencies](#llm-provider-dependencies)
- [Quick Start](#quick-start)
  - [Synchronous Example](#synchronous-example)
  - [Asynchronous Example](#asynchronous-example)
- [Customization: `from_spec`](#customization-from_spec)
- [Library Scope](#library-scope)
  - [OpenAPI Specification Validation](#openapi-specification-validation)
- [Requirements](#requirements)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [License](#license)
- [Security](#security)
- [Contributing](#contributing)

## Features

- Convert OpenAPI specifications into LLM-compatible tool/function definitions
- Support for multiple LLM providers (OpenAI, Anthropic, Cohere)
- Handle complex request bodies and parameter types
- Support for multiple authentication mechanisms
- Support for OpenAPI 3.0.x and 3.1.x specifications
- Handles both YAML and JSON OpenAPI specifications

## Installation

```bash
pip install openapi-llm
```

### Supported Python Versions
- Python >= 3.8

### LLM Provider Dependencies

By default, **OpenAPI-LLM** does not install any particular LLM provider. You can install exactly the ones you need:

```bash
pip install openai     # For OpenAI
pip install anthropic  # For Anthropic
pip install cohere     # For Cohere
```

## Quick Start

Below are minimal working examples for synchronous and asynchronous usage.

### Synchronous Example

```python
import os
from openai import OpenAI
from openapi_llm.client.openapi import OpenAPIClient

# Create the client from a spec URL (or file path, or raw string)
service_api = OpenAPIClient.from_spec(
    openapi_spec="https://bit.ly/serperdev_openapi",
    credentials=os.getenv("SERPERDEV_API_KEY")
)

# Initialize your chosen LLM provider (e.g., OpenAI)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ask the LLM to call the SerperDev API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
    tools=service_api.tool_definitions,  # LLM tool definitions from the client
)

# Now actually invoke the OpenAPI call based on the LLM's generated tool call
service_response = service_api.invoke(response)
assert "inventions" in str(service_response)
```

### Asynchronous Example

```python
import os
import asyncio
from openapi_llm.client.openapi_async import AsyncOpenAPIClient
from openai import AsyncOpenAI

async def main():
    # Firecrawl openapi spec
    openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/v1-openapi.json"

    # Create the async client
    service_api = AsyncOpenAPIClient.from_spec(
        openapi_spec=openapi_spec_url,
        credentials=os.getenv("FIRECRAWL_API_KEY")
    )

    # Initialize an async LLM (OpenAI)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Ask the LLM to call Firecrawl's scraping endpoint
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Scrape URL: https://news.ycombinator.com/"}],
        tools=service_api.tool_definitions,
    )

    # Use context manager to manage aiohttp sessions
    async with service_api as api:
        service_response = await api.invoke(response)
        assert isinstance(service_response, dict)
        assert service_response.get("success", False), "Firecrawl scrape API call failed"

asyncio.run(main())
```

## Customization: `from_spec`

Both **`OpenAPIClient`** and **`AsyncOpenAPIClient`** provide a classmethod called `from_spec`, which automatically:

- Loads the OpenAPI specification from a file path, URL, or raw string.
- Builds a `ClientConfig` for you.
- Constructs the client instance.

For example, you can validate the spec before using it by supplying a custom `config_factory`:

```python
from openapi_llm.client.config import ClientConfig, create_client_config
from openapi_llm.client.openapi import OpenAPIClient
from openapi_spec_validator import validate_spec

def my_custom_config_factory(openapi_spec: str, **kwargs) -> ClientConfig:
    config = create_client_config(openapi_spec, **kwargs)
    validate_spec(config.openapi_spec.spec_dict)
    return config

# Usage:
client = OpenAPIClient.from_spec(
    openapi_spec="path/to/local_spec.yaml",
    config_factory=my_custom_config_factory,
    credentials="secret_token"
)
```

This design gives you **full control** over the spec-loading and configuration-building process while still offering simple defaults.


## Library Scope

OpenAPI-LLM focuses on the **core** of bridging LLM function calls with OpenAPI specifications. It does **not** perform advanced validation or impose a high-level framework. You can integrate it into your existing app or build additional logic on top.

### OpenAPI Specification Validation

This library does **not** automatically validate your specs. If your OpenAPI file is invalid, you might see errors during usage. Tools like [openapi-spec-validator](https://github.com/p1c2u/openapi-spec-validator) or [prance](https://github.com/RonnyPfannschmidt/prance) can help ensure correctness before you load your spec here.

## Requirements

- Python >= 3.8
- Dependencies:
  - jsonref
  - requests
  - PyYAML

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vblagoje/openapi-llm.git
   ```
2. **Install Hatch** (if you haven’t already):
   ```bash
   pip install hatch
   ```
3. **Install Pre-Commit Hooks**:
   ```bash
   pip install pre-commit
   ```
4. **Install Desired LLM Provider Dependencies** (e.g., openai, anthropic, cohere):
   ```bash
   pip install openai anthropic cohere
   ```

## Testing

Run tests using [hatch](https://github.com/pypa/hatch):

```bash
# Unit tests
hatch run test:unit

# Integration tests
hatch run test:integration

# Type checking
hatch run test:typing

# Linting
hatch run test:lint
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Security

For security concerns, please see our [Security Policy](SECURITY.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Vladimir Blagojevic (dovlex@gmail.com)

Early reviews and guidance by Madeesh Kannan
