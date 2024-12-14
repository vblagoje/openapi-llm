# OpenAPI-LLM

[![PyPI](https://img.shields.io/pypi/v/openapi-llm)](https://pypi.org/project/openapi-llm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openapi-llm?color=blue&logo=pypi&logoColor=gold)](https://pypi.org/project/openapi-llm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/openapi-llm?logo=python&logoColor=gold)](https://pypi.org/project/openapi-llm/)
[![Tests](https://github.com/vblagoje/openapi-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/vblagoje/openapi-llm/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/vblagoje/openapi-llm/badge.svg)](https://coveralls.io/github/vblagoje/openapi-llm)
[![GitHub](https://img.shields.io/github/license/vblagoje/openapi-llm?color=blue)](LICENSE)

A Python library that converts OpenAPI specifications into Large Language Model (LLM) tool/function definitions, enabling OpenAPI invocations through LLM generated tool calls.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Library Scope](#library-scope)
- [Quick Start](#quick-start)
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

This library focuses on OpenAPI-to-LLM conversion and doesn't include LLM provider libraries by default. Install the ones you need:

```bash
# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For Cohere
pip install cohere
```

## Library Scope

OpenAPI-LLM provides core functionality for converting OpenAPI specifications into LLM-compatible tool/function definitions. It intentionally does not provide an opinionated, high-level interface for OpenAPI-LLM interactions. Users are encouraged to develop their own thin application layer above this library that suits their specific needs and preferences for OpenAPI-LLM integration.

### OpenAPI Specification Validation

This library does not perform OpenAPI specification validation. It is the user's responsibility to ensure that the provided OpenAPI specifications are valid. We recommend using established validation tools such as:

- [openapi-spec-validator](https://github.com/p1c2u/openapi-spec-validator)
- [prance](https://github.com/RonnyPfannschmidt/prance)
- [Swagger Editor](https://editor.swagger.io/)

Example of validating a spec before using it with openapi-llm:

```python
from openapi_spec_validator import validate_spec
import yaml

# Load and validate your OpenAPI spec
with open('your_spec.yaml', 'r') as f:
    spec_dict = yaml.safe_load(f)
validate_spec(spec_dict)
```

## Quick Start

Here's a practical example using OpenAI to perform a Google search via SerperDev API:

```python
import os
from openai import OpenAI

from openapi_llm.client.config import ClientConfig
from openapi_llm.client.openapi import OpenAPIClient
from openapi_llm.core.spec import OpenAPISpecification


# Configure the OpenAPI client with SerperDev API spec and credentials
config = ClientConfig(
    openapi_spec=OpenAPISpecification.from_url("https://bit.ly/serperdev_openapi"), 
    credentials=os.getenv("SERPERDEV_API_KEY")
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a chat completion with tool definitions
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
    tools=config.get_tool_definitions(),
)

# Execute the API call based on the LLM's response
service_api = OpenAPIClient(config)
service_response = service_api.invoke(response)
```

This example demonstrates:
- Loading an OpenAPI specification from a URL
- Integrating with OpenAI's function calling
- Handling API authentication
- Converting and executing OpenAPI calls based on LLM responses

## Requirements

- Python >= 3.8
- Dependencies:
  - jsonref
  - requests
  - PyYAML

## Development Setup

1. Clone the repository

```bash
git clone https://github.com/vblagoje/openapi-llm.git
```

2. Install hatch if you haven't already

```bash
pip install hatch
```

3. Install pre-commit hooks

```bash
pre-commit install
```

4. Install desired LLM provider dependencies (as needed)

```bash
pip install openai anthropic cohere
```

## Testing

Run tests using hatch:

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

MIT License - See [LICENSE](LICENSE) for details.

## Security

For security concerns, please see our [Security Policy](SECURITY.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Vladimir Blagojevic (dovlex@gmail.com)

Reviews and guidance by Madeesh Kannan
