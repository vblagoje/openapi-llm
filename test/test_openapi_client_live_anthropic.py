# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import anthropic
import pytest

from openapi_llm.client.openapi import OpenAPIClient
from openapi_llm.providers.anthropic import AnthropicProvider
from .conftest import create_openapi_spec


class TestClientLiveAnthropic:

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY", ""), reason="ANTHROPIC_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):
        service_api = OpenAPIClient.from_spec(
            openapi_spec=test_files_path / "yaml" / "serper.yml",
            credentials=os.getenv("SERPERDEV_API_KEY"),
            llm_provider=AnthropicProvider()
        )
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=service_api.tool_definitions,
            messages=[{"role": "user", "content": "Do a google search: Who was Nikola Tesla?"}],
        )
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY", ""), reason="ANTHROPIC_API_KEY not set or empty")
    @pytest.mark.integration
    @pytest.mark.unstable("This test hits rate limit on Github API.")
    def test_github(self, test_files_path):
        service_api = OpenAPIClient.from_spec(
            openapi_spec=test_files_path / "yaml" / "github_compare.yml",
            llm_provider=AnthropicProvider()
        )
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=service_api.tool_definitions,
            messages=[
                {
                    "role": "user",
                    "content": "Compare branches main and add_default_adapter_filters in repo"
                    " haystack and owner deepset-ai",
                }
            ],
        )
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)
