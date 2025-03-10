# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import cohere
import pytest

from openapi_llm.client.openapi import OpenAPIClient
from openapi_llm.providers.cohere import CohereProvider
from .conftest import create_openapi_spec

# Copied from Cohere's documentation
preamble = """
## Task & Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of
 requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to
 help you, which you use to research your answer. You should focus on serving the user's needs as best you can,
 which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and
 spelling.
"""


class TestClientLiveCohere:

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("COHERE_API_KEY", ""), reason="COHERE_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):
        service_api = OpenAPIClient.from_spec(
            openapi_spec=test_files_path / "yaml" / "serper.yml",
            credentials=os.getenv("SERPERDEV_API_KEY"),
            llm_provider=CohereProvider()
        )
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        response = client.chat(
            model="command-r",
            preamble=preamble,
            tools=service_api.tool_definitions,
            message="Do a google search: Who was Nikola Tesla?",
        )
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("COHERE_API_KEY", ""), reason="COHERE_API_KEY not set or empty")
    @pytest.mark.integration
    @pytest.mark.unstable("This test hits rate limit on Github API.")
    def test_github(self, test_files_path):
        service_api = OpenAPIClient.from_spec(
            openapi_spec=test_files_path / "yaml" / "github_compare.yml",
            llm_provider=CohereProvider()
        )
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        response = client.chat(
            model="command-r",
            preamble=preamble,
            tools=service_api.tool_definitions,
            message="Compare branches main and add_default_adapter_filters in repo haystack and owner deepset-ai",
        )
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)
