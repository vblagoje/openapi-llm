# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import yaml
from openapi_llm.client.config import ClientConfig
from openapi_llm.client.openapi import OpenAPIClient
from .conftest import create_openapi_spec


class TestClientLive:

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"), credentials=os.getenv("SERPERDEV_API_KEY"))
        serper_api = OpenAPIClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"q": "Who was Nikola Tesla?"}',
                "name": "serperdev_search",
            },
            "type": "function",
        }
        response = serper_api.invoke(payload)
        assert "invention" in str(response)

    @pytest.mark.integration
    @pytest.mark.unstable("This test hits rate limit on Github API.")
    def test_github(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "github_compare.yml"))
        api = OpenAPIClient(config)

        params = {"owner": "deepset-ai", "repo": "haystack", "basehead": "main...add_default_adapter_filters"}
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": json.dumps(params),
                "name": "compare_branches",
            },
            "type": "function",
        }
        response = api.invoke(payload)
        assert "deepset" in str(response)
