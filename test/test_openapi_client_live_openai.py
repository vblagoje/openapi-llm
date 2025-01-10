# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from openai import OpenAI

from openapi_llm.client.config import ClientConfig
from openapi_llm.client.openapi import OpenAPIClient
from .conftest import create_openapi_spec


class TestClientLiveOpenAPI:

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):

        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        service_api = OpenAPIClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

        # make a few more requests to test the same tool
        service_response = service_api.invoke(response)
        assert "Serbian" in str(service_response)

        service_response = service_api.invoke(response)
        assert "American" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev_json_spec(self, test_files_path):

        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "json" / "serperdev_openapi_spec.json"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        service_api = OpenAPIClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    def test_serperdev_json_spec_url(self):

        config = ClientConfig(openapi_spec=create_openapi_spec("https://bit.ly/serperdev_openapi"),
                              credentials=os.getenv("SERPERDEV_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        service_api = OpenAPIClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    @pytest.mark.unstable("This test hits rate limit on Github API.")
    def test_github(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "github_compare.yml"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Compare branches main and add_default_adapter_filters in repo"
                    " haystack and owner deepset-ai",
                }
            ],
            tools=config.get_tool_definitions(),
        )
        service_api = OpenAPIClient(config)
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("FIRECRAWL_API_KEY", ""), reason="FIRECRAWL_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    def test_firecrawl(self):
        """
        Test Firecrawl API integration with both scraping and search endpoints.

        Test passes if either the API call is successful or returns a payment required error (402).
        """
        from openapi_llm.utils import HttpClientError

        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/v1-openapi.json"
        config = ClientConfig(openapi_spec=create_openapi_spec(openapi_spec_url), credentials=os.getenv("FIRECRAWL_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Scrape URL: https://news.ycombinator.com/"}],
            tools=config.get_tool_definitions(),
        )
        service_api = OpenAPIClient(config)

        try:
            service_response = service_api.invoke(response)
            assert isinstance(service_response, dict)
            assert service_response.get("success", False), "Firecrawl scrape API call failed"

            # Only proceed with search test if scrape was successful
            top_k = 2
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Search Google for `Why was Sam Altman ousted from OpenAI?`, limit to {top_k} results",
                    }
                ],
                tools=config.get_tool_definitions(),
            )
            service_response = service_api.invoke(response)
            assert isinstance(service_response, dict)
            assert service_response.get("success", False), "Firecrawl search API call failed"
            assert len(service_response.get("data", [])) == top_k
            assert "Sam" in str(service_response)

        except HttpClientError as e:
            # Accept 402 Payment Required as a valid test outcome
            assert "402" in str(e) or "Payment Required" in str(e), \
                f"Unexpected HTTP error: {str(e)}"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("TOMTOM_API_KEY", ""), reason="TOMTOM_API_KEY not set or empty")
    def test_tomtom(self, test_files_path):

        # LLM can't accept the original operation name with {} and other special characters, 
        # so we need to normalize it, see normalize_operation_name in utils.py
        target_operation = "search_versionNumber_categorySearch_query_ext__get"
        spec="https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/tomtom.com/search/1.0.0/openapi.yaml"
        config = ClientConfig(openapi_spec=create_openapi_spec(spec),
                              credentials=os.getenv("TOMTOM_API_KEY"),
                              allowed_operations=["search_versionNumber_categorySearch_query_ext__get"])

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for pizza in San Francisco, US and don't use long/lat"}],
            tools=config.get_tool_definitions(),
        )
        api = OpenAPIClient(config)
        service_response = api.invoke(response)

        # verify that we get some valid response
        assert isinstance(service_response, dict)
        assert "pizza" in str(service_response)
        assert "summary" in str(service_response)
        assert "countrySubdivisionCode" in str(service_response)
