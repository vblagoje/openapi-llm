# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pytest
from openai import AsyncOpenAI
import aiohttp

from openapi_llm.client.config import ClientConfig
from openapi_llm.client.openapi_async import AsyncOpenAPIClient, AsyncOpenAPIClientError, AsyncHttpClientError
from .conftest import create_openapi_spec


@pytest.mark.asyncio
class TestClientLiveOpenAPIAsync:

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        yield
        # Force cleanup any remaining sessions
        await asyncio.sleep(0.1)  # Allow pending callbacks to complete

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_serperdev(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        async with AsyncOpenAPIClient(config) as service_api:
            service_response = await service_api.invoke(response)
            assert "inventions" in str(service_response)

            # make a few more requests to test the same tool
            service_response = await service_api.invoke(response)
            assert "Serbian" in str(service_response)

            service_response = await service_api.invoke(response)
            assert "American" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_serperdev_json_spec(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "json" / "serperdev_openapi_spec.json"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        async with AsyncOpenAPIClient(config) as service_api:
            service_response = await service_api.invoke(response)
            assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_serperdev_json_spec_url(self):
        config = ClientConfig(openapi_spec=create_openapi_spec("https://bit.ly/serperdev_openapi"),
                              credentials=os.getenv("SERPERDEV_API_KEY"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )
        async with AsyncOpenAPIClient(config) as service_api:
            service_response = await service_api.invoke(response)
            assert "inventions" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    @pytest.mark.unstable("This test hits rate limit on Github API.")
    async def test_github(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "github_compare.yml"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
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
        async with AsyncOpenAPIClient(config) as service_api:
            service_response = await service_api.invoke(response)
            assert "deepset" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("FIRECRAWL_API_KEY", ""), reason="FIRECRAWL_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_firecrawl(self):
        """
        Test Firecrawl API integration with both scraping and search endpoints.

        Test passes if either the API call is successful or returns a payment required error (402).
        """
        from openapi_llm.utils import HttpClientError

        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/v1-openapi.json"
        config = ClientConfig(openapi_spec=create_openapi_spec(openapi_spec_url), credentials=os.getenv("FIRECRAWL_API_KEY"), allowed_operations=["scrapeAndExtractFromUrl"])
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Scrape URL: https://news.ycombinator.com/"}],
            tools=config.get_tool_definitions(),
        )
        async with AsyncOpenAPIClient(config) as service_api:            
            service_response = await service_api.invoke(response)
            assert isinstance(service_response, dict)
            assert service_response.get("success", False), "Firecrawl scrape API call failed"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("TOMTOM_API_KEY", ""), reason="TOMTOM_API_KEY not set or empty")
    async def test_tomtom(self, test_files_path):
        # LLM can't accept the original operation name with {} and other special characters, 
        # so we need to normalize it, see normalize_operation_name in utils.py
        target_operation = "search_versionNumber_categorySearch_query_ext__get"
        spec="https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/tomtom.com/search/1.0.0/openapi.yaml"
        config = ClientConfig(openapi_spec=create_openapi_spec(spec),
                              credentials=os.getenv("TOMTOM_API_KEY"),
                              allowed_operations=["search_versionNumber_categorySearch_query_ext__get"])

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for pizza in San Francisco, US and don't use long/lat"}],
            tools=config.get_tool_definitions(),
        )
        async with AsyncOpenAPIClient(config) as service_api:
            service_response = await service_api.invoke(response)

            # verify that we get some valid response
            assert isinstance(service_response, dict)
            assert "pizza" in str(service_response)
            assert "summary" in str(service_response)
            assert "countrySubdivisionCode" in str(service_response)

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_serperdev_without_context_manager(self, test_files_path):
        """Test the async client without using context management."""
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )

        # Create client and set it up
        service_api = AsyncOpenAPIClient(config)
        await service_api.setup()

        try:
            service_response = await service_api.invoke(response)
            assert "inventions" in str(service_response)

            # make a few more requests to test the same tool
            service_response = await service_api.invoke(response)
            assert "Serbian" in str(service_response)

            service_response = await service_api.invoke(response)
            assert "American" in str(service_response)
        finally:
            await service_api.cleanup()

    @pytest.mark.skipif(not os.environ.get("SERPERDEV_API_KEY", ""), reason="SERPERDEV_API_KEY not set or empty")
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", ""), reason="OPENAI_API_KEY not set or empty")
    @pytest.mark.integration
    async def test_serperdev_with_shared_session(self, test_files_path):
        """Test the async client with a shared aiohttp session."""
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tool_definitions(),
        )

        # Create a shared session
        async with aiohttp.ClientSession() as session:
            # Create client and set it up with the shared session
            service_api = AsyncOpenAPIClient(config)
            await service_api.setup(session=session)

            service_response = await service_api.invoke(response)
            assert "inventions" in str(service_response)

            # make a few more requests to test the same tool
            service_response = await service_api.invoke(response)
            assert "Serbian" in str(service_response)

            service_response = await service_api.invoke(response)
            assert "American" in str(service_response)

            # No need to call cleanup() since we didn't create the session 

    @pytest.mark.asyncio
    async def test_missing_required_payload_fields(self, test_files_path):
        """Test handling of payloads missing required fields."""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="dummy_key"
        )
        
        incomplete_payload = {
            "function": {
                "arguments": {"query": "test"}
                # Missing 'name' field
            }
        }
        
        async with AsyncOpenAPIClient(config) as client:
            with pytest.raises(AsyncOpenAPIClientError) as exc_info:
                await client.invoke(incomplete_payload)
            assert "does not contain 'name' or 'arguments' keys" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_session_cleanup(self, test_files_path):
        """Test cleanup behavior with invalid session state."""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="dummy_key"
        )
        
        client = AsyncOpenAPIClient(config)
        # Should not raise any errors even though session doesn't exist
        await client.cleanup()
        
        # Test double cleanup
        await client.setup()
        await client.cleanup()
        await client.cleanup()  # Should not raise error

    @pytest.mark.asyncio
    async def test_session_reuse(self, test_files_path):
        """Test proper session reuse and cleanup."""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="dummy_key"
        )
        
        # Create a shared session
        async with aiohttp.ClientSession() as session:
            client = AsyncOpenAPIClient(config)
            
            # Setup with shared session
            await client.setup(session=session)
            assert client._session == session
            assert not client._owns_session
            
            # Cleanup shouldn't close the shared session
            await client.cleanup()
            assert not session.closed
            
            # Setup again with the same session
            await client.setup(session=session)
            assert client._session == session
            assert not client._owns_session

    @pytest.mark.asyncio
    async def test_http_error_handling(self, test_files_path):
        """Test handling of HTTP errors."""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="invalid_key"  # This should cause authentication errors
        )
        
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Do a serperdev google search: test query"}],
            tools=config.get_tool_definitions(),
        )
        
        async with AsyncOpenAPIClient(config) as service_api:
            with pytest.raises(AsyncHttpClientError) as exc_info:
                await service_api.invoke(response)
            assert "HTTP error occurred" in str(exc_info.value) 