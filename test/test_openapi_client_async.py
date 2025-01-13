import asyncio
import pytest
from openapi_llm.client.openapi_async import AsyncOpenAPIClient, AsyncOpenAPIClientError
from openapi_llm.client.config import ClientConfig
from .conftest import create_openapi_spec


@pytest.mark.asyncio
class TestAsyncOpenAPIClient:
    @pytest.fixture(autouse=True)
    async def cleanup(self):
        yield
        await asyncio.sleep(0.1)

    async def test_invoke_invalid_payload(self, test_files_path):
        """Test error case in async payload extraction"""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="dummy_key"
        )
        client = AsyncOpenAPIClient(config)
        with pytest.raises(AsyncOpenAPIClientError):
            await client.invoke({"invalid": "payload"})

    async def test_invoke_missing_required_keys(self, test_files_path):
        """Test error in async _get_operation"""
        config = ClientConfig(
            openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
            credentials="dummy_key"
        )
        client = AsyncOpenAPIClient(config)
        with pytest.raises(AsyncOpenAPIClientError):
            await client.invoke({"some": "data"})