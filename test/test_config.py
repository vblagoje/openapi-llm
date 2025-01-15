import pytest
from pathlib import Path

from openapi_llm.client.config import create_client_config, ClientConfig
from openapi_llm.core.spec import OpenAPISpecification


class TestConfig:
    def test_create_client_config_invalid_format(self):
        """Test error case when invalid input type is provided"""
        with pytest.raises(ValueError):
            create_client_config(123)  # Invalid input type

    def test_create_client_config_from_url(self):
        """Test creating config from URL"""
        spec_url = "https://raw.githubusercontent.com/example/api/main/openapi.json"
        with pytest.raises(ValueError):
            create_client_config(spec_url)

    def test_create_client_config_from_str(self, test_files_path):
        """Test creating config from raw string"""
        raw_spec = (test_files_path / "yaml" / "serper.yml").read_text()
        config = create_client_config(raw_spec)
        assert isinstance(config, ClientConfig)
        assert isinstance(config.openapi_spec, OpenAPISpecification)

    def test_unsupported_auth_type(self):
        """Test error handling for unsupported authentication type"""
        spec_dict = {
            "openapi": "3.0.0",
            "components": {
                "securitySchemes": {
                    "custom": {
                        "type": "unsupported"
                    }
                }
            }
        }
        with pytest.raises(ValueError):
            spec = OpenAPISpecification(spec_dict)