import pytest
from pathlib import Path
from openapi_llm.client.openapi import OpenAPIClient
from openapi_llm.core.spec import OpenAPISpecification, create_openapi_spec
from openapi_llm.client.config import create_client_config

def test_create_openapi_spec_from_file(tmp_path):
    # Create a temporary YAML file
    spec_file = tmp_path / "test_spec.yaml"
    spec_file.write_text("""
    openapi: 3.0.0
    info:
        title: Test API
        version: 1.0.0
    paths: {}
    """)
    
    spec = create_openapi_spec(spec_file)
    assert isinstance(spec, OpenAPISpecification)

def test_create_openapi_spec_from_string():
    spec_str = """
    openapi: 3.0.0
    info:
        title: Test API
        version: 1.0.0
    paths: {}
    """
    spec = create_openapi_spec(spec_str)
    assert isinstance(spec, OpenAPISpecification)

def test_create_client_config():
    spec_str = """
    openapi: 3.0.0
    info:
        title: Test API
        version: 1.0.0
    paths: {}
    """
    config = create_client_config(spec_str)
    assert config.openapi_spec is not None

def test_openapi_client_from_spec():
    spec_str = """
    openapi: 3.0.0
    info:
        title: Test API
        version: 1.0.0
    paths: {}
    """
    client = OpenAPIClient.from_spec(spec_str, credentials="test_key")
    assert isinstance(client, OpenAPIClient) 