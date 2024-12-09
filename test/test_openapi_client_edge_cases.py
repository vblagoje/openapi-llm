# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from openapi_llm.client.config import ClientConfig
from openapi_llm.client.openapi import OpenAPIClient
from .conftest import FastAPITestClient, create_openapi_spec


class TestEdgeCases:

    def test_missing_operation_id(self, test_files_path):
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))
        client = OpenAPIClient(config)

        payload = {
            "type": "function",
            "function": {
                "arguments": '{"name": "John", "message": "Hola"}',
                "name": "missingOperationId",
            },
        }
        with pytest.raises(ValueError, match="No operation found with operationId"):
            client.invoke(payload)

    def test_missing_operation_id_in_operation(self, test_files_path):
        """
        Test that the tool definition is generated correctly when the operationId is missing in the specification.
        """
        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))

        tools = config.get_tool_definitions(),
        tool_def = tools[0][0]
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "missing_operation_id_get"

    def test_servers_order(self, test_files_path):
        """
        Test that servers defined in different locations in the specification are used correctly.
        """

        config = ClientConfig(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))

        op = config.openapi_spec.find_operation_by_id("servers_order_path")
        assert op.get_server() == "https://inpath.example.com"
        op = config.openapi_spec.find_operation_by_id("servers_order_operation")
        assert op.get_server() == "https://inoperation.example.com"
        op = config.openapi_spec.find_operation_by_id("missing_operation_id_get")
        assert op.get_server() == "http://localhost"

    def test_allowed_operations(self):
         """
         Although the tool definition is generated from the OpenAPI spec and 
         firecrawl's API has multiple operations, only the ones we specify in the 
         allowed_operations list are registered with LLMs via the tool definition.
         """
         
         spec="https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json"
         
         config = ClientConfig(
             openapi_spec=create_openapi_spec(spec),
             request_sender=FastAPITestClient(None),
             allowed_operations=["scrape"],
         )
         tools = config.get_tool_definitions()
         assert len(tools) == 1
         assert tools[0]["function"]["name"] == "scrape"

         # test two operations
         config = ClientConfig(
             openapi_spec=create_openapi_spec(spec),
             request_sender=FastAPITestClient(None),
             allowed_operations=["scrape", "crawlUrls"],
         )
         tools = config.get_tool_definitions()
         assert len(tools) == 2
         assert tools[0]["function"]["name"] == "scrape"
         assert tools[1]["function"]["name"] == "crawlUrls"

         # test non-existent operation
         config = ClientConfig(
             openapi_spec=create_openapi_spec(spec),
             request_sender=FastAPITestClient(None),
             allowed_operations=["scrape", "non-existent-operation"],
         )
         tools = config.get_tool_definitions()
         assert len(tools) == 1
         assert tools[0]["function"]["name"] == "scrape"

         # test all non-existent operations
         config = ClientConfig(
             openapi_spec=create_openapi_spec(spec),
             request_sender=FastAPITestClient(None),
             allowed_operations=["non-existent-operation", "non-existent-operation-2"],
         )
         tools = config.get_tool_definitions()
         assert len(tools) == 0    
    