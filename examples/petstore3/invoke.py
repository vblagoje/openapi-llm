from openapi_llm.client.openapi import OpenAPIClient

# Create the client from a spec URL (or file path, or raw string)
service_api = OpenAPIClient.from_spec(
    openapi_spec="https://petstore3.swagger.io/api/v3/openapi.json",
    url="https://petstore3.swagger.io/api/v3"
)
response = {
  "arguments": {
      "petId": 1
      },
  "name": "getPetById"
}
service_response = service_api.invoke(response)
print(service_response)
