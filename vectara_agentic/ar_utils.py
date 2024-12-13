import json
import requests
import re

from typing import Dict, Any

from llama_index.indices.managed.vectara import VectaraIndex, VectaraQueryEngine, VectaraAutoRetriever
from llama_index.core.vector_stores.types import VectorStoreInfo, VectorStoreQuerySpec
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle

from .types import LLMRole
from .utils import get_llm


filter_attribute_types = {
    "FILTER_ATTRIBUTE_TYPE__TEXT": "str",
    "FILTER_ATTRIBUTE_TYPE__BOOLEAN": "bool",
    "FILTER_ATTRIBUTE_TYPE__INTEGER": "int",
    "FILTER_ATTRIBUTE_TYPE__REAL": "float"
}

def get_filter_attributes_from_corpus(vectara_tool_factory):
    url = "https://api.vectara.io/v1/read-corpus"

    payload = json.dumps({
    "corpusId": [
        vectara_tool_factory.vectara_corpus_id
    ],
    "readBasicInfo": False,
    "readSize": False,
    "readApiKeys": False,
    "readCustomDimensions": False,
    "readFilterAttributes": True
    })

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'customer-id': vectara_tool_factory.vectara_customer_id,
    'x-api-key': vectara_tool_factory.vectara_api_key
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    return response['corpora'][0]['filterAttribute']

def get_filter_attribute_examples(vectara_tool_factory) -> Dict:
                url = "https://api.vectara.io/v1/list-documents"

                payload = json.dumps({
                "corpusId": vectara_tool_factory.vectara_corpus_id,
                "numResults": 100,
                "pageKey": "",
                "metadataFilter": ""
                })
                headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'customer-id': vectara_tool_factory.vectara_customer_id,
                'x-api-key': vectara_tool_factory.vectara_api_key
                }

                response = requests.request("POST", url, headers=headers, data=payload)
                response = response.json()

                examples = {}

                for document in response['document']:
                    for field in document['metadata']:
                        if field['name'] in examples:
                            examples[field['name']].add(field['value'])
                        else:
                            examples[field['name']] = set([field['value']])

                return examples

def process_filter_string(filter_string) -> str:
    # Check for any operators that are supported by different names
    filter_string = filter_string.replace(" nin ", " NOT IN ")
    filter_string = filter_string.replace(" is_empty ", " IS NULL ")

    # Replace list brackets with parentheses
    filter_string = re.sub(r"'\[(.*?)\]'", r"(\1)", filter_string)

    return filter_string

def _build_auto_retriever(
    query_string: str,
    index: VectaraIndex,
    vector_store_info: VectorStoreInfo,
    llm: LLM,
    **kwargs: Any,
) ->  VectaraQueryEngine:
    auto_retriever = VectaraAutoRetriever(
        index=index,
        vector_store_info = vector_store_info,
        llm=get_llm(LLMRole.TOOL),
        **kwargs
    )
    query_bundle = QueryBundle(query_str=query_string)
    spec = auto_retriever.generate_retrieval_spec(query_bundle)
    print(f"DEBUG: RECEIVED SPEC FROM GENERATOR {spec} of type {type(spec)}")
    vectara_retriever, new_query = auto_retriever._build_retriever_from_spec(
        VectorStoreQuerySpec(
            query=spec.query, filters=spec.filters
        )
    )

    return VectaraQueryEngine.from_args(retriever=vectara_retriever), new_query
