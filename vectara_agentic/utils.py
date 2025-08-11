"""
Utilities for the Vectara agentic.
"""

from inspect import signature
import json
import asyncio
import aiohttp


def is_float(value: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def remove_self_from_signature(func):
    """Decorator to remove 'self' from a method's signature for introspection."""
    sig = signature(func)
    params = list(sig.parameters.values())
    # Remove the first parameter if it is named 'self'
    if params and params[0].name == "self":
        params = params[1:]
    new_sig = sig.replace(parameters=params)
    func.__signature__ = new_sig
    return func


async def summarize_vectara_document(
    llm_name: str, corpus_key: str, api_key: str, doc_id: str
) -> str:
    """
    Summarize a document in a Vectara corpus using the Vectara API.
    """
    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents/{doc_id}/summarize"

    payload = json.dumps(
        {
            "llm_name": llm_name,
            "model_parameters": {"temperature": 0.0},
            "stream_response": False,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key,
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, data=payload) as response:
            if response.status != 200:
                error_json = await response.json()
                return (
                    f"Vectara Summarization failed with error code {response.status}, "
                    f"error={error_json['messages'][0]}"
                )
            data = await response.json()
            return data["summary"]
    return json.loads(response.text)["summary"]


async def summarize_documents(
    corpus_key: str,
    api_key: str,
    doc_ids: list[str],
    llm_name: str = "gpt-4o",
) -> dict[str, str | BaseException]:
    """
    Summarize multiple documents in a Vectara corpus using the Vectara API.
    """
    if not doc_ids:
        return {}
    if llm_name is None:
        llm_name = "gpt-4o"
    tasks = [
        summarize_vectara_document(
            corpus_key=corpus_key, api_key=api_key, llm_name=llm_name, doc_id=doc_id
        )
        for doc_id in doc_ids
    ]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(doc_ids, summaries))
