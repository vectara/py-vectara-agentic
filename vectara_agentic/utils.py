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


class remove_self_from_signature:  # pylint: disable=invalid-name
    """Descriptor that hides 'self' on the class attribute, but leaves bound methods alone.

    This solves the issue where modifying __signature__ on methods causes problems
    with Python's bound method creation. Instead, we use a descriptor that:
    - Returns a wrapper with 'self' removed when accessed on the class (for tool creation)
    - Returns a normal bound method when accessed on instances (for normal method calls)
    """
    def __init__(self, func):
        import functools
        functools.update_wrapper(self, func)
        self.func = func
        sig = signature(func)
        params = list(sig.parameters.values())
        # Remove the first parameter if it is named 'self'
        if params and params[0].name == "self":
            params = params[1:]
        self._unbound_sig = sig.replace(parameters=params)

    def __get__(self, obj, objtype=None):
        import functools
        import types
        if obj is None:
            # Accessed on the class: provide a function-like object with 'self' removed.
            @functools.wraps(self.func)
            def wrapper(*args, **kwargs):
                return self.func(*args, **kwargs)
            wrapper.__signature__ = self._unbound_sig
            return wrapper
        # Accessed on an instance: return the original bound method so inspect removes 'self' exactly once.
        return types.MethodType(self.func, obj)

    # Allow direct calls via the descriptor if someone invokes it off the class attribute.
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


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
