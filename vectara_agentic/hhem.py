"""Vectara HHEM (Hypothesis Hypothetical Evaluation Model) client."""

import requests
from commonmark import Parser


def markdown_to_text(md: str) -> str:
    """
    Convert a Markdown-formatted string into plain text.
    """
    parser = Parser()
    ast = parser.parse(md)
    out: list[str] = []

    def recurse(node):
        if node.t in ("text", "code", "html_inline"):
            out.append(node.literal or "")
        elif node.t == "softbreak":
            out.append(" ")
        elif node.t == "linebreak":
            out.append("\n")
        child = getattr(node, "first_child", None)
        while child is not None:
            recurse(child)
            child = getattr(child, "next", None)

    recurse(ast)
    text = "".join(out)
    # collapse runs of spaces but preserve newlines
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line if line.strip() else "" for line in lines)


class HHEM:
    """Vectara HHEM (Hypothesis Hypothetical Evaluation Model) client."""

    def __init__(self, vectara_api_key: str):
        self._vectara_api_key = vectara_api_key

    def compute(self, context: str, hypothesis: str) -> float:
        """
        Calls the Vectara HHEM endpoint to evaluate the factual consistency of a hypothesis against a given context.

        Parameters:
            context (str): The source text against which the hypothesis will be evaluated.
            hypothesis (str): The generated text to be evaluated for factual consistency.

        Returns:
            float: The factual consistency score rounded to four decimal places.

        Raises:
            requests.exceptions.RequestException: If there is a network-related error or the API call fails.
        """

        # clean response from any markdown or other formatting.
        try:
            clean_hypothesis = markdown_to_text(hypothesis)
        except Exception as e:
            # If markdown parsing fails, use the original text
            raise ValueError(f"Markdown parsing of hypothesis failed: {e}") from e

        # compute HHEM with Vectara endpoint
        payload = {
            "model_parameters": {"model_name": "hhem_v2.3"},
            "generated_text": clean_hypothesis,
            "source_texts": [context],
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._vectara_api_key,
        }

        response = requests.post(
            "https://api.vectara.io/v2/evaluate_factual_consistency",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return round(data.get("score", 0.0), 4)
