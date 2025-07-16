"""Vectara HHEM (Hypothesis Hypothetical Evaluation Model) client."""

import requests


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
        payload = {
            "model_parameters": {"model_name": "hhem_v2.3"},
            "generated_text": hypothesis,
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
