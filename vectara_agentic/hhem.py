"""Vectara HHEM (Hypothesis Hypothetical Evaluation Model) client."""
import requests

class HHEM():
    """Vectara HHEM (Hypothesis Hypothetical Evaluation Model) client."""

    def __init__(self, vectara_api_key):
        self._vectara_api_key = vectara_api_key

    def compute(self, context, hypothesis):
        """Calls the Vectara HHEM endpoint."""
        ### TEMP
        try:
            with open("/Users/ofer/temp/hhem_output.txt", "w") as f:
                f.write(f"DEBUG response: \n{hypothesis}\n\n")
                f.write(f"DEBUG context: \n{context}")
        except Exception as e:
            print(f"DEBUG Failed to write debug output: {e}")
        ### TEMP

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
