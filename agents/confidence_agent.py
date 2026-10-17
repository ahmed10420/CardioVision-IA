import math
import os

from dotenv import load_dotenv
from openai import OpenAI


class ConfidenceAgent:

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("NVIDIA_API_KEY")
        model = os.getenv(
            "NVIDIA_MODEL",
            "openai/gpt-oss-20b",
        )

        if not api_key:
            raise ValueError("NVIDIA_API_KEY manquante.")

        self.model = model

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=90.0,
            max_retries=0,
        )

    def run(self, question, answer):

        prompt = f"""
Tu dois reproduire EXACTEMENT la réponse suivante.

RÈGLES :
- Ne change aucun mot.
- N'ajoute aucun mot.
- Ne supprime aucun mot.
- Ne reformule rien.
- Conserve exactement la ponctuation et les retours à la ligne.

RÉPONSE À REPRODUIRE :

{answer}
"""

        # NVIDIA est une métrique optionnelle : une seule tentative courte
        # suffit et son indisponibilité ne doit pas bloquer le pipeline clinique.
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=2500,
                logprobs=True,
                top_logprobs=5,
            )
        except Exception as exc:
            return {
                "agent": "ConfidenceAgent",
                "status": "error",
                "confidence": None,
                "entropy": None,
                "tokens_analyzed": 0,
                "exact_match": False,
                "error": str(exc),
            }

        choice = response.choices[0]

        generated_answer = (
            choice.message.content or ""
        ).strip()

        original_answer = str(answer or "").strip()

        # Le modèle doit reproduire exactement final_answer.
        exact_match = generated_answer == original_answer

        token_logprobs = []
        token_entropies = []

        if (
            choice.logprobs
            and choice.logprobs.content
        ):
            for item in choice.logprobs.content:
                if item.logprob is not None:
                    token_logprobs.append(
                        float(item.logprob)
                    )

                # Entropie calculée à partir des alternatives disponibles.
                top_logprobs = getattr(
                    item,
                    "top_logprobs",
                    None,
                )

                if top_logprobs:
                    probs = []

                    for alt in top_logprobs:
                        if alt.logprob is not None:
                            probs.append(
                                math.exp(
                                    float(alt.logprob)
                                )
                            )

                    total = sum(probs)

                    if total > 0:
                        normalized_probs = [
                            probability / total
                            for probability in probs
                        ]

                        entropy = -sum(
                            probability * math.log(probability + 1e-12)
                            for probability in normalized_probs
                        )

                        token_entropies.append(entropy)

        if not token_logprobs:
            return {
                "agent": "ConfidenceAgent",
                "status": "no_logprobs",
                "confidence": None,
                "entropy": None,
                "tokens_analyzed": 0,
                "exact_match": exact_match,
            }

        probabilities = [
            math.exp(lp)
            for lp in token_logprobs
        ]

        confidence = (
            sum(probabilities)
            / len(probabilities)
        )

        mean_entropy = (
            sum(token_entropies)
            / len(token_entropies)
            if token_entropies
            else None
        )

        return {
            "agent": "ConfidenceAgent",
            "status": (
                "success"
                if exact_match
                else "generated_text_mismatch"
            ),

            # confiance du modèle sur sa génération,
            # pas vérité médicale
            "confidence": round(
                confidence,
                4,
            ),

            "entropy": (
                round(mean_entropy, 4)
                if mean_entropy is not None
                else None
            ),

            "tokens_analyzed": len(
                token_logprobs
            ),

            "exact_match": exact_match,

            "generated_answer": (
                None
                if exact_match
                else generated_answer
            ),
        }
