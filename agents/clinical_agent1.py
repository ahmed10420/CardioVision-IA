import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.rag_with_llm import (
    MEDICAL_WARNING,
    build_prompt,
)



def _compact_text(value, max_chars):
    """Réduit défensivement un bloc de contexte avant l'appel LLM."""
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[contexte tronqué]..."


class ClinicalAgent:

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        if not api_key:
            raise ValueError("GROQ_API_KEY manquante.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            reasoning_effort="low",
            max_tokens=1200,
            api_key=api_key,
        )

    def run(
        self,
        question,
        context,
        critic_feedback=None,
        signal_result=None,
        ecg_reference_context=None,
        research_context=None,
        ptbxl_context=None,
        ptbxl_case_ids=None,
        web_context=None,
        web_source=None,
        previous_answer=None,
    ):
        """
        context:
            Contexte normatif provenant des guidelines ESC/AHA.

        signal_result:
            Résultat structuré du SignalAgent.
            Il n'est utilisé que si status == "success".

        ecg_reference_context:
            Références techniques et pédagogiques ECG.

        research_context:
            Articles scientifiques locaux complémentaires.

        ptbxl_context:
            Cas PTB-XL similaires. Ils servent uniquement de cas comparatifs
            et ne remplacent jamais les recommandations ESC/AHA.

        ptbxl_case_ids:
            Liste structurée des identifiants ECG PTB-XL retrouvés.

        web_context:
            Littérature externe récupérée par WebSearchAgent.
            Source complémentaire uniquement.

        web_source:
            Fournisseur effectivement retenu (PubMed, arXiv, DOAJ ou DuckDuckGo).
        """

        # Prompt principal fondé sur les guidelines.
        prompt = build_prompt(question, context)
        prompt += """

RÈGLES CLINIQUES COMPLÉMENTAIRES :
- Les faits explicitement présents dans la QUESTION sont des données d'entrée
  autorisées et peuvent être repris tels quels.
- Toute interprétation médicale (diagnostic, critère, seuil, urgence, traitement,
  pronostic ou recommandation) doit être explicitement soutenue par le contexte RAG.
- Les connaissances internes du LLM ne sont jamais une source valide.
- Une valeur fournie par l'utilisateur (QRS, QTc, FC, TA, amplitude ST, etc.)
  peut être reprise ; un seuil normatif ne peut être énoncé que s'il apparaît
  explicitement dans un passage documentaire avec ses conditions.
- Si une interprétation n'est pas soutenue, indique que le contexte documentaire
  fourni ne permet pas de la confirmer.
- N'invente jamais une source, une page ou un numéro de passage.
"""

        # Réduction défensive des contextes complémentaires pour respecter
        # la limite de tokens du modèle Groq.
        ecg_reference_context = _compact_text(ecg_reference_context, 2200)
        research_context = _compact_text(research_context, 2200)
        ptbxl_context = _compact_text(ptbxl_context, 1800)
        web_context = _compact_text(web_context, 1800)

        # ========================================================
        # RÉSULTAT SIGNALAGENT
        # ========================================================

        signal_block = ""

        if (
            signal_result
            and isinstance(signal_result, dict)
            and signal_result.get("status") == "success"
        ):
            raw_probabilities = signal_result.get("probabilities", {}) or {}

            rounded_probabilities = {}
            for class_name, value in raw_probabilities.items():
                try:
                    rounded_probabilities[class_name] = round(float(value), 3)
                except (TypeError, ValueError):
                    rounded_probabilities[class_name] = value

            signal_block = f"""
════════════════════════════════════
RÉSULTAT DU SIGNALAGENT
════════════════════════════════════

Le bloc suivant est une sortie de modèle sur l'ECG courant.
Il ne constitue PAS une recommandation clinique et ne doit jamais
être cité comme un passage documentaire.

Classes prédites :
{signal_result.get("predicted_classes", [])}

Classes au-dessus du seuil :
{signal_result.get("classes_above_threshold", signal_result.get("predicted_classes", []))}

Classe retenue par repli :
{signal_result.get("fallback_class")}

Probabilités / scores du modèle (arrondis à 3 décimales) :
{rounded_probabilities}

Mode de décision :
{signal_result.get("decision_mode")}

Règles :
- Toujours afficher les scores avec au maximum 3 décimales.
- Décris ces valeurs comme des sorties du modèle.
- Ne transforme jamais une prédiction en diagnostic certain.
- Ne cite jamais ce bloc comme "passage" ou "source documentaire".
- Les références passage/page concernent uniquement les blocs documentaires
  et doivent conserver la catégorie exacte indiquée dans chaque bloc.
"""

        # ========================================================
        # ECG REFERENCE
        # ========================================================

        ecg_reference_block = ""

        if ecg_reference_context and str(ecg_reference_context).strip():
            ecg_reference_block = f"""
====================================
RÉFÉRENCES TECHNIQUES ECG
====================================

{ecg_reference_context}

Règles :
- Support technique ECG uniquement ; ce bloc n'est pas une guideline ESC/AHA.
- Cite la source et la page si tu l'utilises.
"""

        # ========================================================
        # RESEARCH PAPERS LOCAUX
        # ========================================================

        research_block = ""

        if research_context and str(research_context).strip():
            research_block = f"""
====================================
LITTÉRATURE SCIENTIFIQUE LOCALE
====================================

{research_context}

Règles :
- Littérature scientifique complémentaire, non normative.
- Attribue chaque résultat à son étude et ne généralise pas quelques articles
  à l'ensemble du domaine.
- Cite la source et la page.
"""

        # ========================================================
        # CAS PTB-XL
        # ========================================================

        ptbxl_block = ""

        if ptbxl_context and str(ptbxl_context).strip():
            case_ids = list(ptbxl_case_ids or [])[:3]
            case_ids_text = (
                ", ".join(str(case_id) for case_id in case_ids)
                if case_ids
                else "non disponibles"
            )

            ptbxl_block = f"""
════════════════════════════════════
CAS PTB-XL SIMILAIRES
════════════════════════════════════

IDs des cas PTB-XL les plus proches :
{case_ids_text}

{ptbxl_context}

Règles PTB-XL :
- Cas comparatifs uniquement ; ils ne remplacent jamais les guidelines.
- Une similarité PTB-XL ne confirme pas un diagnostic.
- Si utilisés, cite les ECG ID fournis et ne les mélange pas avec les passages.
"""

        # ========================================================
        # RECHERCHE WEB EXTERNE
        # ========================================================

        web_block = ""

        if web_context and str(web_context).strip():
            web_block = f"""
════════════════════════════════════
LITTÉRATURE WEB EXTERNE
════════════════════════════════════

Source retenue : {web_source or "non précisée"}

{web_context}

Règles Web :
- Source complémentaire uniquement ; les guidelines locales restent prioritaires.
- Attribue chaque information à sa source et ne la présente jamais comme ESC/AHA.
- Ne généralise pas les résultats d'un petit nombre d'études.
"""

        if (
            signal_block
            or ecg_reference_block
            or research_block
            or ptbxl_block
            or web_block
        ):
            prompt += f"""

════════════════════════════════════
DONNÉES COMPLÉMENTAIRES
════════════════════════════════════

{signal_block}

{ecg_reference_block}

{research_block}

{ptbxl_block}

{web_block}

RÈGLES DE SOURCES :
1. Guidelines ESC/AHA = source normative principale.
2. SignalAgent = sortie du modèle, jamais recommandation normative.
3. ECG reference = support technique.
4. Research/Web = littérature complémentaire.
5. PTB-XL = comparaison uniquement.

Traçabilité :
- guideline -> "passage guideline" ;
- ECG reference -> "passage ECG REFERENCE" ;
- research -> "passage RESEARCH" ;
- PTB-XL -> "cas PTB-XL" + ECG ID ;
- web -> "passage WEB".
N'invente jamais une référence et ne change jamais la catégorie d'une source.
"""

        # ========================================================
        # FEEDBACK DU CRITICAGENT
        # ========================================================

        if critic_feedback:
            previous_block = ""
            if previous_answer:
                previous_block = f"""
RÉPONSE PRÉCÉDENTE À CORRIGER :
<<<
{previous_answer}
>>>
"""
            prompt += f"""
{previous_block}
════════════════════════════════════
RETOUR DU CRITICAGENT
════════════════════════════════════

La réponse précédente contient les problèmes suivants :

{critic_feedback}

Tu dois corriger TOUS les problèmes signalés avant de répondre.

Règles de correction :
- Corrige tous les points signalés sans ajouter de nouvelle information médicale.
- Conserve les faits du cas utilisateur.
- Toute interprétation doit rester soutenue par les sources fournies.
- PTB-XL, Web et SignalAgent ne doivent jamais être présentés comme guidelines.
- N'invente aucune référence, page ou numéro de passage.
- Produis directement les cinq sections complètes.
"""

        response = self.llm.invoke(prompt)

        return {
            "agent": "ClinicalAgent",
            "answer": response.content.strip(),
        }
