from dotenv import load_dotenv

from agents.groq_client import create_groq_llm
from rag.rag_with_llm import MEDICAL_WARNING


class SynthesizerAgent:
    
    

    def __init__(self):
        load_dotenv()

        self.llm = create_groq_llm(
             model="openai/gpt-oss-20b",
            max_tokens=1800,
        )

    def run(
        self,
        question,
        clinical_answer,
        critic_result,
        signal_context="",
    ):

        prompt = f"""
Tu es SynthesizerAgent.

Question utilisateur :
{question}

Réponse de ClinicalAgent :
{clinical_answer}

Évaluation de CriticAgent :
{critic_result}

Résultat structuré de SignalAgent :
{signal_context}

Produis une seule réponse finale cohérente.

Règles générales :
- Réponds uniquement en français.
- Ne rajoute aucune information médicale.
- Si un résultat SignalAgent est présent, ne l'omets jamais de la réponse finale.
- S'il existe une ou plusieurs classes au-dessus du seuil, rapporte-les comme
  "catégories du modèle" avec leurs scores, sans les présenter comme diagnostic certain.
- Si SignalAgent a retenu une classe sans franchir son seuil, rapporte uniquement
  la classe et le score chiffré indiqués dans son contexte.
  N'emploie pas « probabilité élevée », « forte confiance » ni « ECG normal »
  pour décrire ce résultat du modèle.
- Toute interprétation clinique d'une catégorie SignalAgent doit rester fondée
  sur les éléments déjà présents dans clinical_answer ; n'en invente aucune.
- Ne modifie pas une information correcte.
- Corrige les problèmes explicitement signalés par CriticAgent.
- Si des ECG ID PTB-XL sont présents, conserve ensemble la liste complète et
  exacte ; n'en supprime aucun et ne produis pas une sous-liste différente.
- Ne transforme pas les modèles testés par quelques études en méthodes
  principales ou les plus utilisées.
- Conserve la portée exacte des durées : n'applique jamais à un ECG 12 dérivations
  une durée que la source réserve au tracé monodérivation.
- Ne mentionne pas les agents.
- Ne mentionne pas le processus de critique.
- Conserve une réponse concise.

Règles d'hygiène toujours appliquées, même si CriticAgent ne les a pas signalées :
- Ne reproduis jamais une phrase du contexte ou de la réponse de ClinicalAgent entre
  guillemets, et ne la traduis jamais mot à mot. Si la réponse de ClinicalAgent contient
  une citation littérale, reformule-la entièrement en paraphrase française avant de
  l'inclure.
- Attention : paraphraser ne veut pas dire supprimer la référence au passage. La
  section "Justification documentaire" doit toujours nommer le numéro du passage et sa
  page (ex. "Le passage 2, extrait des recommandations ESC 2024, page 11, indique que...")
  tout en reformulant son contenu sans guillemets. Si la réponse de ClinicalAgent citait
  "Le passage N, page P" avant correction, conserve cette référence exacte en ne
  paraphrasant que le contenu qui suit.
- Dans la section "Limites du contexte", n'utilise jamais les mots « par exemple » ni
  « ex. » : si ClinicalAgent en contient, retire l'exemple et garde uniquement la
  catégorie d'information manquante.
- N'utilise aucun anglicisme tel que « pattern », « non-ST-elevation » ou « FMC ».

Structure obligatoire :
- Produis exactement ces cinq titres, textuellement identiques, en début de ligne, dans
  cet ordre, sans les renuméroter, fusionner, renommer ni changer leur format (pas de
  gras, pas de "Section", pas de deux-points après le chiffre) :
1. Réponse directe
2. Critères cliniques et ECG
3. Justification documentaire
4. Limites du contexte
5. Avertissement médical
- Chaque titre doit être suivi du contenu de sa section, jamais laissé vide.
- La section 5 doit contenir exactement, mot pour mot et rien d'autre, cet avertissement :


{MEDICAL_WARNING}
IMPORTANT :
Le SynthesizerAgent ne doit jamais enrichir la réponse avec de nouvelles informations médicales.
Il doit uniquement reformuler les éléments déjà présents dans clinical_answer et validés par critic_result.
"""

        response = self.llm.invoke(prompt)

        return {
            "agent": "SynthesizerAgent",
            "answer": response.content.strip(),
        }
