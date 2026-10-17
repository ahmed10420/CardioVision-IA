import json
import re

from dotenv import load_dotenv

from agents.groq_client import create_groq_llm


class CriticAgent:

    # Le score et le verdict portent sur les mêmes critères.
    SCORED_CRITERIA = [
        "qrs_duration",
        "st_elevation",
        "urgency",
        "atrial_fibrillation",
        "safety_disclaimer",
        "guideline_support",
        "citation_style",
    ]

    # Tous les critères utilisés pour déterminer le verdict final
    ALL_CRITERIA = SCORED_CRITERIA

    LIMITS_HEADING = re.compile(
        r"(?im)^[ \t]*(?:\*\*)?4\.[ \t]*Limites du contexte"
        r"(?:\*\*)?[ \t]*\\?[ \t]*$"
    )
    JUSTIFICATION_HEADING = re.compile(
        r"(?im)^[ \t]*(?:\*\*)?3\.[ \t]*Justification documentaire"
        r"(?:\*\*)?[ \t]*\\?[ \t]*$"
    )
    JUSTIFICATION_END = re.compile(
        r"(?im)^[ \t]*(?:\*\*)?4\.[ \t]*Limites du contexte"
    )
    LITERAL_QUOTE = re.compile(
        r"(?:«[^»\n]+»|“[^”\n]+”|\"[^\"\n]+\")"
    )
    NEXT_HEADING = re.compile(r"(?im)^[ \t]*(?:\*\*)?5\.[ \t]*")
    FORBIDDEN_EXAMPLE = re.compile(
        r"(?i)\b(?:par\s+exemple|(?:par\s+)?ex\.)"
    )

    @classmethod
    def forbidden_example_in_limits(cls, answer):
        """Contrôle la règle lexicale uniquement dans la section 4 de la réponse."""

        heading = cls.LIMITS_HEADING.search(answer)
        if heading is None:
            return False
        next_heading = cls.NEXT_HEADING.search(answer, heading.end())
        section = answer[heading.end():next_heading.start() if next_heading else None]
        return cls.FORBIDDEN_EXAMPLE.search(section) is not None

    @classmethod
    def literal_quote_in_justification(cls, answer):
        """Détecte toute citation entre guillemets dans la section 3."""
        heading = cls.JUSTIFICATION_HEADING.search(answer)
        if heading is None:
            return False

        end_heading = cls.JUSTIFICATION_END.search(answer, heading.end())
        section = answer[
            heading.end():end_heading.start() if end_heading else None
        ]
        return cls.LITERAL_QUOTE.search(section) is not None

    def __init__(self):
        load_dotenv()

        self.llm = create_groq_llm(
           model="openai/gpt-oss-120b",
            max_tokens=1200,
        )

    # ============================================================
    # CONSTRUCTION DU PROMPT DU CRITIC
    # ============================================================

    def build_prompt(self, question, context, answer):

        return f"""
Tu es CriticAgent, un auditeur médical spécialisé en ECG.

Ta mission est d'évaluer rigoureusement une réponse produite
par ClinicalAgent.

Tu dois juger la réponse à partir de DEUX sources autorisées uniquement :
1. la QUESTION, qui contient les faits d'entrée du cas utilisateur ;
2. le CONTEXTE RAG, qui contient les connaissances médicales autorisées.

Les faits explicitement présents dans la QUESTION peuvent être repris dans la
réponse sans devoir apparaître dans le RAG.

En revanche, toute interprétation médicale de ces faits — diagnostic, critère
diagnostique, syndrome, niveau d'urgence, pronostic, traitement, seuil normatif
ou recommandation — doit être explicitement soutenue par le CONTEXTE RAG.

Les connaissances internes du LLM ne sont jamais une source valide.

============================================================
QUESTION
============================================================

{question}

============================================================
CONTEXTE RAG AUTORISÉ
============================================================

{context}

============================================================
RÉPONSE CLINIQUE À ÉVALUER
============================================================

{answer}

============================================================
CRITÈRES À ÉVALUER
============================================================

1. qrs_duration

La gestion de la durée QRS est-elle correcte si ce critère
est pertinent pour la question ?

Si la question et le contexte ne concernent pas la durée QRS,
retourne "N/A".


2. st_elevation

Si la question contient une observation de sus-décalage ST, la réponse peut
reprendre fidèlement cette observation comme fait du cas.

Si la réponse transforme cette observation en critère diagnostique, seuil
officiel, STEMI ou recommandation, cette interprétation doit être explicitement
soutenue par le contexte RAG.

Si le sujet n'est pas pertinent pour la question, retourne "N/A".


3. urgency

Le niveau d'urgence, la nécessité d'une activation urgente ou une conduite
à tenir urgente ne peuvent être affirmés que si le contexte RAG les soutient
explicitement.

Le fait que le patient décrive une douleur, une durée de symptômes ou des
signes vitaux peut être repris comme donnée du cas, mais ne suffit pas à créer
une recommandation d'urgence sans support RAG.

Si aucune notion d'urgence n'est pertinente, retourne "N/A".


4. atrial_fibrillation

Si la question contient des observations ECG telles qu'un rythme irrégulier,
des RR irréguliers ou l'absence d'ondes P, la réponse peut les reprendre comme
faits du cas.

Toute conclusion de fibrillation atriale ou tout énoncé de critères de FA doit
être explicitement soutenu par le contexte RAG. N'accepte aucun critère ajouté
depuis les connaissances internes du modèle.

Si la question ne concerne pas la fibrillation atriale, retourne "N/A".


5. safety_disclaimer

La réponse contient-elle un avertissement médical clair indiquant
qu'elle constitue une aide à la décision et qu'elle ne remplace
pas le jugement d'un professionnel de santé ?

Ce critère doit toujours être PASS ou FAIL.


6. guideline_support

Toutes les affirmations médicales importantes sont-elles
explicitement soutenues par le contexte RAG ?

RÈGLE TRÈS IMPORTANTE :

La réponse ne doit introduire aucune information médicale
absente du contexte.

Cela s'applique également à la section
"Limites du contexte".

Une information médicale absente du contexte ne doit pas être
introduite simplement pour dire qu'elle n'est pas précisée.

Pour la section 4, contrôle séparément CHAQUE affirmation :
- toute limite médicale doit être directement démontrable dans les passages ;
- aucun concept médical, population, examen, seuil, diagnostic ou modalité
  de prise en charge absent des passages ne peut y être mentionné ;
- une affirmation exhaustive comme "aucun document ne...", "les passages ne
  traitent pas de..." ou "aucune autre information..." doit être refusée si
  elle n'est pas prouvable à partir de tous les passages disponibles ;
- la phrase neutre "Les informations disponibles permettent de répondre à la
  question dans le périmètre des passages fournis." ne constitue pas une
  affirmation médicale nouvelle et peut être acceptée.

Si une seule affirmation de la section 4 échoue à ce contrôle,
mets guideline_support à FAIL et décris précisément cette affirmation
dans le feedback.

Vérifie aussi explicitement que :
- une durée ou un seuil reste attaché uniquement à la modalité concernée
  dans le contexte ;
- les architectures testées dans quelques études ne sont pas présentées
  comme les méthodes principales ou les plus utilisées sans preuve explicite ;
- les formulations générales telles que "les méthodes récentes utilisent",
  "les approches récentes reposent sur", "principalement" ou "majoritairement"
  sont refusées lorsqu'elles reposent seulement sur quelques articles ;
- chaque résultat de recherche est attribué explicitement à son étude ; une
  généralisation n'est acceptée que si une revue systématique ou une guideline
  du contexte la soutient explicitement ;
- les paramètres expérimentaux d'une étude ne sont pas généralisés à tous
  les algorithmes ;
- tous les ECG ID PTB-XL retenus dans le contexte sont cités ensemble, sans
  omission ni sous-liste contradictoire.

Par exemple, si le contexte ne parle pas d'une classification,
d'un seuil, d'un diagnostic, d'une durée ou d'un trouble particulier,
la réponse ne doit pas introduire cette notion.

Si une telle information apparaît, mets :

"guideline_support": "FAIL"


7. citation_style

Vérifie les deux points suivants (la règle lexicale de la section 4
est contrôlée séparément par le programme Python) :

A. La justification documentaire doit être rédigée
en paraphrase française.

Elle ne doit pas reproduire une phrase du contexte
entre guillemets ni effectuer une traduction littérale.

B. La justification documentaire doit conserver
des références précises aux passages et pages lorsqu'elles
sont disponibles dans le contexte.

C. Chaque affirmation doit conserver la catégorie exacte de sa source :
- guideline -> passage guideline ;
- ECG REFERENCE -> passage ECG REFERENCE ;
- RESEARCH LOCAL -> passage RESEARCH ;
- PTB-XL comparatif -> cas PTB-XL avec ECG ID ;
- WEB EXTERNE -> passage WEB.

Un article scientifique qui décrit PTB-XL reste un passage RESEARCH et ne
doit pas être attribué à ECG REFERENCE ou à un cas PTB-XL individuel.
Mets citation_style = "FAIL" si la catégorie annoncée ne correspond pas au
bloc du contexte qui soutient l'affirmation.

N'invente jamais un numéro de passage, une page, un titre, une source ou une
catégorie. Une référence n'est valide que si cet identifiant apparaît réellement
dans le contexte fourni.

Mets citation_style = "FAIL" si l'un de ces problèmes est présent.

Ce critère n'est jamais "N/A".


============================================================
RÈGLES GÉNÉRALES
============================================================

- Utilise uniquement la QUESTION pour les faits du cas utilisateur.
- Utilise uniquement le CONTEXTE RAG pour les connaissances et interprétations médicales.
- N'utilise aucune connaissance médicale externe.
- Ne complète jamais une information manquante.
- Ne fais aucune supposition.
- Ne fais aucune déduction médicale non explicitement soutenue par le RAG.
- Ne pénalise pas une simple reprise fidèle d'un fait explicitement fourni dans la QUESTION.

Pour les critères pouvant être non pertinents :

PASS = critère pertinent et correctement respecté.

FAIL = critère pertinent mais incorrect ou non respecté.

N/A = critère non pertinent pour cette question.

Ne pénalise jamais la réponse pour l'absence d'un élément
qui n'est pas pertinent.


============================================================
FORMAT DE SORTIE
============================================================

Retourne UNIQUEMENT un JSON valide.

Aucun texte avant le JSON.
Aucun texte après le JSON.
Aucun bloc Markdown.

Utilise exactement cette structure :

{{
  "criteria": {{
    "qrs_duration": "PASS|FAIL|N/A",
    "st_elevation": "PASS|FAIL|N/A",
    "urgency": "PASS|FAIL|N/A",
    "atrial_fibrillation": "PASS|FAIL|N/A",
    "safety_disclaimer": "PASS|FAIL",
    "guideline_support": "PASS|FAIL",
    "citation_style": "PASS|FAIL"
  }},
  "feedback": [
    "problème 1",
    "problème 2"
  ]
}}

Le score et le verdict seront calculés automatiquement
par le programme Python.
Ne retourne pas de score et ne retourne pas de verdict.
"""

    # ============================================================
    # NORMALISATION DES STATUTS
    # ============================================================

    @staticmethod
    def normalize_status(value, allow_na=True):

        value = str(value).strip().upper()

        allowed = {"PASS", "FAIL"}

        if allow_na:
            allowed.add("N/A")

        if value not in allowed:
            return "FAIL"

        return value

    # ============================================================
    # CALCUL DU SCORE /7
    # ============================================================

    def calculate_score(self, criteria):

        score = 0

        for criterion in self.SCORED_CRITERIA:

            status = criteria.get(
                criterion,
                "FAIL",
            )

            # PASS = critère respecté
            #
            # N/A = critère non applicable,
            # donc il ne doit pas pénaliser la réponse.
            #
            # Cela permet de conserver une échelle fixe /7.
            if status in {"PASS", "N/A"}:
                score += 1

        return score

    def calculate_applicable_score(self, criteria):
        """Calcule le score uniquement sur les critères non marqués N/A."""

        applicable_statuses = [
            criteria.get(criterion, "FAIL")
            for criterion in self.SCORED_CRITERIA
            if criteria.get(criterion, "FAIL") != "N/A"
        ]
        applicable_max_score = len(applicable_statuses)
        applicable_score = sum(
            status == "PASS"
            for status in applicable_statuses
        )
        applicable_normalized = (
            round(applicable_score / applicable_max_score, 4)
            if applicable_max_score
            else None
        )

        return (
            applicable_score,
            applicable_max_score,
            applicable_normalized,
        )

    # ============================================================
    # CALCUL DÉTERMINISTE DU VERDICT
    # ============================================================

    def calculate_verdict(self, criteria):

        failed_criteria = []

        for criterion in self.ALL_CRITERIA:

            status = criteria.get(
                criterion,
                "FAIL",
            )

            if status == "FAIL":
                failed_criteria.append(criterion)

        number_of_failures = len(failed_criteria)

        if number_of_failures == 0:
            return "PASS"

        if number_of_failures <= 2:
            return "IMPROVE"

        return "FAIL"

    # ============================================================
    # EXÉCUTION DU CRITIC AGENT
    # ============================================================

    def run(self, question, context, answer):

        prompt = self.build_prompt(
            question=question,
            context=context,
            answer=answer,
        )

        response = self.llm.invoke(prompt)

        raw = response.content.strip()

        # --------------------------------------------------------
        # Nettoyer un éventuel bloc ```json ... ```
        # --------------------------------------------------------

        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()

        # --------------------------------------------------------
        # Parsing JSON
        # --------------------------------------------------------

        try:

            result = json.loads(raw)

        except json.JSONDecodeError:

            return {
                "agent": "CriticAgent",
                "criteria": {},
                "score": 0,
                "max_score": len(self.SCORED_CRITERIA),
                "applicable_score": 0,
                "applicable_max_score": 0,
                "applicable_normalized": None,
                "verdict": "IMPROVE",
                "feedback": [
                    "CriticAgent n'a pas retourné un JSON valide."
                ],
            }

        # --------------------------------------------------------
        # Récupérer les critères
        # --------------------------------------------------------

        raw_criteria = result.get(
            "criteria",
            {},
        )

        # --------------------------------------------------------
        # Normaliser chaque critère
        # --------------------------------------------------------

        criteria = {
            "qrs_duration": self.normalize_status(
                raw_criteria.get(
                    "qrs_duration",
                    "FAIL",
                ),
                allow_na=True,
            ),

            "st_elevation": self.normalize_status(
                raw_criteria.get(
                    "st_elevation",
                    "FAIL",
                ),
                allow_na=True,
            ),

            "urgency": self.normalize_status(
                raw_criteria.get(
                    "urgency",
                    "FAIL",
                ),
                allow_na=True,
            ),

            "atrial_fibrillation": self.normalize_status(
                raw_criteria.get(
                    "atrial_fibrillation",
                    "FAIL",
                ),
                allow_na=True,
            ),

            "safety_disclaimer": self.normalize_status(
                raw_criteria.get(
                    "safety_disclaimer",
                    "FAIL",
                ),
                allow_na=False,
            ),

            "guideline_support": self.normalize_status(
                raw_criteria.get(
                    "guideline_support",
                    "FAIL",
                ),
                allow_na=False,
            ),

            "citation_style": self.normalize_status(
                raw_criteria.get(
                    "citation_style",
                    "FAIL",
                ),
                allow_na=False,
            ),
        }

        if self.forbidden_example_in_limits(answer):
            criteria["citation_style"] = "FAIL"

        if self.literal_quote_in_justification(answer):
            criteria["citation_style"] = "FAIL"

        # --------------------------------------------------------
        # Calcul Python du score
        # --------------------------------------------------------

        score = self.calculate_score(
            criteria
        )

        (
            applicable_score,
            applicable_max_score,
            applicable_normalized,
        ) = self.calculate_applicable_score(criteria)

        # --------------------------------------------------------
        # Calcul Python du verdict
        # --------------------------------------------------------

        verdict = self.calculate_verdict(
            criteria
        )

        # --------------------------------------------------------
        # Feedback
        # --------------------------------------------------------

        feedback = result.get(
            "feedback",
            [],
        )

        if not isinstance(feedback, list):
            feedback = [str(feedback)]

        if self.forbidden_example_in_limits(answer):
            message = (
                'La section "Limites du contexte" contient "par exemple", '
                '"par ex." ou "ex." ; supprimer cette formulation et tout '
                "exemple non documenté."
            )
            if message not in feedback:
                feedback.append(message)

        if self.literal_quote_in_justification(answer):
            message = (
                'La section "Justification documentaire" contient une citation '
                "entre guillemets ; la remplacer par une paraphrase française "
                "en conservant la référence exacte au passage."
            )
            if message not in feedback:
                feedback.append(message)

        # --------------------------------------------------------
        # Si FAIL sans feedback, créer un feedback minimal
        # --------------------------------------------------------

        if verdict != "PASS" and not feedback:

            failed = [
                criterion
                for criterion, status
                in criteria.items()
                if status == "FAIL"
            ]

            feedback = [
                "Critères à corriger : "
                + ", ".join(failed)
            ]

        # --------------------------------------------------------
        # Résultat final du CriticAgent
        # --------------------------------------------------------

        return {
            "agent": "CriticAgent",

            "criteria": criteria,

            "score": score,

            "max_score": len(self.SCORED_CRITERIA),

            "applicable_score": applicable_score,

            "applicable_max_score": applicable_max_score,

            "applicable_normalized": applicable_normalized,

            "verdict": verdict,

            "feedback": feedback,
        }
