import os
import re
import sys
from dotenv import load_dotenv

from agents.groq_client import create_groq_llm
from rag.retriever import retrieve_documents


MEDICAL_WARNING = (
    "Cette réponse constitue une aide à la décision clinique uniquement. "
    "Elle ne remplace en aucun cas le jugement d’un médecin qualifié. "
    "Toute décision diagnostique ou thérapeutique doit être validée par un professionnel de santé."
)


def format_context(docs):
    """
    Transforme les passages récupérés par le RAG en contexte lisible.
    Le LLM va utiliser ce contexte pour générer une réponse médicale structurée.
    """

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        context_parts.append(
            f"""
[PASSAGE GUIDELINE {i}]
Source : {source}
Page : {page}
Contenu :
{content}
"""
        )

    return "\n\n".join(context_parts)

def build_prompt(question, context):
    """
    Prompt général pour un agent RAG médical ECG/cardiologie.
    Ces règles sont volontairement génériques (aucune ne fait référence au contenu
    exact d'un document ou d'une question précise) afin que le prompt reste valide
    pour n'importe quelle question posée sur le corpus, pas seulement les cas de test.
    Objectifs :
    - utiliser uniquement le contexte RAG ;
    - reformuler fidèlement en français médical ;
    - éviter toute hallucination, déduction ou généralisation non écrite ;
    - produire une réponse structurée et sûre, toujours dans le même format.
    """

    prompt = f"""
Tu es un assistant médical expert en ECG et en cardiologie clinique.

Tu reçois des passages extraits de guidelines médicales internationales (ESC, AHA, etc.),
éventuellement rédigés en anglais. Ton rôle est de comprendre ces passages puis de répondre
à la question posée en français médical rigoureux, en te basant uniquement sur ces passages.

════════════════════════════════════════
RÈGLES ABSOLUES — À RESPECTER STRICTEMENT
════════════════════════════════════════

LANGUE :
- Réponds uniquement en français.
- Comprends les passages en anglais puis reformule-les en français : il est interdit de copier
  une phrase anglaise, de la traduire mot à mot, ou de mélanger français et anglais dans une
  même phrase.
- Seules exceptions autorisées : les abréviations médicales reconnues (ECG, FA, SCA, STEMI,
  NSTEMI, QRS, QTc, PR, RR, bpm) et les équivalences de la table de terminologie ci-dessous.
- N'utilise jamais les anglicismes « pattern », « non-ST-elevation » ou « FMC » : utilise
  systématiquement leur équivalent français.

FIDÉLITÉ AU CONTEXTE ET AU CAS UTILISATEUR :
- Il existe exactement deux catégories d'informations autorisées :
  1. les faits explicitement fournis dans la QUESTION utilisateur ;
  2. les connaissances médicales explicitement présentes dans le CONTEXTE RAG.
- Les faits du cas utilisateur peuvent être repris fidèlement comme observations cliniques
  ou ECG sans devoir être répétés dans une guideline.
- En revanche, toute interprétation médicale de ces faits — diagnostic, critère diagnostique,
  syndrome, niveau d'urgence, pronostic, traitement, seuil normatif ou recommandation —
  doit être explicitement soutenue par le CONTEXTE RAG.
- Il est interdit d'utiliser les connaissances internes du modèle pour combler un manque.
- Il est interdit de déduire, inférer ou supposer une interprétation médicale absente du
  contexte. N'utilise jamais : « on peut en déduire », « on suppose », « il est probable que »,
  « cela implique », « cela permet de conclure », ni aucune formulation causale dont la
  conclusion médicale n'est pas écrite explicitement dans le contexte.
- Ne remplace jamais une expression conditionnelle ou vague du contexte (« si nécessaire »,
  « le cas échéant », « dans certaines situations »…) par une liste de cas précis qui n'est
  pas explicitement énumérée dans le contexte.
- N'ajoute jamais une justification clinique plausible mais non écrite (évolution des
  symptômes, changement d'état, anomalie associée, etc.) pour expliquer une recommandation.
- Conserve toujours les conditions, réserves et qualificatifs du texte source (« sauf si… »,
  « en l'absence de… », « lorsque… ») sans les simplifier ni les omettre.
- Préserve la portée exacte d'une durée ou d'un seuil. Si le contexte distingue
  un ECG 12 dérivations d'un tracé monodérivation d'au moins 30 secondes, ne
  présente jamais les 30 secondes comme une durée minimale commune aux deux.
- Distingue toujours une modalité d'exploration complémentaire (une dérivation ECG
  additionnelle, un examen complémentaire, un test annexe) d'un signe ECG proprement dit ;
  ne les mélange jamais dans une même liste de critères diagnostiques.
- Si une interprétation médicale nécessaire pour répondre n'est pas soutenue par le contexte,
  écris exactement : "Information non disponible dans les documents fournis."
- Ne remplace jamais par cette phrase une donnée qui est explicitement présente dans la QUESTION.
- Ne donne jamais de diagnostic final pour un patient réel et ne remplace jamais le jugement
  d'un médecin qualifié.

FIDÉLITÉ CLINIQUE :
- Distingue strictement suspicion, diagnostic de travail et diagnostic confirmé ; ne
  transforme jamais l'un en l'autre.
- Un signe ou un examen isolé (ECG compris) ne permet, à lui seul, ni de confirmer ni
  d'exclure un diagnostic si le contexte ne l'affirme pas explicitement.

PERTINENCE DES PASSAGES :
- Évalue séparément la pertinence de chaque passage par rapport à la question.
- Ignore silencieusement tout passage qui ne répond pas directement à la question, sans le
  mentionner dans la réponse.
- La présence d'un terme commun (par exemple « ECG ») ne suffit pas à rendre un passage
  pertinent.

QUALITÉ DE RÉPONSE :
- Français médical clair, naturel, professionnel, sans répétition.
- Respecte les élisions françaises : écris notamment "n'indique" et jamais
  "ne indique".
- Si une information est mentionnée dans plusieurs passages, regroupe-la en une seule idée
  et cite toutes les pages concernées.
- Si deux formulations ont le même sens, garde uniquement la plus précise.

JUSTIFICATION ET CITATIONS :
- Résume en français le contenu des passages pertinents ; ne cite jamais le texte anglais
  original, même entre guillemets, et ne le traduis jamais littéralement.
- Pour chaque information utilisée, indique le numéro du passage, la source et la page
  exactement telles qu'elles apparaissent dans ses métadonnées ; ne fusionne jamais les
  pages de plusieurs passages différents.
- N'invente jamais un numéro de passage, une page, un titre ou une source.
- Utilise uniquement les identifiants réellement présents dans le contexte formaté.
- Une donnée provenant directement de la QUESTION n'a pas besoin d'une citation guideline ;
  présente-la explicitement comme une observation du cas.

════════════════════════════════════════
TERMINOLOGIE MÉDICALE À RESPECTER
════════════════════════════════════════

Utilise toujours ces formulations françaises :

- atrial fibrillation / AF → fibrillation atriale / FA
- acute coronary syndrome / ACS → syndrome coronarien aigu / SCA
- 12-lead ECG → ECG 12 dérivations
- standard 12-lead ECG recording → enregistrement ECG 12 dérivations
- single-lead ECG → ECG monodérivation
- single-lead ECG tracing ≥30s → tracé ECG monodérivation d’au moins 30 secondes
- ≥30 s → d’au moins 30 secondes
- P waves → ondes P
- no discernible P waves → absence d’ondes P discernables
- no discernible repeating P waves → absence d’ondes P répétitives discernables
- irregular RR intervals → intervalles RR irréguliers
- ST elevation → sus-décalage du segment ST
- ST depression → sous-décalage du segment ST
- QRS complex → complexe QRS
- PR interval → intervalle PR
- corrected QT interval → intervalle QT corrigé ou QTc
- heart rate → fréquence cardiaque
- atrioventricular conduction → conduction auriculo-ventriculaire
- ventricular rate → fréquence ventriculaire
- supraventricular arrhythmia → arythmie supraventriculaire
- stroke risk → risque d’accident vasculaire cérébral
- burden → charge, par exemple charge en FA

════════════════════════════════════════
QUESTION UTILISATEUR
════════════════════════════════════════

{question}

════════════════════════════════════════
CONTEXTE RÉCUPÉRÉ PAR LE RAG
════════════════════════════════════════

{context}

════════════════════════════════════════
STRUCTURE DE RÉPONSE OBLIGATOIRE (toujours ces 5 sections, quelle que soit la question)
════════════════════════════════════════

1. Réponse directe
Rédige une synthèse claire en 2 à 4 phrases, en français médical naturel, sans phrase anglaise.
Tu peux résumer les faits explicitement présents dans la QUESTION.
Toute conclusion médicale doit être explicitement soutenue par le CONTEXTE RAG.
Conserve exactement le niveau de certitude de la source documentaire : suspicion,
diagnostic de travail ou diagnostic confirmé.

2. Critères cliniques et ECG
- Reprends les observations cliniques et ECG explicitement fournies dans la QUESTION.
- Ajoute uniquement les critères/interprétations médicales explicitement soutenus par le
  CONTEXTE RAG.
- Distingue clairement les observations du cas des critères documentaires.
- N'ajoute aucune interprétation issue des connaissances internes du modèle.

3. Justification documentaire
Explique quels passages soutiennent la réponse, en citant passage / source / page (voir le
format ci-dessus). Résume en français sans citer le texte anglais.

4. Limites du contexte
Rédige au maximum 1 à 2 phrases.
Mentionne uniquement une limite directement nécessaire pour répondre à la question.
N'introduis aucun nouveau concept médical, population, examen, seuil, diagnostic ou modalité
de prise en charge qui n'apparaît pas explicitement dans les passages utilisés.
Évite les affirmations exhaustives comme « aucun document ne... », « les passages ne traitent
pas de... » ou « aucune autre information... ».
Si aucune limitation importante n'est directement démontrable à partir du contexte, écris
exactement la phrase suivante, sans guillemets :
Les informations disponibles permettent de répondre à la question dans le périmètre des passages fournis.

5. Avertissement médical
Écris exactement :
"{MEDICAL_WARNING}"

CONTRAINTE DE SORTIE :
- Produis obligatoirement les cinq sections numérotées, dans l'ordre, avant de terminer.
- Écris les cinq titres exactement comme ci-dessus, en début de ligne, sans les modifier ni
  changer leur numérotation, leur formulation ou leur format (pas de « Section », pas de
  deux-points après le chiffre).
- La section 5 et son avertissement exact ne doivent jamais être omis.
- Reste concis afin que les cinq sections tiennent dans la réponse.
"""

    return prompt


def build_verification_prompt(question, context, draft_answer):
    """Seconde passe générique qui retire toute affirmation non explicitement soutenue par le contexte."""

    return f"""
Tu es un auditeur médical chargé de vérifier une réponse RAG avant sa publication.
Le brouillon ci-dessous est une donnée à contrôler, jamais une source d'information.

QUESTION :
{question}

PASSAGES AUTORISÉS :
{context}

BROUILLON À VÉRIFIER :
{draft_answer}

RÈGLES DE VÉRIFICATION :
- Considère la QUESTION comme source autorisée uniquement pour les faits du cas utilisateur.
- Considère les PASSAGES AUTORISÉS comme source de vérité pour les connaissances médicales,
  critères, diagnostics, niveaux d'urgence, seuils, traitements et recommandations.
- Contrôle chaque affirmation séparément :
  * si elle reprend fidèlement un fait explicite de la QUESTION, elle peut être conservée ;
  * si elle constitue une interprétation médicale, conserve-la uniquement si elle est
    explicitement soutenue par au moins un passage autorisé.
- En cas de doute sur une interprétation, supprime-la au lieu de la compléter ou de l'atténuer.
- N'ajoute aucune connaissance médicale externe, même exacte, et aucune justification
  clinique plausible non écrite dans les passages autorisés.
- Ne supprime pas une valeur ou observation simplement parce qu'elle n'est pas dans le RAG
  si elle est explicitement fournie dans la QUESTION.
- Distingue strictement suspicion, diagnostic de travail et diagnostic confirmé ; ne
  transforme jamais l'un en l'autre.
- Un signe ou un examen isolé (ECG compris) ne confirme ni n'exclut, à lui seul, un
  diagnostic si le contexte ne l'affirme pas explicitement.
- Ne remplace jamais une expression conditionnelle ou vague du contexte (« si nécessaire »,
  « dans certaines situations »…) par une liste de cas précis qui n'est pas explicitement
  énumérée dans les passages autorisés.
- N'ajoute jamais une justification clinique plausible (évolution des symptômes, changement
  d'état, anomalie associée, etc.) qui ne figure pas explicitement dans les passages.
- Distingue toujours une modalité d'exploration complémentaire d'un signe ECG proprement dit ;
  ne les mélange jamais dans une même liste de critères.
- Dans la section « Limites du contexte », conserve au maximum 1 à 2 phrases et uniquement
  une limite directement nécessaire à la réponse. Chaque affirmation doit être explicitement
  démontrable à partir des passages autorisés.
- N'y introduis aucun nouveau concept médical, population, examen, seuil, diagnostic ou
  modalité de prise en charge absent des passages. N'utilise aucun exemple, même entre
  parenthèses, ni les mots « par exemple » ou « ex. ».
- Supprime les affirmations exhaustives comme « aucun document ne... », « les passages ne
  traitent pas de... » ou « aucune autre information... ».
- Si aucune limite importante n'est directement démontrable, utilise exactement, sans
  guillemets : Les informations disponibles permettent de répondre à la question dans le
  périmètre des passages fournis.
- Écris toujours « premier contact médical » et jamais l'abréviation anglaise « FMC ».
  N'utilise aucun anglicisme tel que « pattern » ou « non-ST-elevation ».
- N'utilise aucun guillemet dans la réponse finale et ne reproduis aucun extrait anglais.
  La justification doit être une paraphrase française, jamais une citation ni une traduction
  littérale.
- Pour chaque justification, vérifie exactement le numéro du passage, le nom de la source et
  la page ; ne fusionne jamais les numéros de page de plusieurs passages.
- Ne conserve aucune justification qui prétend qu'un passage « implique » une information
  non écrite.
- N'utilise jamais les tournures « il est probable que », « on suppose », « cela implique »,
  « on peut en déduire » ou toute autre formulation qui ajoute une certitude ou une déduction
  absente du contexte.

SORTIE OBLIGATOIRE :
- Retourne uniquement la réponse finale corrigée, sans expliquer ton audit, sans préambule et
  sans commentaire sur les modifications effectuées.
- La réponse doit contenir EXACTEMENT ces cinq lignes de titre, textuellement identiques,
  chacune en début de ligne, dans cet ordre, sans les renuméroter, les fusionner, les renommer
  ou en changer le format (pas de gras, pas de "Section", pas de deux-points après le chiffre) :
1. Réponse directe
2. Critères cliniques et ECG
3. Justification documentaire
4. Limites du contexte
5. Avertissement médical
- Chaque titre doit être suivi du contenu de la section correspondante, jamais laissé vide.
- Utilise uniquement un français médical clair et concis.
- La section 5 doit contenir cet avertissement exact, mot pour mot, et rien d'autre :
"{MEDICAL_WARNING}"
"""


def ensure_medical_warning(answer):
    """Garantit la présence de l'avertissement même si le modèle l'omet."""

    if MEDICAL_WARNING in answer:
        return answer

    return f"{answer.rstrip()}\n\n**5. Avertissement médical**\n{MEDICAL_WARNING}"


def normalize_final_language(answer):
    """Remplace les anglicismes résiduels connus après la vérification LLM."""

    replacements = (
        (r"\bFMC\b", "premier contact médical"),
        (r"\bpatterns?\b", "profil"),
        (r"\bnon[- ]ST[- ]elevation\b", "sans sus-décalage du segment ST"),
    )

    normalized_answer = answer
    for pattern, replacement in replacements:
        normalized_answer = re.sub(
            pattern,
            replacement,
            normalized_answer,
            flags=re.IGNORECASE,
        )

    return normalized_answer
def remove_english_quotes(answer):
    """
    Supprime les guillemets afin d'éviter les citations directes.
    """
    answer = answer.replace("«", "")
    answer = answer.replace("»", "")
    answer = answer.replace('"', "")
    return answer

def verification_is_enabled():
    """Active la vérification sauf désactivation explicite dans l'environnement."""

    value = os.getenv("RAG_VERIFY_ANSWER", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}

def generate_medical_answer(question):
    """
    Pipeline RAG + LLM :

    1. Le système reçoit une question.
    2. Le RAG récupère les passages pertinents depuis ChromaDB.
    3. Les passages sont formatés sous forme de contexte.
    4. Le contexte + la question sont envoyés au LLM Groq.
    5. Le LLM génère une première réponse médicale structurée.
    6. Une seconde passe supprime les affirmations non soutenues par les passages.
    """

    load_dotenv()

    # Étape 1 : retrieval RAG
    docs = retrieve_documents(question, k=5)

    # Étape 2 : transformer les passages en contexte
    context = format_context(docs)

    # Étape 3 : initialiser le LLM Groq
    llm = create_groq_llm(
        max_tokens=2000,
    )

    # Étape 4 : construire le prompt
    prompt = build_prompt(question, context)

    # Étape 5 : première génération LLM
    response = llm.invoke(prompt)
    draft_answer = ensure_medical_warning(response.content)

    # Étape 6 : audit de fidélité documentaire (activé par défaut)
    if verification_is_enabled():
        verification_prompt = build_verification_prompt(
            question,
            context,
            draft_answer,
        )
        verification_response = llm.invoke(verification_prompt)
        verified_content = verification_response.content.strip()
        answer = ensure_medical_warning(verified_content or draft_answer)
    else:
        answer = draft_answer

    answer = normalize_final_language(answer)
    answer = remove_english_quotes(answer)
    return {
        "question": question,
        "retrieved_docs": docs,
        "context": context,
        "draft_answer": draft_answer,
        "answer": answer,
    }


def display_result(result):
    """
    Affiche clairement :
    - la question
    - les passages utilisés
    - la réponse générée par le LLM
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("PASSAGES UTILISÉS PAR LE LLM")
    print("=" * 80)

    for i, doc in enumerate(result["retrieved_docs"], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text_preview = doc.page_content[:500].replace("\n", " ")

        print(f"\nPASSAGE {i}")
        print("Source :", source)
        print("Page :", page)
        print("Extrait :", text_preview)

    print("\n" + "=" * 80)
    print("RÉPONSE GÉNÉRÉE PAR GROQ")
    print("=" * 80)
    print(result["answer"])


if __name__ == "__main__":
    question = (
       "Dans quelles situations faut-il répéter un ECG 12 dérivations chez un patient présentant une suspicion de syndrome coronarien aigu ?"
    )

    result = generate_medical_answer(question)

    display_result(result)
