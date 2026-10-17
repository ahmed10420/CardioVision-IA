from typing import TypedDict, Dict, Any


class AgentState(TypedDict):

    # Question utilisateur
    question: str

    # =========================================================
    # SIGNAL AGENT
    # =========================================================

    # Chemin vers l'image ECG
    ecg_input_path: str

    # Une analyse d'image est-elle indispensable pour cette requête ?
    ecg_required: bool

    # Résultat du pipeline :
    # image -> reconstruction -> classification
    signal_result: Dict[str, Any]

    # =========================================================
    # LITERATURE / RAG
    # =========================================================

    # Requête réellement utilisée pour le retrieval documentaire
    retrieval_query: str

    # Guidelines ESC/AHA
    rag_context: str

    # Références techniques / pédagogiques ECG
    ecg_reference_context: str

    # Research papers locaux
    research_context: str

    # Cas PTB-XL similaires récupérés par LiteratureAgent
    # Ils servent uniquement de cas comparatifs, pas de source normative.
    ptbxl_context: str

    # Identifiants des principaux cas PTB-XL retrouvés (traçabilité)
    ptbxl_case_ids: list[int]


    # =========================================================
    # WEB SEARCH
    # =========================================================

    # True si la question a déclenché une recherche Web
    web_search_used: bool

    # Résultat structuré complet de WebSearchAgent
    web_search_result: Dict[str, Any]

    # Contexte textuel Web injecté dans ClinicalAgent / CriticAgent
    web_context: str

    # Source réellement retenue : PubMed / arXiv / DOAJ / DuckDuckGo
    web_source: str


    # Diagnostic détaillé de la cascade WebSearch
    web_diagnostic: str

    # =========================================================
    # CLINICAL
    # =========================================================

    clinical_answer: str

    # =========================================================
    # CRITIC
    # =========================================================

    critic_result: Dict[str, Any]

    # Contrôle de la réponse après SynthesizerAgent
    final_critic_result: Dict[str, Any]

    # Mesures de confiance calculées par le modèle NVIDIA
    confidence_result: Dict[str, Any]

    # =========================================================
    # SYNTHESIS
    # =========================================================

    final_answer: str

    # Texte rejeté (non publié), conservé pour le diagnostic des rejets
    rejected_answer: str

    # =========================================================
    # PUBLICATION
    # =========================================================

    publication_status: str

    # Raisons déterministes d'un refus de publication
    publication_feedback: list[str]

    # =========================================================
    # WORKFLOW
    # =========================================================

    iterations: int

    # Test de la boucle IMPROVE
    test_force_improve: bool