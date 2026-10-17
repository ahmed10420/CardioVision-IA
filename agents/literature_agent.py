"""
LiteratureAgent - version compatible Streamlit

Principe :
- aucun import lourd de retriever n'est exécuté au chargement du module ;
- les retrievers sont importés uniquement au moment de run() ;
- cela évite que Streamlit plante pendant l'import de LiteratureAgent
  lorsque sentence_transformers / transformers sont chargés transitivement.
"""


def truncate_context(text, max_chars=5000):
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[contexte tronqué]..."


class LiteratureAgent:
    """Récupère les quatre familles de preuves du pipeline."""

    def __init__(
        self,
        guideline_k=4,
        ecg_reference_k=2,
        research_k=2,
        ptbxl_k=2,
        enable_ecg_reference=True,
        enable_research=True,
        enable_ptbxl=True,
    ):
        self.guideline_k = guideline_k
        self.ecg_reference_k = ecg_reference_k
        self.research_k = research_k
        self.ptbxl_k = ptbxl_k

        self.enable_ecg_reference = enable_ecg_reference
        self.enable_research = enable_research
        self.enable_ptbxl = enable_ptbxl

    def run(self, question):
        """
        Exécute le retrieval.

        IMPORTANT :
        tous les imports susceptibles de charger sentence_transformers,
        transformers, chromadb ou d'autres dépendances lourdes sont faits ici
        et non au niveau global du module.
        """

        # ====================================================
        # 1. GUIDELINES ESC/AHA
        # ====================================================
        try:
            from rag.retriever import retrieve_documents
            from rag.rag_with_llm import format_context

            guideline_docs = retrieve_documents(
                question,
                k=self.guideline_k,
            )

            guideline_context = truncate_context(
                format_context(guideline_docs),
                6500,
            )

        except Exception as exc:
            guideline_docs = []
            guideline_context = (
                "Guidelines indisponibles pour cette requête : "
                f"{exc}"
            )

        # ====================================================
        # 2. ECG REFERENCE
        # ====================================================
        ecg_reference_documents = []
        ecg_reference_context = ""

        if self.enable_ecg_reference:
            try:
                from rag_ecg_reference.retriever import (
                    retrieve_ecg_reference,
                    build_context as build_ecg_reference_context,
                )

                ecg_reference_documents = retrieve_ecg_reference(
                    question,
                    k=self.ecg_reference_k,
                )

                ecg_reference_context = truncate_context(
                    build_ecg_reference_context(
                        ecg_reference_documents
                    ),
                    2600,
                )

            except Exception as exc:
                ecg_reference_context = (
                    "ECG reference indisponible pour cette requête : "
                    f"{exc}"
                )

        # ====================================================
        # 3. RESEARCH LOCAL
        # ====================================================
        research_documents = []
        research_context = ""

        if self.enable_research:
            try:
                from rag_research.retriever import (
                    retrieve_research,
                    build_context as build_research_context,
                )

                research_documents = retrieve_research(
                    question,
                    k=self.research_k,
                )

                research_context = truncate_context(
                    build_research_context(
                        research_documents
                    ),
                    2600,
                )

            except Exception as exc:
                research_context = (
                    "Research local indisponible pour cette requête : "
                    f"{exc}"
                )

        # ====================================================
        # 4. PTB-XL
        # ====================================================
        ptbxl_documents = []
        ptbxl_context = ""
        ptbxl_case_ids = []

        if self.enable_ptbxl:
            try:
                from rag_ptbxl.ptbxl_retriever import search_ptbxl

                ptbxl_result = search_ptbxl(
                    query=question,
                    k=self.ptbxl_k,
                )

                ptbxl_documents = ptbxl_result.get(
                    "documents",
                    [],
                )

                ptbxl_context = truncate_context(
                    ptbxl_result.get(
                        "context",
                        "",
                    ),
                    2200,
                )

                for doc in ptbxl_documents[:self.ptbxl_k]:
                    metadata = getattr(
                        doc,
                        "metadata",
                        {},
                    ) or {}

                    case_id = metadata.get(
                        "ecg_id"
                    )

                    if case_id is not None:
                        ptbxl_case_ids.append(
                            case_id
                        )

            except Exception as exc:
                ptbxl_context = (
                    "PTB-XL indisponible pour cette requête : "
                    f"{exc}"
                )

        # ====================================================
        # OUTPUT
        # ====================================================
        return {
            "agent": "LiteratureAgent",

            # Compatibilité avec l'ancienne sortie.
            "documents": guideline_docs,
            "context": guideline_context,

            # Guidelines.
            "guideline_documents": guideline_docs,
            "guideline_context": guideline_context,

            # Références ECG.
            "ecg_reference_documents": ecg_reference_documents,
            "ecg_reference_context": ecg_reference_context,

            # Recherche locale.
            "research_documents": research_documents,
            "research_context": research_context,

            # PTB-XL.
            "ptbxl_documents": ptbxl_documents,
            "ptbxl_context": ptbxl_context,
            "ptbxl_case_ids": ptbxl_case_ids,
        }
