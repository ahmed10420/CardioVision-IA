"""Utilitaires communs pour les métadonnées issues de PyPDFLoader."""


def use_one_based_page_number(metadata: dict) -> None:
    """Convertit en place l'index de page PyPDFLoader en numéro humain."""

    page_index = metadata.get("page")
    if isinstance(page_index, int) and not isinstance(page_index, bool):
        metadata["page"] = page_index + 1
