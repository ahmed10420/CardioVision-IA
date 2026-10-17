from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_reference_chunk(text: str) -> bool:
    """
    Détecte les chunks de références bibliographiques.
    """

    text_lower = text.lower()

    reference_indicators = [
        "doi.org",
        "https:/",
        "http:/",
        "et al.",
        "eur heart j",
        "circulation",
        "stroke",
        "europace",
        "j am coll cardiol",
        "references",
        "downloaded from",
    ]

    count = sum(1 for pattern in reference_indicators if pattern in text_lower)

    # Beaucoup de DOI / citations = mauvais chunk
    if count >= 2:
        return True

    # Beaucoup d'années de citations = souvent bibliographie
    years_count = sum(text_lower.count(str(year)) for year in range(1990, 2027))
    if years_count >= 5:
        return True

    return False


def is_useful_chunk(text: str) -> bool:
    """
    Garde seulement les chunks utiles médicalement.
    """

    text_lower = text.lower()

    if len(text.strip()) < 400:
        return False

    if is_reference_chunk(text):
        return False

    useless_patterns = [
        "all rights reserved",
        "permission",
        "copyright",
        "downloaded from",
        "author information",
        "document reviewers",
    ]

    if any(pattern in text_lower for pattern in useless_patterns):
        return False

    return True


def split_documents(documents):
    """
    Découpe les documents puis supprime les chunks inutiles.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
    )

    chunks = splitter.split_documents(documents)

    useful_chunks = []

    for chunk in chunks:
        if is_useful_chunk(chunk.page_content):
            useful_chunks.append(chunk)

    print(f"Nombre de chunks créés : {len(chunks)}")
    print(f"Nombre de chunks utiles gardés : {len(useful_chunks)}")

    return useful_chunks


if __name__ == "__main__":
    print("Chunking avec filtrage renforcé.")