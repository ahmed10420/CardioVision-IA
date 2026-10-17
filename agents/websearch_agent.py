import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests


class WebSearchAgent:
    """
    WebSearchAgent inspiré de l'architecture CardioVision IA.

    Cascade "first-match-wins" :
        PubMed -> arXiv -> DOAJ -> DuckDuckGo

    Rôle :
    - compléter le RAG local avec de la littérature externe ;
    - privilégier les sources académiques ;
    - conserver la traçabilité (source, titre, date, URL, identifiant) ;
    - ne jamais remplacer les guidelines ESC/AHA locales.

    IMPORTANT :
    Cet agent ne "prouve" pas qu'une information est vraie.
    Il applique seulement des filtres de qualité/traçabilité.
    La validation clinique finale reste du ressort du ClinicalAgent/CriticAgent.
    """

    PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ARXIV_BASE = "https://export.arxiv.org/api/query"
    DOAJ_BASE = "https://doaj.org/api/search/articles"

    def __init__(
        self,
        timeout: int = 15,
        max_results: int = 5,
        enable_duckduckgo: bool = True,
    ):
        self.name = "WebSearchAgent"
        self.timeout = timeout
        self.max_results = max_results
        self.enable_duckduckgo = enable_duckduckgo

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "CardioVision-ECG-Agent/1.0 "
                "(academic research project)"
            )
        })

    # ============================================================
    # OUTILS COMMUNS
    # ============================================================

    @staticmethod
    def _build_search_query(question: str) -> str:
        """
        Transforme une question utilisateur longue en requête de recherche
        plus courte et plus adaptée aux moteurs académiques.

        Pas de LLM ici : uniquement règles déterministes simples.
        """
        q = str(question or "").lower()

        replacements = {
            "fibrillation atriale": "atrial fibrillation",
            "détection": "detection",
            "detecter": "detection",
            "détecter": "detection",
            "electrocardiogramme": "ECG",
            "électrocardiogramme": "ECG",
            "recommandations esc": "ESC",
            "critères diagnostiques": "diagnosis criteria",
            "méthodes récentes": "recent methods",
            "méthodes": "methods",
        }

        for src, dst in replacements.items():
            q = q.replace(src, dst)

        # Garde seulement les termes les plus utiles pour la recherche.
        useful_tokens = []
        stopwords = {
            "quelles", "quels", "sont", "les", "des", "pour", "sur",
            "comment", "ces", "se", "aux", "et", "existe", "il", "des",
            "cas", "similaires", "dans", "avec", "une", "un", "de", "du",
            "la", "le", "en", "au", "à", "a", "the", "what", "how", "are",
        }

        for token in re.findall(r"[a-zA-ZÀ-ÿ0-9\-]+", q):
            if token not in stopwords:
                useful_tokens.append(token)

        # Evite les requêtes énormes qui donnent souvent zéro résultat.
        return " ".join(useful_tokens[:12])

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @staticmethod
    def _year_from_text(value: str) -> Optional[int]:
        if not value:
            return None

        for token in value.replace("/", " ").replace("-", " ").split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if 1900 <= year <= 2100:
                    return year

        return None

    @staticmethod
    def _deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []

        for item in results:
            key = (
                item.get("title", "")
                .strip()
                .lower()
            )

            if not key or key in seen:
                continue

            seen.add(key)
            unique.append(item)

        return unique

    @staticmethod
    def _quality_label(provider: str) -> str:
        """
        Etiquette de qualité documentaire.
        Ce n'est PAS une probabilité de vérité.
        """
        labels = {
            "PubMed": "priorité élevée",
            "arXiv": "prépublication / priorité intermédiaire",
            "DOAJ": "revue open access / priorité intermédiaire",
            "DuckDuckGo": "fallback web / priorité faible",
        }

        return labels.get(provider, "non classé")

    def _validate_result(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Vérifications simples de traçabilité.

        On vérifie :
        - titre non vide
        - URL non vide
        - source connue
        - date/année disponible si possible
        - identifiant académique quand disponible

        Ce contrôle ne valide PAS la vérité clinique du contenu.
        """

        issues = []

        title = self._clean_text(item.get("title"))
        url = self._clean_text(item.get("url"))
        provider = self._clean_text(item.get("provider"))
        identifier = self._clean_text(item.get("id"))
        date_text = self._clean_text(item.get("year_or_date"))

        if not title:
            issues.append("titre_manquant")

        if not url:
            issues.append("url_manquante")

        if provider not in {
            "PubMed",
            "arXiv",
            "DOAJ",
            "DuckDuckGo",
        }:
            issues.append("source_inconnue")

        if provider in {"PubMed", "arXiv"} and not identifier:
            issues.append("identifiant_manquant")

        if not date_text:
            issues.append("date_manquante")

        traceability_ok = len(issues) == 0

        return {
            "traceability_ok": traceability_ok,
            "issues": issues,
            "quality_label": self._quality_label(provider),
        }

    # ============================================================
    # PUBMED
    # ============================================================

    def search_pubmed(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        search_query = self._build_search_query(query)

        params = {
            "db": "pubmed",
            "term": search_query,
            "retmode": "json",
            "retmax": self.max_results,
            "sort": "relevance",
        }

        email = os.getenv("NCBI_EMAIL")
        api_key = os.getenv("NCBI_API_KEY")

        if email:
            params["email"] = email

        if api_key:
            params["api_key"] = api_key

        response = self.session.get(
            f"{self.PUBMED_BASE}/esearch.fcgi",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        ids = (
            response.json()
            .get("esearchresult", {})
            .get("idlist", [])
        )

        if not ids:
            return []

        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }

        if email:
            summary_params["email"] = email

        if api_key:
            summary_params["api_key"] = api_key

        summary_response = self.session.get(
            f"{self.PUBMED_BASE}/esummary.fcgi",
            params=summary_params,
            timeout=self.timeout,
        )
        summary_response.raise_for_status()

        payload = summary_response.json()
        block = payload.get("result", {})

        results = []

        for pmid in ids:
            item = block.get(pmid, {})

            title = self._clean_text(
                item.get("title")
            )

            date_text = self._clean_text(
                item.get("pubdate")
            )

            source = self._clean_text(
                item.get("source")
            )

            if not title:
                continue

            result = {
                "provider": "PubMed",
                "id": str(pmid),
                "title": title,
                "journal": source,
                "year_or_date": date_text,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "snippet": "",
            }

            result["validation"] = self._validate_result(
                result
            )

            results.append(result)

        return results

    # ============================================================
    # ARXIV
    # ============================================================

    def search_arxiv(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        params = {
            "search_query": f"all:{self._build_search_query(query)}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
        }

        response = self.session.get(
            self.ARXIV_BASE,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)

        ns = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        results = []

        for entry in root.findall(
            "atom:entry",
            ns,
        ):
            title_node = entry.find(
                "atom:title",
                ns,
            )
            summary_node = entry.find(
                "atom:summary",
                ns,
            )
            id_node = entry.find(
                "atom:id",
                ns,
            )
            published_node = entry.find(
                "atom:published",
                ns,
            )

            title = (
                self._clean_text(title_node.text)
                if title_node is not None
                else ""
            )

            summary = (
                self._clean_text(summary_node.text)
                if summary_node is not None
                else ""
            )

            url = (
                self._clean_text(id_node.text)
                if id_node is not None
                else ""
            )

            published = (
                self._clean_text(
                    published_node.text
                )[:10]
                if published_node is not None
                else ""
            )

            if not title:
                continue

            arxiv_id = (
                url.rsplit("/", 1)[-1]
                if url
                else ""
            )

            result = {
                "provider": "arXiv",
                "id": arxiv_id,
                "title": title,
                "journal": "arXiv preprint",
                "year_or_date": published,
                "url": url,
                "snippet": summary[:800],
            }

            result["validation"] = self._validate_result(
                result
            )

            results.append(result)

        return results

    # ============================================================
    # DOAJ
    # ============================================================

    def search_doaj(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        search_query = self._build_search_query(query)

        # DOAJ attend une requête de recherche structurée.
        doaj_query = (
            f'bibjson.title:"{search_query}" OR '
            f'bibjson.abstract:"{search_query}"'
        )

        safe_query = requests.utils.quote(
            doaj_query,
            safe=""
        )

        url = f"{self.DOAJ_BASE}/{safe_query}"

        response = self.session.get(
            url,
            params={
                "pageSize": self.max_results
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()

        results = []

        for item in payload.get(
            "results",
            [],
        )[: self.max_results]:

            bib = item.get(
                "bibjson",
                {},
            )

            title = self._clean_text(
                bib.get("title")
            )

            abstract = self._clean_text(
                bib.get("abstract")
            )

            year = self._clean_text(
                bib.get("year")
            )

            journal = self._clean_text(
                (
                    bib.get("journal")
                    or {}
                ).get("title")
            )

            links = (
                bib.get("link")
                or []
            )

            article_url = ""

            for link in links:
                candidate = self._clean_text(
                    link.get("url")
                )
                if candidate:
                    article_url = candidate
                    break

            if not title:
                continue

            result = {
                "provider": "DOAJ",
                "id": self._clean_text(
                    item.get("id")
                ),
                "title": title,
                "journal": journal,
                "year_or_date": year,
                "url": article_url,
                "snippet": abstract[:800],
            }

            result["validation"] = self._validate_result(
                result
            )

            results.append(result)

        return results

    # ============================================================
    # DUCKDUCKGO — FALLBACK UNIQUEMENT
    # ============================================================

    def search_duckduckgo(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        if not self.enable_duckduckgo:
            return []

        try:
            from ddgs import DDGS
        except ImportError:
            raise RuntimeError(
                "Le package 'ddgs' n'est pas installé. "
                "Installe-le avec : pip install ddgs"
            )

        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                self._build_search_query(query),
                max_results=self.max_results,
            )

            for item in search_results:
                title = self._clean_text(
                    item.get("title")
                )
                url = self._clean_text(
                    item.get("href")
                )
                body = self._clean_text(
                    item.get("body")
                )

                if not title:
                    continue

                result = {
                    "provider": "DuckDuckGo",
                    "id": "",
                    "title": title,
                    "journal": "",
                    "year_or_date": "",
                    "url": url,
                    "snippet": body[:800],
                }

                result["validation"] = self._validate_result(
                    result
                )

                results.append(result)

        return results

    # ============================================================
    # FORMATAGE POUR LE CLINICALAGENT
    # ============================================================

    @staticmethod
    def format_context(
        results: List[Dict[str, Any]],
    ) -> str:

        if not results:
            return (
                "Aucune source Web externe "
                "pertinente n'a été retrouvée."
            )

        blocks = []

        for index, item in enumerate(
            results,
            start=1,
        ):
            validation = item.get(
                "validation",
                {},
            )

            block = [
                f"PASSAGE WEB {index}",
                f"Source : {item.get('provider', '')}",
                f"Identifiant : {item.get('id', '')}",
                f"Titre : {item.get('title', '')}",
                f"Journal : {item.get('journal', '')}",
                f"Date : {item.get('year_or_date', '')}",
                f"URL : {item.get('url', '')}",
                (
                    "Qualité documentaire : "
                    f"{validation.get('quality_label', '')}"
                ),
                (
                    "Traçabilité : "
                    + (
                        "OK"
                        if validation.get("traceability_ok")
                        else "INCOMPLÈTE"
                    )
                ),
            ]

            snippet = item.get(
                "snippet",
                "",
            )

            if snippet:
                block.append(
                    f"Résumé : {snippet}"
                )

            blocks.append(
                "\n".join(block)
            )

        return "\n\n".join(blocks)

    # ============================================================
    # CASCADE PRINCIPALE
    # ============================================================

    def run(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Cascade du papier :
        PubMed -> arXiv -> DOAJ -> DuckDuckGo

        Arrêt dès qu'une source retourne au moins un résultat.
        """

        attempts = []

        cascade = [
            ("PubMed", self.search_pubmed),
            ("arXiv", self.search_arxiv),
            ("DOAJ", self.search_doaj),
        ]

        if self.enable_duckduckgo:
            cascade.append(
                (
                    "DuckDuckGo",
                    self.search_duckduckgo,
                )
            )

        selected_provider = None
        selected_results = []

        for provider_name, search_fn in cascade:
            try:
                results = search_fn(
                    question
                )

                results = self._deduplicate(
                    results
                )

                attempts.append({
                    "provider": provider_name,
                    "status": (
                        "success"
                        if results
                        else "empty"
                    ),
                    "count": len(results),
                    "error": None,
                })

                if results:
                    selected_provider = (
                        provider_name
                    )
                    selected_results = results
                    break

            except Exception as exc:
                attempts.append({
                    "provider": provider_name,
                    "status": "error",
                    "count": 0,
                    "error": str(exc),
                })

            time.sleep(0.15)

        status = (
            "success"
            if selected_results
            else "no_results"
        )

        return {
            "agent": self.name,
            "status": status,
            "query": question,
            "search_query": self._build_search_query(question),
            "selected_provider": selected_provider,
            "results": selected_results,
            "context": self.format_context(
                selected_results
            ),
            "attempts": attempts,
            "result_count": len(selected_results),
            "usage_rule": (
                "Les résultats Web sont complémentaires. "
                "Les guidelines ESC/AHA locales restent "
                "la source normative prioritaire. "
                "La traçabilité d'une source ne garantit pas "
                "la vérité clinique de son contenu."
            ),
        }


if __name__ == "__main__":
    agent = WebSearchAgent()

    result = agent.run(
        "atrial fibrillation ECG diagnosis"
    )

    print("\nStatus :", result["status"])
    print(
        "Source retenue :",
        result["selected_provider"],
    )

    print("\nTentatives :")
    for attempt in result["attempts"]:
        print(attempt)

    print("\nContexte Web :")
    print(result["context"])
