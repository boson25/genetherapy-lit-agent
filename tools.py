"""PubMed retrieval tool for the gene-therapy literature agent.

Uses NCBI E-utilities (free, no key required):
  1. esearch: search query -> list of PMIDs
  2. efetch:  list of PMIDs -> titles + abstracts (XML)
"""

import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# NCBI allows 3 requests/sec w/o a key, 10 requests/sec with one key.
REQUEST_DELAY = 0.12 if NCBI_API_KEY else 0.35
MAX_RESULTS = 10
MAX_ABSTRACT_CHARS = 2000

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------
def _common_params() -> dict:
    """Identification params NCBI asks every client to send."""
    params = {"tool": "genetherapy-lit-agent"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _get(endpoint: str, params: dict) -> requests.Response:
    """GET an E-utilities endpoint, respecting NCBI's rate limit."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(
        f"{EUTILS_BASE}/{endpoint}",
        params={**_common_params(), **params},
        timeout=20,
    )
    resp.raise_for_status()  # turns HTTP errors (e.g., 429, 500) into exceptions
    return resp

# ---------------------------------------------------------------------------
# Step 1: query -> PMIDs
# ---------------------------------------------------------------------------
def _esearch(query: str, n: int) -> list[str]:
    resp = _get(
        "esearch.fcgi",
        {"db": "pubmed", "term": query, "retmax": n,
         "retmode": "json", "sort": "relevance"},
    )
    return resp.json()["esearchresult"]["idlist"]


# ---------------------------------------------------------------------------
# Step 2: PMIDs -> parsed records
# ---------------------------------------------------------------------------
def _text(elem: ET.Element, path: str) -> str:
    """Full text of the first matching node, including inline tags like <i>."""
    node = elem.find(path)
    return "".join(node.itertext()).strip() if node is not None else ""


def _parse_article(article: ET.Element) -> dict:
    """Extract the fields we need from one <PubmedArticle> element."""
    # Structured abstracts have several sections (BACKGROUND, METHODS, ...).
    sections = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = "".join(node.itertext()).strip()
        if text:
            label = node.get("Label")
            sections.append(f"{label}: {text}" if label else text)
    abstract = " ".join(sections)[:MAX_ABSTRACT_CHARS]

    year = (_text(article, ".//JournalIssue/PubDate/Year")
            or _text(article, ".//JournalIssue/PubDate/MedlineDate")[:4])

    return {
        "pmid": _text(article, ".//MedlineCitation/PMID"),
        "title": _text(article, ".//ArticleTitle"),
        "journal": _text(article, ".//Journal/Title"),
        "year": year,
        "abstract": abstract or "(no abstract available)",
    }


def _efetch(pmids: list[str]) -> list[dict]:
    resp = _get(
        "efetch.fcgi",
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )
    root = ET.fromstring(resp.content)
    records = {r["pmid"]: r for r in map(_parse_article, root.findall(".//PubmedArticle"))}
    # Keep esearch's relevance order.
    return [records[p] for p in pmids if p in records]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def search_pubmed(query: str, n: int = 5) -> list[dict]:
    """Search PubMed and return up to n records with title and abstract.

    Returns [] if nothing matches. Raises requests.RequestException on
    network/HTTP errors (the agent loop will catch these).
    """
    n = max(1, min(int(n), MAX_RESULTS))
    pmids = _esearch(query, n)
    if not pmids:
        return []
    return _efetch(pmids)


def format_results(records: list[dict]) -> str:
    """Render records as plain text for the LLM to read and cite."""
    if not records:
        return "No PubMed results found for this query."
    blocks = [
        f"[PMID {r['pmid']}] {r['title']} ({r['journal']}, {r['year']})\n{r['abstract']}"
        for r in records
    ]
    return "\n\n".join(blocks)


# Tool definition Claude will see (used by agent.py later).
SEARCH_PUBMED_TOOL = {
    "name": "search_pubmed",
    "description": (
        "Search PubMed for biomedical literature. Returns titles and abstracts "
        "with PMIDs. Use focused keyword queries (e.g., 'AAV hemophilia A "
        "factor VIII liver'), not full sentences. Call again with a refined "
        "query if results are off-topic."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "PubMed search query."},
            "n": {"type": "integer", "description": "Number of results (1-10).",
                  "default": 5},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Manual test: python tools.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = search_pubmed("AAV gene therapy hemophilia A", n=3)
    print(f"Got {len(results)} results\n")
    for r in results:
        print(f"PMID {r['pmid']} | {r['year']} | {r['journal']}")
        print(f"  {r['title']}")
        print(f"  {r['abstract'][:200]}...\n")