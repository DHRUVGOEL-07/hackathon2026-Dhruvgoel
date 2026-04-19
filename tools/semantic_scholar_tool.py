import os
import time
import requests
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

_last_request_time = 0  # module-level throttle tracker

def search_semantic_scholar(query: str, max_results: int = 5, year_from: int = 2022, api_key: str = ""):
    global _last_request_time

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "fields": "title,abstract,year,externalIds,openAccessPdf,citationCount,authors",
        "limit": max_results * 3,
    }

    # Fix 2: use passed api_key param OR fall back to env var
    effective_key = api_key or SEMANTIC_SCHOLAR_API_KEY
    headers = {
        "User-Agent": "ResearchAgent/1.0",
    }
    if effective_key:
        headers["x-api-key"] = effective_key

    max_retries = 3
    data = {}

    for attempt in range(max_retries):
        # ── Rate limiter: enforce 1 req/sec ──────────────────
        elapsed = time.time() - _last_request_time
        if elapsed < 1.1:  # slight buffer above 1s
            time.sleep(1.1 - elapsed)

        try:
            _last_request_time = time.time()
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"[WARN] Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            break

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Failed after {max_retries} attempts: {e}")
                return []
            time.sleep(5)

    papers = data.get("data", [])
    print(f"[DEBUG] Raw results from API: {len(papers)}")

    results = []
    for paper in papers:
        year = paper.get("year") or 0
        if year < year_from:
            continue

        oa = paper.get("openAccessPdf")
        pdf_url = oa.get("url") if oa else None

        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI")

        authors = paper.get("authors", [])
        author_names = ", ".join([a["name"] for a in authors[:3]])
        if len(authors) > 3:
            author_names += " et al."

        results.append({
            "title": paper.get("title", "N/A"),
            "abstract": (paper.get("abstract") or "")[:300],
            "year": year,
            "doi": doi,
            "pdf_url": pdf_url,
            "citations": paper.get("citationCount", 0),
            "authors": author_names,
            "source": "semantic_scholar"
        })

        if len(results) >= max_results:
            break

    print(f"[DEBUG] Filtered results (>= {year_from}): {len(results)}")
    print(f"[DEBUG] Years: {sorted(set(r['year'] for r in results), reverse=True)}")
    return results
# ── LangChain tool wrapper ───────────────────────────────────────
@tool
def semantic_scholar_tool(query: str) -> str:
    """Search Semantic Scholar for research papers.
    Returns titles, abstracts, citations, authors and free PDF links."""
    results = search_semantic_scholar(query)
    return str(results)


# ── Main: test pipeline ──────────────────────────────────────────
def main():
    results = search_semantic_scholar(
        query="cancer detection deep learning",
        max_results=5,
        year_from=2022
    )

    for i, paper in enumerate(results):
        print(f"\n--- Paper {i+1} ---")
        print(f"Title    : {paper['title']}")
        print(f"Year     : {paper['year']}")
        print(f"Authors  : {paper['authors']}")
        print(f"Citations: {paper['citations']}")
        print(f"PDF      : {paper['pdf_url']}")
        print(f"Abstract : {paper['abstract'][:200]}...")


if __name__ == "__main__":
    main()