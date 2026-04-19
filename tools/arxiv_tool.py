import arxiv
import requests
from langchain.tools import tool
import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

# load from root .env file
from dotenv import load_dotenv
import os

load_dotenv()  # auto load from root

# ── Step 3: check raw values ────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
USER_ID        = os.getenv("USER_ID", "research_agent_user")

print("GROQ KEY:", os.getenv("GROQ_API_KEY")[:8] if os.getenv("GROQ_API_KEY") else "NOT FOUND")

def search_arxiv(query: str, max_results: int = 5, year_from: int = 2022):
    client = arxiv.Client(
        page_size=100,
        num_retries=5,
        delay_seconds=3
    )

    stopwords = {"latest", "recent", "new", "for", "using", "with", 
             "and", "the", "of", "in", "a", "an", "methods", 
             "based", "approach", "using"}
    terms = [w for w in query.strip().split() if w.lower() not in stopwords]
    if not terms:
        arxiv_query = query
    else:
        field_parts = " AND ".join([f'(ti:{t} OR abs:{t})' for t in terms])
        arxiv_query = field_parts

    print(f"[DEBUG] Query sent to ArXiv: {arxiv_query}")

    search = arxiv.Search(
        query=arxiv_query,
        max_results=300,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    seen = set()
    for paper in client.results(search):
        if paper.entry_id in seen:
            continue
        seen.add(paper.entry_id)

        if paper.published.year >= year_from:
            results.append({
                "title": paper.title,
                "abstract": paper.summary[:300],
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "published": str(paper.published.date()),
                "year": paper.published.year
            })
            if len(results) >= max_results:
                break

    print(f"[DEBUG] Total collected: {len(results)}")
    print(f"[DEBUG] Years found: {sorted(set(r['year'] for r in results), reverse=True)}")
    return results


def get_free_pdf(doi: str, email: str = "test@gmail.com"):
    """Unpaywall helper — finds a free/open-access PDF for a given DOI."""
    if not doi:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("is_oa"):
            loc = data.get("best_oa_location", {})
            return loc.get("url_for_pdf")
    except Exception:
        return None
    return None


@tool
def arxiv_search_tool(query: str) -> str:
    """Search ArXiv for research papers on a given topic.
    Returns titles, abstracts, PDF links and DOIs."""
    results = search_arxiv(query)

    # Try to find free PDFs for paywalled papers
    for paper in results:
        if not paper["pdf_url"] and paper["doi"]:
            paper["free_pdf"] = get_free_pdf(paper["doi"])

    return str(results)


def main():
    results = search_arxiv(
        query="cancer detection deep learning",
        max_results=7,
        year_from=2020
    )

    for i, paper in enumerate(results):
        print(f"\n--- Paper {i+1} ---")
        print(f"Title    : {paper['title']}")
        print(f"Published: {paper['published']}")
        print(f"PDF      : {paper['pdf_url']}")
        print(f"Abstract : {paper['abstract'][:200]}...")


if __name__ == "__main__":
    main()