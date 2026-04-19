import os
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=TAVILY_API_KEY)


# ── Core search function ─────────────────────────────────────────
def search_web(query: str, max_results: int = 5):
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False
        )

        results = []

        if response.get("answer"):
            print(f"[TAVILY ANSWER] {response['answer']}\n")

        for r in response.get("results", []):
            results.append({
                "title": r.get("title", "N/A"),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:300],
                "score": round(r.get("score", 0), 3)
            })

        print(f"[DEBUG] Web results found: {len(results)}")
        return results

    except Exception as e:
        print(f"[ERROR] Tavily search failed: {e}")
        return []


# ── LangChain tool wrapper ───────────────────────────────────────
@tool
def web_search_tool(query: str) -> str:
    """Search the web using Tavily for recent news, blogs,
    and general information not found in academic databases."""
    results = search_web(query)
    return str(results)


# ── Main: test pipeline ──────────────────────────────────────────
def main():
    results = search_web(
        query="latest deep learning cancer detection 2025",
        max_results=5
    )

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title  : {r['title']}")
        print(f"URL    : {r['url']}")
        print(f"Score  : {r['score']}")
        print(f"Content: {r['content']}")


if __name__ == "__main__":
    main()