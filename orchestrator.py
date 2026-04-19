import os
import json
import time
from datetime import datetime
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
USER_ID = os.getenv("USER_ID", "research_agent_user")
print("GROQ KEY:", os.getenv("GROQ_API_KEY")[:8] if os.getenv("GROQ_API_KEY") else "NOT FOUND")
# import all tools
from tools.arxiv_tool import search_arxiv
from tools.semantic_scholar_tool import search_semantic_scholar
from tools.tavily_tool import search_web
from tools.retrieval_tool import vectordb, store_papers, retrieve_context

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "research_db")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")

# setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

print("[OK] Groq LLM ready")
print("[OK] VectorDB ready")

# ── Memory helpers ──────────────────────────────────────────
def save_to_memory(query: str, synthesis: str):
    try:
        memories = []
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                memories = json.load(f)
        memories.append({
            "query": query,
            "summary": synthesis[:300],
            "timestamp": str(datetime.now())
        })
        with open(MEMORY_FILE, "w") as f:
            json.dump(memories, f, indent=2)
        print("[MEMORY] Saved ✅")
    except Exception as e:
        print(f"[MEMORY] Save failed: {e}")

def load_from_memory(query: str) -> str:
    try:
        if not os.path.exists(MEMORY_FILE):
            return "No prior research found."
        with open(MEMORY_FILE, "r") as f:
            memories = json.load(f)
        relevant = [
            m for m in memories
            if any(w in m["query"].lower()
                   for w in query.lower().split()[:3])
        ]
        if relevant:
            return "\n".join([
                f"- {m['query']}: {m['summary']}"
                for m in relevant[-3:]
            ])
        return "No prior research found."
    except:
        return "No prior research found."

# ── Agent State ─────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    task_plan: str
    memory_context: str
    search_results: List
    rag_context: List
    synthesis: str
    gaps: List
    iteration: int
    final_report: str

# ── Agent Nodes ─────────────────────────────────────────────
def memory_node(state: AgentState) -> AgentState:
    print("\n[MEMORY] Checking past sessions...")
    context = load_from_memory(state["query"])
    if "No prior" not in context:
        print("[MEMORY] Found past research ✅")
    else:
        print("[MEMORY] No past sessions found")
    state["memory_context"] = context
    return state

def orchestrator_node(state: AgentState) -> AgentState:
    print("\n[ORCHESTRATOR] Planning task...")
    prompt = f"""You are a research planning agent.
User query: {state['query']}
Past context: {state['memory_context']}

Create a brief 2-3 line search plan.
What keywords to search? What aspects to focus on?"""
    response = llm.invoke(prompt)
    state["task_plan"] = response.content
    print(f"[ORCHESTRATOR] Plan ready ✅")
    return state

def search_node(state: AgentState) -> AgentState:
    print("\n[SEARCH] Searching all sources...")
    query = state["query"]
    all_results = []

    # ArXiv
    try:
        arxiv_results = search_arxiv(query, max_results=5)
        print(f"[SEARCH] ArXiv: {len(arxiv_results)} papers")
        all_results.extend(arxiv_results)
    except Exception as e:
        print(f"[SEARCH] ArXiv failed: {e}")

    # Semantic Scholar
    try:
        time.sleep(5)
        ss_results = search_semantic_scholar(
            query,
            max_results=5,
            api_key=SEMANTIC_SCHOLAR_API_KEY or ""
        )
        print(f"[SEARCH] Semantic Scholar: {len(ss_results)} papers")
        all_results.extend(ss_results)
    except Exception as e:
        print(f"[SEARCH] Semantic Scholar failed: {e}")

    # Tavily web search
    try:
        web_results = search_web(
            query,
            max_results=3
        )
        print(f"[SEARCH] Web: {len(web_results)} results")
        all_results.extend(web_results)
    except Exception as e:
        print(f"[SEARCH] Tavily failed: {e}")

    # store academic papers in ChromaDB
    papers_only = [
        r for r in all_results
        if r.get("source") in ["arxiv", "semantic_scholar"]
    ]
    if papers_only:
        store_papers(papers_only) 

    state["search_results"] = all_results
    print(f"[SEARCH] Total: {len(all_results)} results")
    return state

def retrieval_node(state: AgentState) -> AgentState:
    print("\n[RETRIEVAL] Querying vector DB...")
    context = retrieve_context(state["query"], k=4)
    state["rag_context"] = context
    print(f"[RETRIEVAL] Got {len(context)} chunks")
    return state

def synthesis_node(state: AgentState) -> AgentState:
    print("\n[SYNTHESIS] Synthesizing findings...")
    papers_text = "\n".join([
        f"- {r.get('title','N/A')} "
        f"({r.get('year') or r.get('published','N/A')}): "
        f"{(r.get('abstract') or r.get('content',''))[:200]}"
        for r in state["search_results"][:8]
    ])
    rag_text = "\n".join([
        f"- {r['title']}: {r['content'][:200]}"
        for r in state["rag_context"]
    ])
    prompt = f"""You are a research synthesis expert.

Query: {state['query']}
Task Plan: {state['task_plan']}

Search Results:
{papers_text}

Related context from database:
{rag_text}

Synthesize key findings in 3-4 paragraphs.
Cover main themes, methodologies, and conclusions."""

    response = llm.invoke(prompt)
    state["synthesis"] = response.content
    print(f"[SYNTHESIS] Done — {len(response.content)} chars")
    return state

def critic_node(state: AgentState) -> AgentState:
    print(f"\n[CRITIC] Checking synthesis (iteration {state['iteration']})...")
    prompt = f"""You are a research critic.

Original query: {state['query']}
Current synthesis: {state['synthesis']}

Are there important gaps or missing aspects?
Reply ONLY with:
- "COMPLETE: <reason>" if synthesis is sufficient
- "GAPS: <list specific gaps>" if more research needed"""

    response = llm.invoke(prompt)
    content = response.content

    if "COMPLETE" in content or state["iteration"] >= 2:
        state["gaps"] = []
        print("[CRITIC] Synthesis approved ✅")
    else:
        gaps = content.replace("GAPS:", "").strip()
        state["gaps"] = [gaps]
        state["iteration"] = state["iteration"] + 1
        print(f"[CRITIC] Gaps found → re-search (iteration {state['iteration']})")
    return state

def report_node(state: AgentState) -> AgentState:
    print("\n[REPORT] Generating final report...")
    sources = "\n".join([
        f"- {r.get('title','N/A')} | "
        f"{r.get('pdf_url') or r.get('url','N/A')}"
        for r in state["search_results"][:8]
        if r.get('title')
    ])
    prompt = f"""You are a research report writer.

Query: {state['query']}
Synthesis: {state['synthesis']}
Sources:
{sources}

Write a structured research report with:
1. Executive Summary
2. Key Findings
3. Methodology Trends
4. Research Gaps
5. References

Be concise and professional."""

    response = llm.invoke(prompt)
    state["final_report"] = response.content
    save_to_memory(state["query"], state["synthesis"])
    return state

def should_retry(state: AgentState) -> str:
    if state["gaps"] and state["iteration"] < 3:
        return "search"
    return "report"

# ── Build Graph ──────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("memory", memory_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("search", search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "orchestrator")
    graph.add_edge("orchestrator", "search")
    graph.add_edge("search", "retrieval")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"search": "search", "report": "report"}
    )
    graph.add_edge("report", END)

    return graph.compile()

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()
    print("[OK] LangGraph compiled ✅")

    query = input("\nEnter research query: ").strip()
    if not query:
        query = "latest deep learning methods for cancer detection 2024"

    initial_state = AgentState(
        query=query,
        task_plan="",
        memory_context="",
        search_results=[],
        rag_context=[],
        synthesis="",
        gaps=[],
        iteration=0,
        final_report=""
    )

    print("\n" + "="*50)
    print("RESEARCH AGENT STARTING...")
    print("="*50)

    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT:")
    print("="*50)
    print(final_state["final_report"])

    # save report to file
    report_file = os.path.join(BASE_DIR, "last_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Query: {query}\n\n")
        f.write(final_state["final_report"])
    print(f"\n[OK] Report saved to: {report_file}")