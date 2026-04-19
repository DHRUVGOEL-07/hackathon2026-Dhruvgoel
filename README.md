# Research System Agent
An autonomous multi-agent research assistant built with LangGraph that searches academic papers, synthesizes findings, and generates structured research reports — completely automated.
---
## What It Does
You type a research query. The agent does everything else:

- Searches ArXiv, Semantic Scholar, and the web simultaneously
- Stores and retrieves papers from a local vector database (ChromaDB)
- Synthesizes all findings using an LLM
- Self-critiques and re-searches if gaps are found
- Generates a structured research report with citations
- Remembers past research sessions for future queries
---
## Architecture
```
User Query
    ↓
Memory Agent        ← recalls past sessions
    ↓
Orchestrator        ← plans the research task
    ↓
Search Agent        ← ArXiv + Semantic Scholar + Tavily Web
    ↓
Retrieval Agent     ← ChromaDB vector similarity search
    ↓
Synthesis Agent     ← LLM combines all findings
    ↓
Critic Agent        ← checks for gaps, re-triggers search if needed
    ↓
Report Generator    ← structured report with citations
    ↓
Final Report (saved to last_report.txt)
```
---
## Project Structure
```
Research_System_Agent/
├── .env                          # API keys
├── orchestrator.py               # Main pipeline — run this
├── last_report.txt               # Latest generated report
├── memory.json                   # Persistent memory across sessions
├── research_db/                  # ChromaDB vector database
│   └── chroma.sqlite3
└── tools/
    ├── arxiv_tool.py             # ArXiv API search
    ├── semantic_scholar_tool.py  # Semantic Scholar API search
    ├── tavily_tool.py            # Tavily web search
    └── retrieval_tool.py         # ChromaDB store + retrieve
```
---
## Tech Stack
| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq (Llama 3.3 70B) — free |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` — free |
| Vector Database | ChromaDB |
| Academic Search | ArXiv API + Semantic Scholar API |
| Web Search | Tavily API |
| Memory | Custom JSON-based persistent memory |
| Environment | Python 3.12 + virtualenv |
---
## Setup

### 1. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install langchain langchain-groq langchain-chroma langchain-huggingface
pip install langgraph chromadb sentence-transformers
pip install arxiv requests tavily-python python-dotenv
pip install langchain-text-splitters langchain-community
```
### 3. Get API Keys (all free)
| Key | Where to get |
|---|---|
| `GROQ_API_KEY` | console.groq.com |
| `TAVILY_API_KEY` | tavily.com |
| `SEMANTIC_SCHOLAR_API_KEY` | semanticscholar.org/product/api |

### 4. Create `.env` file in project root
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
USER_ID=research_agent_user
```

### 5. Run the agent
```bash
python orchestrator.py
```
---

## Usage

```bash
python orchestrator.py

Enter research query: latest deep learning methods for cancer detection
```

The agent will run through all stages automatically and print the final report. The report is also saved to `last_report.txt`.

### Example Output
```
[MEMORY] Found past research ✅
[ORCHESTRATOR] Plan ready ✅
[SEARCH] ArXiv: 5 papers
[SEARCH] Semantic Scholar: 5 papers
[SEARCH] Web: 3 results
[RETRIEVAL] Got 4 chunks from DB
[SYNTHESIS] Done — 3107 chars
[CRITIC] Gaps found → re-search (iteration 1)
[CRITIC] Synthesis approved ✅
[REPORT] Done ✅
[MEMORY] Saved ✅
```

---

## Tools

### `arxiv_tool.py`
- Searches ArXiv using field-specific queries (`ti:` and `abs:`)
- Filters by publication year
- Returns title, abstract, PDF URL, DOI

### `semantic_scholar_tool.py`
- Searches Semantic Scholar's 200M+ paper database
- Returns citation count, authors, open-access PDFs
- Built-in rate limit handling with retry logic

### `tavily_tool.py`
- Real-time web search for recent news and blogs
- Returns relevance scores for each result
- Covers content not yet on academic databases

### `retrieval_tool.py`
- Stores paper chunks in ChromaDB using HuggingFace embeddings
- Semantic similarity search for relevant context
- Persists data locally across sessions

---

## Agent Nodes (LangGraph)

| Node | Role |
|---|---|
| `memory_node` | Loads relevant past research sessions |
| `orchestrator_node` | Creates a search plan using LLM |
| `search_node` | Runs all 3 search tools in sequence |
| `retrieval_node` | Queries ChromaDB for related context |
| `synthesis_node` | Combines all sources into a synthesis |
| `critic_node` | Identifies gaps and triggers re-search |
| `report_node` | Generates final structured report |

---

## How the Critic Loop Works

```python
Synthesis → Critic checks for gaps
    ├── Gaps found + iteration < 3 → re-trigger Search Agent
    └── No gaps OR iteration >= 3 → proceed to Report
```

This gives the agent **self-correction capability** — it automatically searches for missing information before generating the final report.

---

## Memory System

The agent uses a JSON-based persistent memory stored in `memory.json`:

- **Saves** the query and synthesis summary after every run
- **Recalls** relevant past sessions at the start of each new query
- Matches by keyword overlap between new query and past queries
- Stores last 3 relevant sessions as context

---

## Report Structure

Every generated report follows this structure:

```
1. Executive Summary
2. Key Findings
3. Methodology Trends
4. Research Gaps
5. References (with URLs)
```

---

## Environment Notes

- All heavy computation runs on **Groq's servers** (LLM) and **HuggingFace's servers** (embeddings)
- No GPU required — runs on any CPU machine
- Tested on i3 8th Gen with 8GB RAM
- Also runs on Google Colab via VS Code Colab extension

---

## Known Limitations

- Semantic Scholar may rate-limit without an API key
- ArXiv search quality depends on query keyword specificity
- Memory matching is keyword-based, not semantic
- Report references section may need manual formatting

---

## Future Improvements

- [ ] Streamlit web UI for interactive queries
- [ ] PDF full-text extraction and chunking
- [ ] Semantic memory using vector search
- [ ] Export reports as PDF
- [ ] Deploy to Hugging Face Spaces
- [ ] Add citation formatting (APA/MLA)
- [ ] Multi-user support

---
