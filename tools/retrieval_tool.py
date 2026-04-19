import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Embeddings ───────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ── ChromaDB path (Colab or Local) ───────────────────────────────
if os.path.exists("/content"):
    from google.colab import drive
    drive.mount('/content/drive')
    CHROMA_PATH = "/content/drive/MyDrive/research_agent/research_db"
else:
    CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'research_db')

print(f"[OK] Environment: {'Colab' if os.path.exists('/content') else 'Local'}")
print(f"[OK] DB path: {CHROMA_PATH}")

# ── ChromaDB client & vector store ───────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

vectordb = Chroma(
    client=chroma_client,
    collection_name="research_papers",
    embedding_function=embeddings,
)

print(f"[OK] Chunks stored: {chroma_client.get_or_create_collection('research_papers').count()}")


# ── Store papers ─────────────────────────────────────────────────
def store_papers(papers: list):
    """Chunk and store a list of paper dicts into the vector database."""
    if not papers:
        print("[WARN] No papers to store")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts, metadatas = [], []
    for paper in papers:
        content = f"{paper['title']}. {paper.get('abstract', '')}"
        chunks = splitter.split_text(content)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "title": paper.get("title", "N/A"),
                "year": str(paper.get("year") or paper.get("published", "N/A")),
                "pdf_url": paper.get("pdf_url", "") or "",
            })

    vectordb.add_texts(texts=texts, metadatas=metadatas)
    print(f"[OK] Stored {len(texts)} chunks from {len(papers)} papers")


# ── Retrieve context ─────────────────────────────────────────────
def retrieve_context(query: str, k: int = 4):
    """Search the vector DB for the most relevant paper chunks."""
    docs = vectordb.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "title": doc.metadata.get("title"),
            "year": doc.metadata.get("year"),
            "pdf_url": doc.metadata.get("pdf_url")
        })
    print(f"[DEBUG] Retrieved {len(results)} chunks for query: '{query}'")
    return results


# ── LangChain tool wrapper ───────────────────────────────────────
@tool
def retrieval_tool(query: str) -> str:
    """Retrieve relevant research paper chunks from
    the vector database using semantic similarity search."""
    results = retrieve_context(query)
    return str(results)


# ── Main: test pipeline ──────────────────────────────────────────
def main():
    sample_papers = [
        {
            "title": "Deep Learning for Cancer Detection",
            "abstract": "This paper proposes a CNN-based model for early cancer detection using medical imaging. The model achieves 94% accuracy on lung cancer datasets.",
            "pdf_url": "https://arxiv.org/pdf/sample1",
            "year": 2024
        },
        {
            "title": "Transformer Models in Medical Imaging",
            "abstract": "Vision transformers outperform CNNs on radiology image classification tasks. We benchmark 5 architectures on chest X-ray datasets.",
            "pdf_url": "https://arxiv.org/pdf/sample2",
            "year": 2025
        }
    ]

    store_papers(sample_papers)

    results = retrieve_context("cancer detection accuracy", k=3)
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Title  : {r['title']}")
        print(f"Year   : {r['year']}")
        print(f"Content: {r['content']}")

    # Verify data is persisted to disk
    collection = chroma_client.get_or_create_collection("research_papers")
    print(f"\n[OK] DB ready at: {CHROMA_PATH}")
    print(f"[OK] Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()