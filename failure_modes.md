# Failure Modes & Recovery — Research System Agent

## 1. API Rate Limiting (Semantic Scholar 429)

**Issue:** Too many rapid requests → 429 error
**Impact:** No papers returned, weaker synthesis
**Handling:** Exponential backoff + retry, fallback to other tools
**Fix:** Add API key in `.env` to increase rate limits

---

## 2. Irrelevant ArXiv Results

**Issue:** Query becomes too broad after stopword removal
**Impact:** Unrelated papers, poor synthesis
**Handling:** Stopword filtering + year filter + critic re-search loop
**Fix:** Add category filters (cs.AI, cs.LG, cs.CV)

---

## 3. Invalid / Missing API Key (Groq)

**Issue:** Wrong or missing API key → 401 error
**Impact:** Pipeline crash
**Handling:** Early validation at startup
**Fix:** Add valid key in `.env` and update model if deprecated

---

## 4. ChromaDB Path Issue (Local vs Colab)

**Issue:** Hardcoded path fails across environments
**Impact:** Retrieval returns no data
**Handling:** Auto environment detection + relative paths
**Fix:** Avoid absolute paths

---

## 5. Infinite Critic Loop

**Issue:** Critic keeps triggering re-search
**Impact:** High API usage, no final output
**Handling:** Max iteration cap (≤ 3)
**Fix:** Conditional stopping in graph

---

## 6. Memory Save Failure (Mem0)

**Issue:** Missing API key for embeddings
**Impact:** Memory not saved
**Handling:** Try/catch + JSON fallback
**Fix:** Use local JSON memory or add API key

---

## Design Principle

All components follow:

* Fail gracefully
* Continue execution
* Return safe defaults

➡️ System always produces a final output even if some tools fail.
