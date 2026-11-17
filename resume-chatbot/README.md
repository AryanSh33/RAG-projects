# Resume Chatbot (RAG)

Upload a resume PDF and ask questions about it in plain English — *"What are their skills?"*, *"Tell me more about the second project"*, *"Do they know Kubernetes?"*. Answers come **only** from the resume; if something isn't there, the bot says so instead of guessing.

Built with **LangChain**, **Google Gemini**, **FAISS**, and **Streamlit**.

## Features

- **PDF upload** — drop in any text-based resume PDF
- **Grounded answers** — the model may only use retrieved resume excerpts
- **Conversation memory** — understands follow-ups like *"the second one"* or *"what accuracy did it get?"*
- **Streaming responses** — answers appear word by word
- **"How I found this" panel** — every answer shows the rewritten search query and the exact resume sections (with similarity scores) it was based on

## How it works

```
 Resume PDF
     │  1. LOAD      PyPDFLoader extracts the text
     ▼
 Pages
     │  2. SPLIT     RecursiveCharacterTextSplitter → ~500-char chunks, 100-char overlap
     ▼
 Chunks
     │  3. EMBED     Gemini embedding model → one vector per chunk
     ▼
 FAISS index  ◄──────────────────────────────┐
                                             │
 User question + chat history                │
     │  4. REWRITE   LLM turns a follow-up into a standalone question
     ▼                                       │
 Standalone question ──  5. RETRIEVE  ───────┘  top-5 closest chunks
     │
     ▼  6. GENERATE  LLM answers using ONLY those chunks (+ history), streamed
 Answer + "How I found this"
```

**Why rewrite the question?** A follow-up like *"tell me about the second one"* contains nothing searchable. Step 4 uses the chat history to turn it into e.g. *"Tell me about the candidate's Backend & Tools skills"* before searching.

## Project structure

| File | Role |
|---|---|
| `resume_loader.py` | Load the PDF and split it into chunks |
| `vector_store.py` | Embed chunks and build the FAISS index |
| `rag_chain.py` | `ResumeRAG` class: rewrite → retrieve → stream answer, plus the prompts |
| `app.py` | Streamlit chat UI |

Each of the first three files can also be run on its own from the terminal to inspect that stage (see below).

## Setup

**Requirements:** Python 3.11+, a free [Gemini API key](https://aistudio.google.com/apikey).

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
cd resume-chatbot
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env                 # macOS/Linux: cp .env.example .env
#    then edit .env and paste your key after GOOGLE_API_KEY=
```

> `.env` is git-ignored. Never commit your API key.

## Run

```powershell
streamlit run app.py
```

Opens at http://localhost:8501. Upload a resume in the sidebar and start asking.

### Inspect individual stages (terminal)

```powershell
python resume_loader.py resume.pdf   # see the chunks
python vector_store.py resume.pdf    # try semantic search with distance scores
python rag_chain.py resume.pdf       # chat in the terminal, with the hidden stages printed
```

## Configuration

Set in `.env`:

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required |
| `GEMINI_CHAT_MODEL` | `gemini-3.6-flash` | Google retires models often; if you get a 404, check the [model list](https://ai.google.dev/gemini-api/docs/models). `gemini-3.1-flash-lite` is faster with a separate free quota. |
| `GEMINI_EMBED_MODEL` | `models/gemini-embedding-001` | |

## Known limitations & lessons learned

- **Free-tier quota:** some Gemini models allow only ~20 requests/day. A follow-up question costs 2 requests (rewrite + answer).
- **Scanned/image PDFs** (e.g. some Canva exports) contain no extractable text and are rejected.
- **The rewrite step is a single point of failure:** if it misreads a follow-up, retrieval fetches the wrong sections. The "How I found this" panel makes this visible.
- `StrOutputParser` returns a str-like `TextAccessor`; passing it straight to the Gemini embeddings API causes a `500` error, so outputs are converted with `str()` first.
- The FAISS index lives in memory and is rebuilt per upload.
