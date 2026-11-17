# Function Reference

Every function in this project, grouped by pipeline step. Each step lists:

- **Project functions**: the code in this repo you can call yourself
- **Library functions**: the LangChain / Gemini / FAISS / Streamlit calls doing the work underneath

```
 PDF ─► 1. LOAD ─► 2. SPLIT ─► 3. EMBED & STORE ─► 4. REWRITE ─► 5. RETRIEVE ─► 6. GENERATE ─► 7. UI
       resume_loader.py        vector_store.py     └──────────── rag_chain.py ───────────┘     app.py
```

---

## Step 1: Load the PDF
**File:** `resume_loader.py`

| Function | What it does | Input → Output |
|---|---|---|
| `load_resume(pdf_path)` | Reads the PDF and extracts its text, one Document per page | `str` path → `list[Document]` |

| Library function | Package | Role |
|---|---|---|
| `PyPDFLoader(path).load()` | `langchain_community` | Extracts text from each PDF page (uses `pypdf`) |
| `Document.page_content` / `.metadata` | `langchain_core` | The page text / info like `{"source": ..., "page": 0}` |

**Try it:** `python resume_loader.py resume.pdf`

---

## Step 2: Split into chunks
**File:** `resume_loader.py`

| Function | What it does | Input → Output |
|---|---|---|
| `split_into_chunks(pages, chunk_size=500, chunk_overlap=100)` | Cuts pages into small overlapping pieces so search can target one section | `list[Document]` → `list[Document]` |

| Library function | Package | Role |
|---|---|---|
| `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators)` | `langchain_text_splitters` | Splits on `\n\n` → `\n` → `. ` → ` ` (largest boundary that fits) |
| `.split_documents(pages)` | `langchain_text_splitters` | Applies the split and keeps each chunk's metadata |

| Parameter | Default | Effect of changing it |
|---|---|---|
| `chunk_size` | `500` | Smaller = more precise search but less context per chunk |
| `chunk_overlap` | `100` | Larger = less risk of cutting a sentence in half, more duplicate text |

---

## Step 3: Embed and store
**File:** `vector_store.py`

| Function | What it does | Input → Output |
|---|---|---|
| `get_embeddings()` | Creates the Gemini embedding model (text → vector) | → `GoogleGenerativeAIEmbeddings` |
| `build_vector_store(chunks)` | Embeds every chunk and builds a searchable FAISS index | `list[Document]` → `FAISS` |

| Library function | Package | Role |
|---|---|---|
| `GoogleGenerativeAIEmbeddings(model=...)` | `langchain_google_genai` | Calls Gemini's embedding API; reads `GOOGLE_API_KEY` from env |
| `.embed_query(text)` | `langchain_google_genai` | One text → one vector (`list[float]`) |
| `FAISS.from_documents(chunks, embeddings)` | `langchain_community` | Embeds all chunks and stores vectors + original text |
| `load_dotenv()` | `python-dotenv` | Loads `.env` into environment variables |

**Try it:** `python vector_store.py resume.pdf` (semantic search with distance scores)

---

## Steps 4–6: Rewrite → Retrieve → Generate
**File:** `rag_chain.py`

### The `ResumeRAG` class: one method per stage

| Method | What it does | Shown in the UI as | LLM call? |
|---|---|---|---|
| `ResumeRAG(vector_store, k=5)` | Builds the pipeline for one resume: stores the index, creates the rewrite and answer chains | (created on upload) | No |
| `rag.rewrite(question, chat_history)` | Turns a follow-up (*"the second one"*) into a standalone question. Skipped if there's no history | "Understanding the question" | Yes (follow-ups only) |
| `rag.retrieve(query)` | Finds the top `k` chunks closest to the query, with distances (lower = closer) | "Found 5 relevant sections" | No (1 embedding call) |
| `rag.stream_answer(question, chat_history, docs)` | Generates the answer from the chunks + history, yielding it piece by piece | The answer itself | Yes |

**Input / output types**

| Method | Input | Output |
|---|---|---|
| `rewrite()` | `str`, `list[tuple[str, str]]` e.g. `[("human", "..."), ("ai", "...")]` | `str` |
| `retrieve()` | `str` | `list[tuple[Document, float]]` |
| `stream_answer()` | `str`, history list, `list[Document]` | generator of `str` |

### Helper functions

| Function | What it does | Input → Output |
|---|---|---|
| `get_llm()` | Creates the Gemini chat model from `GEMINI_CHAT_MODEL` (60s timeout, 1 retry) | → `ChatGoogleGenerativeAI` |
| `format_docs(docs)` | Joins chunks into one text block for the prompt, separated by `---` | `list[Document]` → `str` |

### Prompts

| Name | Used by | Purpose |
|---|---|---|
| `CONDENSE_PROMPT` | `rewrite()` | Rewrites follow-ups; resolves *"it / that / the second one"* using the latest answer |
| `ANSWER_PROMPT` / `ANSWER_SYSTEM_PROMPT` | `stream_answer()` | Answer only from the excerpts; say *"The resume doesn't mention that."* otherwise |

### Library functions

| Library function | Package | Role |
|---|---|---|
| `ChatPromptTemplate.from_messages([...])` | `langchain_core` | Template with `{context}` / `{question}` placeholders |
| `MessagesPlaceholder("chat_history")` | `langchain_core` | Slot where earlier chat messages are inserted |
| `ChatGoogleGenerativeAI(model, timeout, max_retries)` | `langchain_google_genai` | The Gemini chat model |
| `StrOutputParser()` | `langchain_core` | Extracts plain text from the model's response |
| `prompt \| llm \| parser` | `langchain_core` (LCEL) | Pipes each step's output into the next |
| `.invoke(inputs)` | `langchain_core` | Run a chain once → full result (used by `rewrite`) |
| `.stream(inputs)` | `langchain_core` | Run a chain → yields pieces as generated (used by `stream_answer`) |
| `vector_store.similarity_search_with_score(query, k)` | `langchain_community` | Embeds the query, returns the `k` nearest chunks + distances |

**Try it:** `python rag_chain.py resume.pdf` (terminal chat that prints the hidden stages)

---

## Step 7: Chat UI
**File:** `app.py`

### Project functions

| Function | What it does | Input → Output |
|---|---|---|
| `process_pdf(uploaded_file)` | Saves the upload to a temp file, runs Steps 1–3, deletes the file | `UploadedFile` → `(ResumeRAG, int chunk_count)` |
| `show_rewrite(question, standalone)` | Shows the search query, or says no rewrite was needed | → renders UI |
| `show_sources(sources)` | Lists retrieved chunks with page and distance | `list[dict]` → renders UI |
| `show_details(question, details)` | Replays the "How I found this" panel for past messages | → renders UI |

### Session state (survives Streamlit reruns)

| Key | Holds |
|---|---|
| `st.session_state.messages` | Chat history: `{"role", "content", "details"}` per message |
| `st.session_state.rag` | The `ResumeRAG` pipeline for the uploaded resume |
| `st.session_state.file_key` | `name-size` of the indexed file, so it's only rebuilt for a new upload |

### Streamlit functions

| Library function | Role |
|---|---|
| `st.file_uploader("...", type="pdf")` | PDF upload in the sidebar |
| `st.spinner("...")` | Loading indicator while indexing |
| `st.chat_input("...", submit_mode="disable")` | Message box; disabled while an answer streams |
| `st.chat_message(role)` | A chat bubble for `"user"` or `"assistant"` |
| `st.status(label, type="compact")` | Collapsible "How I found this" panel (live) |
| `st.status(label, type="step")` | One timeline step inside the panel |
| `st.expander(label, type="compact")` | Same panel replayed for past messages |
| `st.write_stream(generator)` | Displays the answer as it streams; returns the full text |
| `st.session_state` | Dictionary that persists across reruns |
| `st.stop()` / `st.rerun()` | Stop drawing the page / redraw it immediately |

**Run it:** `streamlit run app.py`

---

## Using the pipeline in your own code

```python
from resume_loader import load_resume, split_into_chunks
from vector_store import build_vector_store
from rag_chain import ResumeRAG

# Steps 1-3: build the index once per resume
rag = ResumeRAG(build_vector_store(split_into_chunks(load_resume("resume.pdf"))))

history = []
for question in ["What projects has the candidate built?", "Tell me more about the second one"]:
    standalone = rag.rewrite(question, history)          # Step 4
    results = rag.retrieve(standalone)                    # Step 5
    docs = [doc for doc, _distance in results]
    answer = "".join(rag.stream_answer(question, history, docs))   # Step 6
    print(f"Q: {question}\n   searched: {standalone}\nA: {answer}\n")
    history += [("human", question), ("ai", answer)]
```
