"""
Steps 5-7 - CHAT UI with Streamlit, with a "How I found this" panel.

Run with:   streamlit run app.py
"""
import os
import tempfile

import streamlit as st

from rag_chain import ResumeRAG
from resume_loader import load_resume, split_into_chunks
from vector_store import build_vector_store

MAX_HISTORY = 6  # last 3 question/answer pairs; keeps prompts small and cheap

st.set_page_config(page_title="Resume chatbot", page_icon=":material/description:")
st.title(":material/description: Resume chatbot")
st.caption("Upload a resume PDF and ask questions about it.")


# ---------------------------------------------------------------------------
# IMPORTANT STREAMLIT CONCEPT:
# Streamlit re-runs this WHOLE file from top to bottom every time the user
# does anything (types a message, clicks a button). Normal variables are
# wiped on every rerun. st.session_state is a dictionary that SURVIVES reruns,
# so that's where we keep the RAG pipeline and the chat history.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # Each message: {"role": "user"/"assistant", "content": "...",
    #                "details": {...}}  <- details only on assistant messages
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "file_key" not in st.session_state:
    st.session_state.file_key = None  # identifies which PDF the pipeline was built from


def process_pdf(uploaded_file):
    """Run Steps 2-3 on the uploaded file and return a ready RAG pipeline."""
    # PyPDFLoader needs a real file path, but Streamlit gives us bytes in memory,
    # so we write them to a temporary file first and delete it afterwards.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        chunks = split_into_chunks(load_resume(tmp_path))
        if not chunks:
            raise ValueError("No text found in this PDF. Is it a scanned image?")
        return ResumeRAG(build_vector_store(chunks)), len(chunks)
    finally:
        os.remove(tmp_path)


def show_rewrite(question, standalone):
    """Explain what the retriever actually searched for."""
    if standalone == question:
        st.caption("No rewrite needed. Searched for the question as asked.")
    else:
        st.caption("Your follow-up was rewritten into a standalone question:")
        st.markdown(f"> {standalone}")


def show_sources(sources):
    """List the retrieved resume chunks. These are the answer's 'citations'."""
    for i, src in enumerate(sources, start=1):
        st.caption(f"#{i} · page {src['page'] + 1} · distance {src['score']:.2f} (lower = closer)")
        st.text(src["text"])


def show_details(question, details):
    """Replay the 'How I found this' panel for a message from history."""
    with st.expander("How I found this", type="compact"):
        st.markdown("**1. Search query**")
        show_rewrite(question, details["standalone"])
        st.markdown(f"**2. Resume sections used ({len(details['sources'])})**")
        show_sources(details["sources"])


# ---------------- Sidebar: upload ----------------
with st.sidebar:
    st.header("Upload resume")
    uploaded = st.file_uploader("Resume (PDF)", type="pdf")

    if uploaded is not None:
        # Only rebuild when a DIFFERENT file is uploaded, not on every rerun.
        # Embedding costs API calls and time, so we don't want to repeat it.
        file_key = f"{uploaded.name}-{uploaded.size}"
        if file_key != st.session_state.file_key:
            with st.spinner("Reading and indexing resume..."):
                try:
                    rag, n_chunks = process_pdf(uploaded)
                    st.session_state.rag = rag
                    st.session_state.file_key = file_key
                    st.session_state.messages = []  # new resume -> fresh conversation
                    st.success(f"Indexed {n_chunks} chunks from {uploaded.name}")
                except Exception as e:
                    st.session_state.rag = None
                    st.error(f"Could not process PDF: {e}")

    if st.session_state.messages and st.button("Clear chat", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()


# ---------------- Main area: chat ----------------
if st.session_state.rag is None:
    st.info("Upload a resume PDF in the sidebar to start.", icon=":material/upload_file:")
    st.stop()  # don't draw the chat box until a resume is loaded

rag = st.session_state.rag

# Re-draw the whole conversation (remember: the page is rebuilt on every rerun).
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "details" in msg:
            question = st.session_state.messages[i - 1]["content"]
            show_details(question, msg["details"])
        st.markdown(msg["content"])

# submit_mode="disable": block new input until the current answer finishes streaming.
question = st.chat_input(
    "Ask about the resume, e.g. 'What are their skills?'", submit_mode="disable"
)

if question:
    # Convert our saved messages into the (role, text) format LangChain expects.
    # Built BEFORE appending the new question, so history = previous turns only.
    chat_history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in st.session_state.messages[-MAX_HISTORY:]
    ]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # A collapsed "thinking" panel with one timeline step per pipeline stage.
            with st.status(":shimmer[Searching the resume]", type="compact") as panel:
                with st.status("Understanding the question", type="step"):
                    standalone = rag.rewrite(question, chat_history)  # Stage 1
                    show_rewrite(question, standalone)

                with st.status("Finding relevant sections", type="step") as step:
                    results = rag.retrieve(standalone)                # Stage 2
                    sources = [
                        {"text": doc.page_content,
                         "page": doc.metadata.get("page", 0),
                         "score": float(score)}
                        for doc, score in results
                    ]
                    step.update(label=f"Found {len(sources)} relevant sections")
                    show_sources(sources)

                panel.update(label="How I found this", state="complete")

            docs = [doc for doc, _ in results]
            answer = st.write_stream(                                # Stage 3
                rag.stream_answer(question, chat_history, docs)
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "details": {"standalone": standalone, "sources": sources},
            })
        except Exception as e:
            answer = f"Error from the model: {e}"
            st.error(answer, icon=":material/error:")
            st.session_state.messages.append({"role": "assistant", "content": answer})
