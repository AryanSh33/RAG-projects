"""
Steps 4, 6, 7 - RETRIEVE + GENERATE, with conversation memory and transparency.

The pipeline is split into three separate stages (instead of one opaque chain)
so the UI can SHOW what happens at each one:

    1. rewrite()       follow-up question  -> standalone question   (LLM call)
    2. retrieve()      standalone question -> top-k resume chunks   (embeddings + FAISS)
    3. stream_answer() question + chunks + history -> answer        (LLM call, streamed)
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt 1 (Step 6): rewrite a follow-up into a standalone question.
# ---------------------------------------------------------------------------
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the question so it "
     "can be understood WITHOUT the chat history. Replace words like 'it', 'that', "
     "'the second one', 'he' with what they refer to. "
     "Resolve references using the MOST RECENT assistant answer first: if it was a "
     "list, 'the first/second/last one' means that item of THAT list. Name the item "
     "explicitly in the rewritten question (e.g. 'the Backend & Tools skills'). "
     "Do NOT answer the question. Return only the rewritten question. "
     "If it is already standalone, return it unchanged."),
    MessagesPlaceholder("chat_history"),  # the earlier messages get inserted here
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Prompt 2 (Step 4): answer using the retrieved resume chunks.
# ---------------------------------------------------------------------------
ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a candidate's resume.

Rules:
- Answer ONLY using the resume excerpts below. Do not use outside knowledge or guess.
- If the answer is not in the excerpts, say: "The resume doesn't mention that."
- Be concise. Use bullet points when listing several items.
- Call the candidate "the candidate" unless their full name appears clearly as the
  resume owner's name in the excerpts. Never treat a company, college, or project
  name as the candidate's name.

Resume excerpts:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),  # so the answer can refer back naturally
    ("human", "{question}"),
])


def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash"),
        # No temperature: Gemini 3.x models use fixed sampling and ignore it.
        timeout=60,       # give up on a single request after 60s instead of hanging
        max_retries=1,    # retry once on a temporary error, then show the error
    )


def format_docs(docs):
    """Join retrieved chunks into one block of text for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class ResumeRAG:
    """The whole RAG pipeline for one resume, one method per stage."""

    def __init__(self, vector_store, k: int = 5):
        self.store = vector_store
        self.k = k
        llm = get_llm()
        # Each stage is still a small LCEL chain: prompt | llm | parser.
        self.condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def rewrite(self, question: str, chat_history: list) -> str:
        """Stage 1: turn a follow-up into a standalone question."""
        if not chat_history:  # first question: nothing to resolve, skip the LLM call
            return question
        # str(): the parser returns a str-like "TextAccessor" object, which the
        # Gemini embeddings API rejects with a 500 error. Convert to a real str.
        result = self.condense_chain.invoke(
            {"question": question, "chat_history": chat_history}
        )
        return str(result).strip()

    def retrieve(self, query: str):
        """Stage 2: return [(Document, distance), ...], closest first (lower = closer)."""
        return self.store.similarity_search_with_score(query, k=self.k)

    def stream_answer(self, question: str, chat_history: list, docs):
        """Stage 3: yield the answer piece by piece as Gemini writes it."""
        inputs = {
            "question": question,
            "chat_history": chat_history,
            "context": format_docs(docs),
        }
        for piece in self.answer_chain.stream(inputs):
            yield str(piece)


# Run directly to chat in the terminal, with the hidden stages printed:
#   python rag_chain.py resume0.pdf
if __name__ == "__main__":
    import sys

    from resume_loader import load_resume, split_into_chunks
    from vector_store import build_vector_store

    if len(sys.argv) < 2:
        print("Usage: python rag_chain.py <resume.pdf>")
        sys.exit(1)

    rag = ResumeRAG(build_vector_store(split_into_chunks(load_resume(sys.argv[1]))))
    history = []
    print("Resume loaded. Ask a question (empty line to quit).")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            break
        recent = history[-6:]
        standalone = rag.rewrite(question, recent)
        results = rag.retrieve(standalone)
        print(f"  [searched for] {standalone}")
        for doc, score in results:
            print(f"  [chunk d={score:.2f}] {doc.page_content[:70].replace(chr(10), ' ')}...")
        docs = [doc for doc, _ in results]
        answer = "".join(rag.stream_answer(question, recent, docs))
        print("\nBot:", answer)
        history += [("human", question), ("ai", answer)]
