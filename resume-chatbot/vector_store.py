"""
Step 3 - EMBED and STORE (and a first look at RETRIEVE).

Turns each chunk into a vector with Gemini's embedding model, then puts all
vectors into a FAISS index so we can find the chunks closest to a question.
"""
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()  # reads GOOGLE_API_KEY etc. from .env into environment variables


def get_embeddings():
    """The embedding model: text in -> list of numbers (a vector) out."""
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        # The API key is picked up automatically from GOOGLE_API_KEY.
    )


def build_vector_store(chunks):
    """
    Embed every chunk and store the vectors in FAISS.

    FAISS.from_documents does two things in one call:
      1. sends each chunk's text to the embedding model -> gets a vector back
      2. builds an index over those vectors for fast similarity search
    The original text + metadata is kept next to each vector, so a search
    returns the actual chunk, not just numbers.
    """
    return FAISS.from_documents(chunks, get_embeddings())


# Run directly to try semantic search on your resume:
#   python vector_store.py resume0.pdf
if __name__ == "__main__":
    import sys

    from resume_loader import load_resume, split_into_chunks

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <resume.pdf>")
        sys.exit(1)

    chunks = split_into_chunks(load_resume(sys.argv[1]))
    print(f"Embedding {len(chunks)} chunks with Gemini...")
    store = build_vector_store(chunks)

    # Peek at one vector so you can see what an embedding actually is.
    sample = get_embeddings().embed_query("Python developer")
    print(f"An embedding is a list of {len(sample)} numbers, e.g. {sample[:5]}\n")

    print("Ask a question (empty line to quit).")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break
        # k=3 -> return the 3 closest chunks. Score = distance: LOWER is closer.
        results = store.similarity_search_with_score(question, k=3)
        for rank, (doc, score) in enumerate(results, start=1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  #{rank}  distance={score:.3f}  | {preview}...")
