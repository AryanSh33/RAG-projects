"""
Step 2 - LOAD and SPLIT.

Turns a resume PDF into a list of small text "chunks" (LangChain Documents)
that we can later embed and search.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_resume(pdf_path: str):
    """Read the PDF. Returns one Document per page."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Each Document has:
    #   .page_content -> the text of that page
    #   .metadata     -> {"source": "resume.pdf", "page": 0, ...}
    return pages


def split_into_chunks(pages, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Cut pages into overlapping chunks.

    chunk_size    = max characters per chunk. Resumes are dense, so ~500 chars
                    is roughly one section (e.g. one job, or the skills list).
    chunk_overlap = characters shared between neighbouring chunks, so a
                    sentence cut at a boundary still appears whole in one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraphs first, then lines, then sentences, then words.
        # "Recursive" = it falls back to the next separator only when needed.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


# Run this file directly to see what the chunks look like:
#   python resume_loader.py path\to\resume.pdf
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python resume_loader.py <resume.pdf>")
        sys.exit(1)

    pages = load_resume(sys.argv[1])
    print(f"Loaded {len(pages)} page(s)")

    chunks = split_into_chunks(pages)
    print(f"Split into {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} (page {chunk.metadata.get('page')}, {len(chunk.page_content)} chars) ---")
        print(chunk.page_content)
        print()
