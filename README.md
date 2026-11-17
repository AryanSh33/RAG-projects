# RAG Projects

A collection of **Retrieval-Augmented Generation (RAG)** projects built with [LangChain](https://python.langchain.com/).

RAG lets an LLM answer questions about *your* documents: instead of relying on what the model memorized, the app first **retrieves** the relevant parts of a document, then has the model **generate** an answer grounded in them.

## Projects

| Project | Description | Stack |
|---|---|---|
| [resume-chatbot](resume-chatbot/) | Upload a resume PDF and chat with it. Handles follow-up questions and shows which resume sections each answer came from. | LangChain, Gemini, FAISS, Streamlit |

## Repository layout

```
RAG-projects/
├── README.md          ← you are here
├── .gitignore         ← shared: keeps .env, venvs and PDFs out of Git
└── resume-chatbot/    ← one folder per project, each with its own README
```

Each project is self-contained: its own `requirements.txt`, `.env.example`, and setup instructions.
