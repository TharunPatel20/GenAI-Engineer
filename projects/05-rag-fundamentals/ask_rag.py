"""Project 05, steps 4 and 5: retrieve chunks, then answer from them only.

Run with the default question:  python ask_rag.py
Run with your own question:     python ask_rag.py "Which product sold the most units?"

Run build_index.py first. It creates the faiss_index/ folder this script reads.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

HERE = Path(__file__).resolve().parent

# Load GEMINI_API_KEY from the .env file in this folder.
load_dotenv(HERE / ".env")

INDEX_DIR = HERE / "faiss_index"

# Model names change. Check Google AI Studio or the Gemini API docs if this fails.
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

TOP_K = 3  # how many chunks to retrieve

DEFAULT_QUESTION = "How much will Acme Robotics invest in AI in 2026, and where will the money go?"


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your-key-here":
        print("No Gemini API key found.")
        print("1. Create a free key in Google AI Studio (aistudio.google.com, 'Get API key').")
        print("2. Copy .env.example to .env and paste your key after GEMINI_API_KEY=")
        print("3. Run this script again.")
        sys.exit(1)
    return api_key


# Step 5: Build a grounded prompt. The model sees only the retrieved chunks.
def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Chunk {number}]\n{chunk.page_content}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return (
        "You are a careful analyst. Answer the question using ONLY the context below.\n"
        "Rules:\n"
        "1. If the answer is not in the context, say \"I don't know\" and stop.\n"
        "2. After the answer, give supporting evidence: short quotes with their chunk numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    api_key = get_api_key()
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    if not INDEX_DIR.exists():
        print(f"No index found in {INDEX_DIR.name}/. Run this first: python build_index.py")
        return 1

    # Step 4: Load the saved index and retrieve the closest chunks.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    # allow_dangerous_deserialization is needed because FAISS saves a pickle file.
    # Only load an index that you built yourself with build_index.py.
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vector_store.similarity_search_with_score(question, k=TOP_K)

    print(f"Question: {question}")
    print()
    print(f"Top {TOP_K} chunks with scores (check the score direction in the LangChain and FAISS docs):")
    chunks = []
    for number, (chunk, score) in enumerate(results, start=1):
        chunks.append(chunk)
        preview = chunk.page_content[:150].replace("\n", " ")
        print(f"  [{number}] score={score:.4f}  {preview}...")
    print()

    # Step 5: Send the grounded prompt to Gemini and print the answer.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key)
    reply = llm.invoke(build_prompt(question, chunks))
    print("Answer:")
    print(reply.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
