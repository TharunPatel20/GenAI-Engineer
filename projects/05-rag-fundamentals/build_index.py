"""Project 05, steps 1 to 3: load a text file, split it, and build a FAISS index.

Run:    python build_index.py
Output: a folder named faiss_index/ next to this file.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).resolve().parent

# Load GEMINI_API_KEY from the .env file in this folder.
load_dotenv(HERE / ".env")

DOC_PATH = HERE / "sample_docs" / "company_notes.txt"
INDEX_DIR = HERE / "faiss_index"

# Chunk settings. The stretch exercise asks you to try 300 and compare.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Model names change. Check Google AI Studio or the Gemini API docs if this fails.
EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your-key-here":
        print("No Gemini API key found.")
        print("1. Create a free key in Google AI Studio (aistudio.google.com, 'Get API key').")
        print("2. Copy .env.example to .env and paste your key after GEMINI_API_KEY=")
        print("3. Run this script again.")
        sys.exit(1)
    return api_key


def main():
    api_key = get_api_key()

    # Step 1: Load the document as text.
    loader = TextLoader(str(DOC_PATH), encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document from {DOC_PATH.name}")

    # Step 2: Split the text into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Embed each chunk and store the vectors in a FAISS index.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Saved index with {len(chunks)} chunks to {INDEX_DIR.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
