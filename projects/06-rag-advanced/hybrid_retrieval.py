"""Project 06: Hybrid retrieval = BM25 keyword search + FAISS vector search.

The two ranked lists are merged with reciprocal rank fusion (RRF).
evaluate.py imports this file. You can also run it directly:

    python hybrid_retrieval.py ACR-4471

Keyword search needs no API key. Vector and hybrid search need GEMINI_API_KEY in .env.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Step 1: Load GEMINI_API_KEY (and optional model names) from the .env file in this folder.
load_dotenv(Path(__file__).with_name(".env"))

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "sample_docs"
# Model names change, so check Google AI Studio if a name stops working.
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
RRF_CONSTANT = 60  # the usual constant in reciprocal rank fusion


# Step 2: Helpers. Splitting text into chunks and tokens for keyword search.
def get_api_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and paste your key after GEMINI_API_KEY="
        )
    return key


def tokenize(text):
    # Keeps codes such as "acr-4471" as one token, so an exact code can match exactly.
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


def load_chunks(chunk_size=300, chunk_overlap=0):
    text = (DOCS_DIR / "faq.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents([text], metadatas=[{"source": "faq.txt"}])
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
    return chunks


def reciprocal_rank_fusion(rankings, k=RRF_CONSTANT):
    # Each list gives an item 1 / (k + rank). Items found by several lists add up.
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


# Step 4: The retriever. Keyword (BM25), vector (FAISS), and hybrid (RRF) search.
class HybridRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])
        self._vector_store = None  # built on the first vector search, so keyword search needs no key

    def _vectors(self):
        if self._vector_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=get_api_key())
            self._vector_store = FAISS.from_documents(self.chunks, embeddings)
        return self._vector_store

    def keyword_ranking(self, query):
        # Returns chunk ids, best first. Chunks with no matching word are left out.
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0]

    def vector_ranking(self, query, pool=10):
        # FAISS returns the closest chunks first. We only use the order, not the raw distance.
        pool = min(pool, len(self.chunks))
        docs = self._vectors().similarity_search(query, k=pool)
        return [d.metadata["chunk_id"] for d in docs]

    def search(self, query, k=3, mode="hybrid", pool=10):
        if mode == "keyword":
            ids = self.keyword_ranking(query)
        elif mode == "vector":
            ids = self.vector_ranking(query, pool)
        elif mode == "hybrid":
            ids = reciprocal_rank_fusion([self.keyword_ranking(query), self.vector_ranking(query, pool)])
        else:
            raise ValueError(f"Unknown mode: {mode}. Use keyword, vector, or hybrid.")
        return [self.chunks[i] for i in ids[:k]]


# Step 5 (OPTIONAL): Query rewriting. Ask Gemini for a search-friendly version of the question.
REWRITE_PROMPT = (
    "Rewrite the user's question so it works well as a search query over a product FAQ. "
    "Keep every code, number and product name exactly as written. Use plain keywords. "
    "Return only the rewritten query.\n\nQuestion: {question}"
)


def rewrite_query(question):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=get_api_key(), temperature=0)
    reply = llm.invoke(REWRITE_PROMPT.format(question=question))
    return reply.content.strip()


# Step 6: PLACEHOLDER reranker. It scores each chunk by how many query words it contains.
# A real cross-encoder reranker (a model that reads the query and chunk together) is a stretch exercise.
def rerank_by_overlap(query, docs):
    query_terms = set(tokenize(query))

    def overlap(doc):
        return len(query_terms & set(tokenize(doc.page_content)))

    return sorted(docs, key=overlap, reverse=True)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is order code ACR-4471?"
    retriever = HybridRetriever()
    print(f"Query: {query}\n")

    print("Keyword (BM25) top 3:")
    for doc in retriever.search(query, mode="keyword"):
        print(" -", doc.page_content[:90].replace("\n", " | "))

    if os.getenv("GEMINI_API_KEY", "").strip():
        print("\nHybrid (BM25 + vectors, RRF) top 3:")
        for doc in retriever.search(query, mode="hybrid"):
            print(" -", doc.page_content[:90].replace("\n", " | "))
    else:
        print("\nSkipped hybrid search: GEMINI_API_KEY is not set in .env.")
