"""M2: load documents, split them into chunks, embed them, and save a FAISS index.

TODO (M2): load every file in data/ with a document loader.
  Pattern: ../../05-rag-fundamentals/build_index.py (steps 1 to 3).
  Start with TextLoader on company_notes.txt, then add other loaders only if needed.

TODO (M2): split into chunks of 1000 characters with 100 overlap.
  Pattern: ../../05-rag-fundamentals/build_index.py (RecursiveCharacterTextSplitter).
  Write down the chunk count. You will need it for the write-up.

TODO (M2): embed the chunks with the Gemini embedding model and save FAISS to faiss_index/.
  Pattern: ../../05-rag-fundamentals/build_index.py (FAISS.from_documents, save_local).

TODO (M2): remove personal identifiers from documents before indexing, or leave them out.
  See safety.py and README M6.

TODO (M3): also build a BM25 index over the same chunks for keyword search.
  Pattern: ../../06-rag-advanced/hybrid_retrieval.py (rank_bm25, tokenizer that keeps codes).
"""
