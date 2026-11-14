"""M3: find the chunks that answer a question, using keyword, vector, or hybrid search.

TODO (M3): load the saved FAISS index and the BM25 index built in ingest.py.
  Pattern: ../../05-rag-fundamentals/ask_rag.py (FAISS.load_local) and
  ../../06-rag-advanced/hybrid_retrieval.py (HybridRetriever).

TODO (M3): write search(question, mode, k) with modes "keyword", "vector", and "hybrid".
  Pattern: ../../06-rag-advanced/hybrid_retrieval.py (reciprocal_rank_fusion).
  Do not read meaning into raw scores. FAISS with default L2 returns distances,
  where lower is closer. Check the docs for your installed version.

TODO (M3): return each chunk with its source file name, so answers can cite it.

TODO (M3): write the hit-rate check. It belongs in an evaluate script, not here.
  Pattern: ../../06-rag-advanced/evaluate.py (function hit() and the table in main()).
  Use your own questions and expected phrases, with at least 10 questions.
"""
