"""Settings for the final app: secrets from .env, model names, and folder paths.

Keep names and paths here, so no other file hard-codes them.
"""

# TODO (M2): load the .env file from the project folder (one level above this file).
#   Pattern: ../../05-rag-fundamentals/build_index.py (reads GEMINI_API_KEY the same way).
#   Expected: dotenv.load_dotenv(PROJECT_DIR / ".env")

# TODO (M2): read GEMINI_API_KEY from the environment. Never print it.
#   Pattern: ../../05-rag-fundamentals/ask_rag.py

# TODO (M2): set the model names. Confirm current names in Google AI Studio first.
#   Chat model pattern: ../../05-rag-fundamentals/ask_rag.py (gemini-2.5-flash)
#   Embedding model pattern: ../../05-rag-fundamentals/build_index.py (models/gemini-embedding-001)
#   Optional override: GEMINI_EMBED_MODEL, as in ../../06-rag-advanced/hybrid_retrieval.py

# TODO (M2): set the folder paths: data/ (your documents), faiss_index/ (saved index),
#   and logs/ (step log for M6 and the stretch dashboard).

# TODO (M4): set MAX_STEPS, the agent's step limit.
#   Pattern: ../../07-agentic-ai-intro/agent_loop.py (MAX_STEPS = 6)

# TODO (M3): set TOP_K, the number of chunks to retrieve (project 06 uses 3).
