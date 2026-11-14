"""M6: guardrails and checks. Each TODO is a check to write and test, with one test question each.

Input checks (before the agent runs):
- TODO: reject empty questions and questions longer than a limit you choose
- TODO: refuse questions outside the use case you wrote in M1
- TODO: block requests for personal identifiers and sensitive data. The course policy
  forbids requesting, storing, processing, or displaying: SSN, Aadhaar, PAN, passport
  numbers, driver's license numbers, credit card numbers, and patient records.
- TODO: rate-limit questions per session, if you add multi-user access

Agent and tool checks:
- TODO: cap the number of agent steps (MAX_STEPS in config.py)
  Pattern: ../../07-agentic-ai-intro/agent_loop.py
- TODO: allow only the tools you listed in agent.py. Reject any other tool name.
- TODO: keep the agent read-only. Any action that changes data needs a person to approve it first.

Output checks (after the agent answers):
- TODO: run a groundedness check. Flag answers whose key terms are not in the sources.
  Pattern: ../../06-rag-advanced/hybrid_retrieval.py and evaluate.py (groundedness heuristic)
  Limit: this is a rough check, not proof that an answer is correct.
- TODO: if an answer has no source, show "I don't know" instead.
- TODO: check that the answer does not repeat personal identifiers from the documents.

Fairness checks (run before the demo, not on each question):
- TODO: write paired questions that differ only in a group name, and compare the answers.
  Pattern: AI-Engineer-Guide.md section 11 (bias and fairness).
"""
