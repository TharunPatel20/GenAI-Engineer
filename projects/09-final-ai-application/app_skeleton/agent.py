"""M4: an agent that plans, calls the search tool, reads the result, and answers with evidence.

TODO (M4): write the plan, act, observe loop, stopping at a final answer or MAX_STEPS.
  Pattern: ../../07-agentic-ai-intro/agent_loop.py (run_agent, build_prompt, parse_reply).
  Note: project 08 (LangGraph) is not in this repository. Use project 07's plain loop.

TODO (M4): register search_documents(question) as a tool, with a docstring the model reads.
  Pattern: ../../07-agentic-ai-intro/tools.py (TOOLS dict and describe_tools).
  The tool calls retrieval.search() from M3.

TODO (M4): tell the model to answer only from the retrieved text, to quote the source,
  and to say "I don't know" when the text does not contain the answer.
  Pattern: ../../05-rag-fundamentals/ask_rag.py (grounded prompt rules).

TODO (M4): parse the model's JSON reply safely. A bad reply should become an error
  message for the next step, not a crash.
  Pattern: ../../07-agentic-ai-intro/agent_loop.py (parse_reply).

TODO (M6): log each step (question, tool call, sources, answer) to logs/ for review.
"""
