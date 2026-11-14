"""M4: an agent that plans, calls the search tool, reads the result, and answers with evidence.

TODO (M4): build a LangGraph graph with a typed state, nodes, and conditional edges, stopping at a final answer or MAX_STEPS.
  Pattern: ../../08-langgraph-agents-mcp/router_graph.py (StateGraph, router, guardrail, conditional edges).

TODO (M4): register search_documents(question) as a tool node, with a docstring the model reads.
  Pattern: ../../08-langgraph-agents-mcp/router_graph.py (node and tool wiring).
  The tool calls retrieval.search() from M3.

TODO (M4): tell the model to answer only from the retrieved text, to quote the source,
  and to say "I don't know" when the text does not contain the answer.
  Pattern: ../../05-rag-fundamentals/ask_rag.py (grounded prompt rules).

TODO (M4): parse the model's JSON reply safely. A bad reply should become an error
  message for the next step, not a crash.
  Pattern: ../../08-langgraph-agents-mcp/router_graph.py (router node output handling).

TODO (M6): log each step (question, tool call, sources, answer) to logs/ for review.
"""
