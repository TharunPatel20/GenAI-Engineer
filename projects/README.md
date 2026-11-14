# Hands-On Projects Learning Track

This folder turns the course topics that need practice into beginner projects. You learn by building. Each project has a README with objectives, setup, numbered build steps, checkpoints, troubleshooting, and stretch goals, plus a `notebook.ipynb` you can run in Google Colab.

**New here? Start with [START-HERE.md](../START-HERE.md).** It explains the learning path and how to work through these projects in order.

## Before You Start

- **Recommended: run in Google Colab.** Click the **Open in Colab** badge at the top of a project README. Colab gives each project a fresh, isolated environment, so version clashes between projects do not matter. Run the cells top to bottom.
- **Gemini API key.** Projects that call Gemini read the key from Colab Secrets (the key icon in the left sidebar; name it `GEMINI_API_KEY`). Never paste a key into a cell, code file, or chat. Get a free key from Google AI Studio.
- **Running locally (optional).** Use Python 3.10 or newer and a virtual environment inside each project folder (`python -m venv .venv`). Copy `.env.example` to `.env` and add your key there. Never commit `.env`.
- **Version pins are starting points.** Each `requirements.txt` lists exact versions that were chosen but not yet run end to end. If an install fails, upgrade the one package that fails, then continue. Note the change in your own notes.
- **Local dependency clash.** Projects 05, 06, and 07 pin slightly different `python-dotenv` versions. This matters only for local installs. Colab avoids it, or use a separate virtual environment for each project.

## Projects in Order

| # | Project | What you build | Needs API key? |
|---|---|---|---|
| 01 | [Python Foundations](01-python-foundations/) | A script that loads, summarizes, and charts a loan dataset with Pandas and Matplotlib | No (fully offline) |
| 02 | [ML and DL Intro](02-ml-and-dl-intro/) | Logistic regression, an overfitting demo, and a small neural network on a loan-approval dataset | No (fully offline) |
| 03 | [CNN and Sequence Models](03-cnn-and-sequence-models/) | A hand-written convolution and pooling walkthrough, then a small PyTorch CNN on the digits dataset | No (fully offline) |
| 04 | [Prompting and PEFT](04-prompting-and-peft/) | Zero-shot, few-shot, and chain-of-thought prompts for classifying feedback, plus a prompt-component builder | Yes for live answers. Has an offline demo mode. |
| 05 | [RAG Fundamentals](05-rag-fundamentals/) | A basic RAG pipeline: load, chunk, embed, index with FAISS, retrieve, and answer with evidence | Yes (embeddings and answers) |
| 06 | [RAG Advanced](06-rag-advanced/) | Hybrid retrieval (BM25 plus vectors), a retrieval hit-rate evaluation, and a groundedness check | Yes for vector retrieval and evaluation. The BM25 part alone runs offline. |
| 07 | [Agentic AI Intro](07-agentic-ai-intro/) | A plan-act-observe agent loop written from scratch, with tools and a simple memory | Yes for live runs. Has an offline demo mode. |
| 08 | [LangGraph Agents and MCP](08-langgraph-agents-mcp/) | A LangGraph router graph with a guardrail node, plus an optional MCP (Model Context Protocol) stretch section | Keyword routing runs offline. LLM routing needs a key. |
| 09 | [Final AI Application](09-final-ai-application/) | A scoped capstone template with milestones and a starter skeleton that combines projects 05 to 08 | Yes, once you build it |

## Suggested Learning Path

Work in order. Each project builds on the one before it:

1. **Foundations (01):** get comfortable with Python, Pandas, and charts.
2. **ML basics (02):** train, test, and compare models.
3. **Neural networks (03):** see how convolution and pooling work, then train a small CNN.
4. **Prompting (04):** learn how prompt structure changes model output.
5. **RAG (05, 06):** ground answers in your own documents, then improve retrieval and measure it.
6. **Agents (07, 08):** build a tool-using agent by hand, then with a framework.
7. **Capstone (09):** combine retrieval and agents into one application, with safety checks.

Projects 01 to 03 are the best place to start without an API key. Projects 04 to 09 need a Gemini key for full behavior. Some projects also run in an offline demo mode, which is labeled as a demonstration, not real model output.

## Related Course Materials

- Study guide: [../AI-Engineer-Guide.md](../AI-Engineer-Guide.md)
- Per-session recaps: [../claude-files/](../claude-files/)
- Repository index: [../README.md](../README.md)

## Status Note

None of these projects have been run end to end on a clean machine. Each was written and checked by reading. Expect to fix small version or path issues, and check the README troubleshooting sections first.
