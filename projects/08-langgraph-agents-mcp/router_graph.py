"""Project 08: A minimal LangGraph router with a guardrail.

Flow: router -> billing or technical specialist -> guardrail -> end.

Keyword routing runs first and needs no key. Gemini is asked only when the
keywords do not decide, and only if GEMINI_API_KEY is set.

Run:           python router_graph.py
Offline only:  python router_graph.py --offline
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Step 1: Load GEMINI_API_KEY from the .env file in this folder.
load_dotenv(Path(__file__).with_name(".env"))

log = logging.getLogger("router_graph")

# Model names and free-tier limits change. Check Google AI Studio for the current name.
MODEL_NAME = "gemini-2.5-flash"

# Step 5: Any refund above this amount (in dollars) is blocked by the guardrail.
REFUND_LIMIT = 100.0

# Step 3: Keyword rules. The category with more matches wins.
BILLING_WORDS = ("bill", "billing", "invoice", "charge", "charged", "refund", "payment", "price", "subscription")
TECHNICAL_WORDS = ("error", "crash", "bug", "login", "log in", "install", "slow", "broken", "not working")

REFUND_PATTERN = re.compile(r"refund \$(\d+(?:\.\d+)?)", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")

# Set in main(): True only when a real key is present and --offline is not used.
USE_LLM = False


# Step 2: The typed state. Every node reads it and returns only the keys it changes.
class RouterState(TypedDict):
    request: str
    category: str
    reply: str
    blocked: bool


def keyword_category(text):
    """Return "billing", "technical", or None when the keywords do not decide."""
    lowered = text.lower()
    billing = sum(1 for word in BILLING_WORDS if re.search(rf"\b{re.escape(word)}\b", lowered))
    technical = sum(1 for word in TECHNICAL_WORDS if re.search(rf"\b{re.escape(word)}\b", lowered))
    if billing > technical:
        return "billing"
    if technical > billing:
        return "technical"
    return None


def llm_category(text):
    """Ask Gemini to classify the request. Returns "billing", "technical", or None."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    prompt = "Classify this customer request as exactly one word: billing or technical.\n\nRequest: " + text
    answer = str(llm.invoke(prompt).content).strip().lower()
    if "billing" in answer:
        return "billing"
    if "technical" in answer:
        return "technical"
    return None


def router_node(state):
    """Step 3 and 4: keywords first, then the LLM if enabled, then a safe default."""
    category = keyword_category(state["request"])
    source = "keywords"
    if category is None and USE_LLM:
        try:
            category = llm_category(state["request"])
            source = "llm"
        except Exception as error:  # for example a network or quota error
            log.warning("LLM routing failed (%s), so the default is used", error)
    if category is None:
        category = "technical"
        source = "default"
    log.info("router chose %s via %s", category, source)
    return {"category": category}


def route_by_category(state):
    """Conditional edge: the name of the next node."""
    return state["category"]


def billing_agent(state):
    """Specialist for money questions. Canned reply, no real account is touched."""
    match = AMOUNT_PATTERN.search(state["request"])
    if match:
        reply = f"I can refund ${float(match.group(1)):.2f} to your original payment method."
    else:
        reply = "Please tell me the amount you were charged so I can check it."
    return {"reply": reply}


def technical_agent(state):
    """Specialist for app problems. Canned reply."""
    return {"reply": "Please clear your browser cache, then try logging in again. If the error continues, send us the error text."}


def guardrail_node(state):
    """Step 5: block any answer that promises a refund over REFUND_LIMIT."""
    match = REFUND_PATTERN.search(state["reply"])
    if match and float(match.group(1)) > REFUND_LIMIT:
        blocked_reply = (
            f"Blocked by guardrail: refunds over ${REFUND_LIMIT:.0f} need a human agent. "
            "A person will review this request."
        )
        return {"reply": blocked_reply, "blocked": True}
    return {"blocked": False}


def traced(name, node_fn):
    """Observability: log the node name and the state KEYS only (not values, so no customer text is logged)."""

    def wrapper(state):
        log.info("enter node=%s state_keys=%s", name, sorted(state.keys()))
        update = node_fn(state)
        log.info("exit node=%s updated_keys=%s", name, sorted(update.keys()))
        return update

    return wrapper


def build_graph():
    """Step 6: add the nodes and the edges, then compile."""
    graph = StateGraph(RouterState)
    graph.add_node("router", traced("router", router_node))
    graph.add_node("billing", traced("billing", billing_agent))
    graph.add_node("technical", traced("technical", technical_agent))
    graph.add_node("guardrail", traced("guardrail", guardrail_node))
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_by_category, {"billing": "billing", "technical": "technical"})
    graph.add_edge("billing", "guardrail")
    graph.add_edge("technical", "guardrail")
    graph.add_edge("guardrail", END)
    return graph.compile()


def main():
    global USE_LLM
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    offline = "--offline" in sys.argv
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    USE_LLM = not offline and bool(api_key) and api_key != "your-key-here"
    if USE_LLM:
        print(f"Mode: keywords first, then Gemini ({MODEL_NAME}) if the keywords do not decide")
    else:
        print("Mode: OFFLINE keyword routing only (no network calls)")

    app = build_graph()
    examples = [
        "I was charged $250 twice, please refund the payment.",
        "The app shows an error when I try to log in.",
    ]
    for request in examples:
        result = app.invoke({"request": request, "category": "", "reply": "", "blocked": False})
        print(f"\nRequest: {request}")
        print(f"Category: {result['category']}  Blocked: {result['blocked']}")
        print(f"Reply: {result['reply']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
