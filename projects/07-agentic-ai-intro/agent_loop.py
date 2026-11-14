"""Project 07: A from-scratch agent loop (plan, act, observe) with Google Gemini.

No agent framework is used. The loop is plain Python: ask the model for the
next step, parse its JSON reply, run a tool, feed the result back, and repeat.

Run:           python agent_loop.py
Offline demo:  python agent_loop.py --offline   (also runs when no API key is set)
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from tools import TOOLS, describe_tools

# Step 1: Load GEMINI_API_KEY from the .env file in this folder.
load_dotenv(Path(__file__).with_name(".env"))

# Model names and free-tier limits change. Check Google AI Studio for the current name.
MODEL_NAME = "gemini-2.5-flash"

# Step 2: Stop after this many steps, so a confused model cannot loop forever.
MAX_STEPS = 6

GOAL = (
    "Find the price of a laptop, add 12 and 30, count the words in "
    "'plan act observe', and report all three results."
)

OFFLINE_NOTE = "OFFLINE DEMO: scripted replies, not real Gemini output"

# Replies the offline stub returns, one per step. They are fixed, not model output.
OFFLINE_SCRIPT = [
    '{"action": "lookup_price", "args": {"item": "laptop"}}',
    '{"action": "add_numbers", "args": {"a": 12, "b": 30}}',
    '{"action": "get_word_count", "args": {"text": "plan act observe"}}',
    '{"final_answer": "The laptop costs 999.0 dollars, 12 + 30 is 42.0, and the phrase has 3 words."}',
]

# Step 3: The prompt. The model must answer with one JSON object.
PROMPT_TEMPLATE = """You are an agent that reaches a goal one step at a time.

Tools you can use:
{tools}

Reply with exactly one JSON object and nothing else. Use one of these two forms:
{{"action": "<tool name>", "args": {{"<argument name>": <value>}}}}
{{"final_answer": "<your answer>"}}

Use a tool when you need a fact or a calculation. Give the final answer only when you have every result you need.

Goal: {goal}

Steps so far (your memory):
{memory}

Your next reply (JSON only):"""


def build_prompt(goal, memory):
    memory_text = "\n".join(memory) if memory else "(none yet)"
    return PROMPT_TEMPLATE.format(tools=describe_tools(), goal=goal, memory=memory_text)


# Step 4: Parse the reply. Models sometimes wrap JSON in ``` fences, so remove those first.
def parse_reply(text):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    reply = json.loads(cleaned[start : end + 1])
    if not isinstance(reply, dict):
        raise ValueError("reply is not a JSON object")
    return reply


# Step 5: The Gemini call. Same pattern as project 04.
def gemini_reply(client, prompt):
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text or ""


# Offline stub: returns the scripted reply for this step.
def offline_model(prompt, step):
    index = step - 1
    if index < len(OFFLINE_SCRIPT):
        return OFFLINE_SCRIPT[index]
    return '{"final_answer": "(script ended)"}'


# Step 6: The loop. Plan (ask the model), act (run a tool), observe (store the result).
def run_agent(goal, ask_model):
    memory = []  # one line per past step; it is added to every prompt
    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(goal, memory)
        try:
            reply_text = ask_model(prompt, step)
        except Exception as error:  # for example a network or quota error
            print(f"Model call failed: {error}")
            return None
        print(f"\nStep {step} reply: {reply_text.strip()}")

        try:
            reply = parse_reply(reply_text)
        except (ValueError, json.JSONDecodeError) as error:
            observation = f"Error: your reply was not valid JSON ({error}). Reply with one JSON object."
            memory.append(f"Step {step}: bad reply -> {observation}")
            print(f"  Observation: {observation}")
            continue

        if "final_answer" in reply:
            return reply["final_answer"]

        action = reply.get("action")
        args = reply.get("args", {})
        if not isinstance(action, str) or action not in TOOLS:
            observation = f"Error: unknown tool '{action}'. Use one of: {', '.join(TOOLS)}."
        else:
            try:
                observation = f"Result: {TOOLS[action](**args)}"
            except Exception as error:  # the tool failed, so tell the model why
                observation = f"Error: {type(error).__name__}: {error}"

        memory.append(f"Step {step}: action={action} args={args} -> {observation}")
        print(f"  Observation: {observation}")

    return None


def main():
    offline = "--offline" in sys.argv
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not offline and (not api_key or api_key == "your-key-here"):
        print("No Gemini API key found, so the offline demo will run.")
        print("To use Gemini: copy .env.example to .env, paste your key after GEMINI_API_KEY=, and run again.")
        offline = True

    if offline:
        print(f"Mode: {OFFLINE_NOTE}")
        ask_model = offline_model
    else:
        client = genai.Client(api_key=api_key)
        print(f"Model: {MODEL_NAME}")
        ask_model = lambda prompt, step: gemini_reply(client, prompt)

    print(f"Goal: {GOAL}")
    answer = run_agent(GOAL, ask_model)
    print()
    if answer is None:
        print(f"No final answer after {MAX_STEPS} steps, or a model call failed.")
    else:
        print(f"Final answer: {answer}")
    if offline:
        print(OFFLINE_NOTE + ". The answer is scripted, not produced by a model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
