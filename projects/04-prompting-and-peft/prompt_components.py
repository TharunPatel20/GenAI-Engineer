"""Project 04: Build a prompt from parts, then see what a guardrail changes.

The scenario comes from the 2026-11-02 course recap: a customer was charged
twice and wants a refund.

Run:           python prompt_components.py
Offline demo:  python prompt_components.py --offline   (no API key needed)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).with_name(".env"))

# Model names and free-tier limits change. Check Google AI Studio for the current name.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Step 1: Each prompt component is a separate value.
PERSONA = "Senior Customer Care Executive"
INSTRUCTION = "Write an email to the customer."
CONTEXT = "The customer was charged twice for the subscription in one month."
TONE = "Empathetic"
GUARDRAIL = "Do not promise a refund. Say that an investigation will take place."
OUTPUT_FORMAT = "A greeting, a body, and a conclusion."


# Step 2: Assemble the parts into one prompt string.
# The guardrail is optional, so the same builder can make a prompt with or without it.
def build_prompt(persona, instruction, context, tone, output_format, guardrail=None):
    lines = [
        f"You are a {persona}.",
        f"Task: {instruction}",
        f"Context: {context}",
        f"Tone: {tone}",
    ]
    if guardrail:
        lines.append(f"Rules you must follow: {guardrail}")
    lines.append(f"Output format: {output_format}")
    return "\n".join(lines)


# Step 3: Canned replies for the offline demo. They are NOT real model output.
OFFLINE_WITHOUT_GUARDRAIL = (
    "Dear customer, we are sorry for the double charge. "
    "We have issued a full refund to your card today."
)
OFFLINE_WITH_GUARDRAIL = (
    "Dear customer, we are sorry for the double charge. "
    "We have opened an investigation and will update you within three business days. "
    "We cannot promise a refund until it is complete."
)


def gemini_reply(client, prompt):
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


def main():
    prompt_plain = build_prompt(PERSONA, INSTRUCTION, CONTEXT, TONE, OUTPUT_FORMAT)
    prompt_guarded = build_prompt(PERSONA, INSTRUCTION, CONTEXT, TONE, OUTPUT_FORMAT, GUARDRAIL)

    # Step 4: Show the two prompts. The only difference is the "Rules" line.
    print("PROMPT WITHOUT GUARDRAIL")
    print("-" * 40)
    print(prompt_plain)
    print()
    print("PROMPT WITH GUARDRAIL")
    print("-" * 40)
    print(prompt_guarded)
    print()

    # Step 5: Send both prompts to Gemini if a key is set. Otherwise use the offline demo.
    offline = "--offline" in sys.argv
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if offline or not api_key or api_key == "your-key-here":
        print("No live call made (no key, or --offline was used).")
        print("Offline demo replies below are canned text, NOT real model output.")
        reply_plain, reply_guarded = OFFLINE_WITHOUT_GUARDRAIL, OFFLINE_WITH_GUARDRAIL
    else:
        client = genai.Client(api_key=api_key)
        print(f"Calling Gemini model: {MODEL_NAME}")
        try:
            reply_plain = gemini_reply(client, prompt_plain)
            reply_guarded = gemini_reply(client, prompt_guarded)
        except Exception as error:
            print(f"Call failed: {error}")
            return 1

    print()
    print("REPLY WITHOUT GUARDRAIL")
    print("-" * 40)
    print(reply_plain)
    print()
    print("REPLY WITH GUARDRAIL")
    print("-" * 40)
    print(reply_guarded)
    print()
    print("Check: does the guarded reply avoid promising a refund?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
