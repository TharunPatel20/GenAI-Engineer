"""Project 04: Compare three prompting strategies with Google Gemini.

The same customer-feedback classification task is sent with three prompts:
zero-shot, few-shot (2 labeled examples), and chain-of-thought.

Run:           python prompt_strategies.py
Offline demo:  python prompt_strategies.py --offline   (no API key needed)
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Step 1: Load GEMINI_API_KEY (and optional GEMINI_MODEL) from the .env file in this folder.
load_dotenv(Path(__file__).with_name(".env"))

# Model names and free-tier limits change. Check Google AI Studio for the current name.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Sample feedback with the label a human would give (used to check the output).
FEEDBACK = [
    ("The delivery was fast and the driver was friendly. Great service!", "positive"),
    ("I waited three hours and nobody answered the phone. Very disappointed.", "negative"),
    ("The package arrived on Tuesday, as the tracking page said.", "neutral"),
]

TASK = "Classify the customer feedback as positive, negative, or neutral."


# Step 2: Zero-shot prompt. An instruction only, with no examples.
def zero_shot_prompt(feedback):
    return f"{TASK}\nAnswer with one word only.\n\nFeedback: {feedback}\nLabel:"


# Step 3: Few-shot prompt. Two labeled examples come before the real feedback.
FEW_SHOT_EXAMPLES = [
    ("The app crashes every time I open the cart.", "negative"),
    ("Support solved my issue in five minutes. Thank you!", "positive"),
]


def few_shot_prompt(feedback):
    lines = [TASK, "Answer with one word only.", ""]
    for example_text, example_label in FEW_SHOT_EXAMPLES:
        lines.append(f"Feedback: {example_text}")
        lines.append(f"Label: {example_label}")
        lines.append("")
    lines.append(f"Feedback: {feedback}")
    lines.append("Label:")
    return "\n".join(lines)


# Step 4: Chain-of-thought prompt. The model reasons in steps before the answer.
def chain_of_thought_prompt(feedback):
    return (
        f"{TASK}\n"
        "Think step by step: list the good points, then the bad points, "
        "then decide which side is stronger.\n"
        "End with a final line in this form: Final label: <positive|negative|neutral>\n\n"
        f"Feedback: {feedback}"
    )


# Step 5: Offline stub. Returns canned replies so the structure can be run without a key.
# These are NOT real model outputs. They are fixed demo answers.
OFFLINE_NOTE = "OFFLINE DEMO: canned replies, not real Gemini output"


def offline_reply(prompt, feedback):
    label = dict(FEEDBACK).get(feedback, "neutral")
    if "step by step" in prompt:
        return f"Good points: fast, friendly. Bad points: none.\nFinal label: {label}"
    return label


def gemini_reply(client, prompt):
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


# Step 6: Pull the label out of the reply. The last label word wins, so the
# "Final label:" line of a chain-of-thought reply is used.
def extract_label(text):
    matches = re.findall(r"\b(positive|negative|neutral)\b", text, flags=re.IGNORECASE)
    return matches[-1].lower() if matches else "unclear"


def main():
    offline = "--offline" in sys.argv
    client = None

    if not offline:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your-key-here":
            print("No Gemini API key found.")
            print("1. Create a free key in Google AI Studio (aistudio.google.com, 'Get API key').")
            print("2. Copy .env.example to .env and paste your key after GEMINI_API_KEY=")
            print("3. Run this script again.")
            print("To see the structure without a key, run: python prompt_strategies.py --offline")
            return 0
        client = genai.Client(api_key=api_key)

    if client:
        print(f"Model: {MODEL_NAME}")
    else:
        print(f"Mode: {OFFLINE_NOTE}")
    print()

    header = f"{'Feedback':<38} {'Expected':<9} {'Zero-shot':<10} {'Few-shot':<10} {'CoT':<10}"
    print(header)
    print("-" * len(header))

    for feedback, expected in FEEDBACK:
        labels = []
        for build_prompt in (zero_shot_prompt, few_shot_prompt, chain_of_thought_prompt):
            prompt = build_prompt(feedback)
            if client:
                try:
                    text = gemini_reply(client, prompt)
                except Exception as error:  # keep going if one call fails
                    print(f"  Call failed: {error}")
                    text = ""
            else:
                text = offline_reply(prompt, feedback)
            labels.append(extract_label(text))

        short = feedback if len(feedback) <= 35 else feedback[:32] + "..."
        print(f"{short:<38} {expected:<9} {labels[0]:<10} {labels[1]:<10} {labels[2]:<10}")

    print()
    print("CoT = chain-of-thought. Compare the columns with the Expected column.")
    if not client:
        print(OFFLINE_NOTE + ". Nothing here shows how Gemini would answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
