"""Tools the agent can call. Each tool is a plain Python function.

The docstring of each function is shown to the model as its description,
so keep it short and clear. TOOLS maps the name the model uses to the function.
"""

import inspect

# A small fake price list. No real shop is contacted.
PRICES = {"laptop": 999.0, "phone": 499.0, "headphones": 89.5}


def add_numbers(a, b):
    """Add two numbers and return the sum."""
    return float(a) + float(b)


def get_word_count(text):
    """Count the words in a piece of text and return the number."""
    return len(str(text).split())


def lookup_price(item):
    """Look up the price of an item in dollars. Known items: laptop, phone, headphones."""
    key = str(item).strip().lower()
    if key not in PRICES:
        raise ValueError(f"no price for '{item}'")
    return PRICES[key]


TOOLS = {
    "add_numbers": add_numbers,
    "get_word_count": get_word_count,
    "lookup_price": lookup_price,
}


def describe_tools():
    """Return one line per tool: its name, its arguments, and its docstring."""
    lines = []
    for name, func in TOOLS.items():
        lines.append(f"- {name}{inspect.signature(func)}: {func.__doc__}")
    return "\n".join(lines)
