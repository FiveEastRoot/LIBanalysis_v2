import re


def remove_parentheses(text: str) -> str:
    """Remove content inside parentheses from text and strip whitespace."""
    return re.sub(r"\(.*?\)", "", text).strip()


def wrap_label(label: str, width: int = 10) -> str:
    """Wrap a label string every ``width`` characters with ``<br>``."""
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])
