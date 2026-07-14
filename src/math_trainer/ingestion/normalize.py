"""Text normalization shared by Cleaner-adjacent logic and the Embedder.

Deliberately conservative: collapse redundant whitespace and unify a few common
math-delimiter variants so equivalent content produces equivalent embeddings.
"""

import re

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize_math(text: str | None) -> str:
    if not text:
        return ""
    out = text.replace("\\(", "$").replace("\\)", "$")
    out = out.replace("\\[", "$$").replace("\\]", "$$")
    out = _WS.sub(" ", out)
    out = _BLANKS.sub("\n\n", out)
    return out.strip()
