# Mirrors capgym/cap-x capx/integrations/base_api.py (MIT, Copyright (c) 2026 Max Fu).
"""Local copy of the public API documentation contract for isolated recorder tests."""
import inspect


def combined_doc(api):
    lines = []
    for name, fn in api.functions().items():
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(…)"
        doc = inspect.getdoc(fn) or ""
        lines.append(f"{name}{sig}")
        if doc:
            lines.append("  Doc:")
            lines.extend(f"    {line}" for line in doc.splitlines())
        lines.append("")
    return "\n".join(lines).strip()
