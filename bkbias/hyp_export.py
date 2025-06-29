"""Utilities to export Popper hypotheses to Prolog files."""
from pathlib import Path
from typing import Iterable, Any

try:
    from popper.util import format_rule
except Exception:
    format_rule = None  # type: ignore


def literal_to_prolog(lit: Any) -> str:
    """Return ``lit`` formatted as a Prolog literal."""
    pred = getattr(lit, "predicate", None)
    args = getattr(lit, "arguments", None)
    if pred is None or args is None:
        # fallback to tuple representation
        pred, args = lit
    return f"{pred}({','.join(map(str, args))})"


def dump_hypothesis(hyps: Iterable[Any], out_file: Path) -> None:
    """Write Popper hypothesis ``hyps`` to ``out_file`` in Prolog format."""
    with out_file.open("w") as f:
        for rule in hyps:
            if format_rule is not None:
                line = format_rule(rule)
            else:
                head, body = rule
                head_str = literal_to_prolog(head) if head else ""
                body_strs = [literal_to_prolog(b) for b in body]
                line = f"{head_str} :- {', '.join(body_strs)}."
            f.write(line.strip() + "\n")
