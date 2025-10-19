"""
Simplified IE reward for patent JSON extraction.

Total reward = w1*validity + w2*mean(field_sims) + w3*constraints + w4*format

- validity (1/0): parsed JSON with all required keys
- field_sims (0..1): per-field normalized string similarity (fast)
- constraints (1/0): basic date ordering if dates exist
- format (0/0.1): exactly expected keys and ISO dates if present

Minimal helpers, fast defaults, no jsonschema.
"""

import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


RELEVANT_FIELDS: List[str] = [
    "publication_number",
    "application_number",
    "patent_number",
    "date_published",
    "filing_date",
    "patent_issue_date",
    "abandon_date",
    "decision",
    "main_cpc_label",
    "main_ipcr_label",
    "title",
    "abstract",
    "summary",
    "claims",
]


from typing import Optional


def _first_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        return None
    return None


def _norm(s: Any, max_len: int = 4000) -> str:
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = "\n".join(map(str, s))
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _parse_date(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            continue
    return None


def compute_reward(
    model_output_text: str,
    gold: Dict[str, Any],
    weights: Tuple[float, float, float, float] = (0.5, 0.4, 0.1, 0.05),
) -> Tuple[float, Dict[str, float]]:
    pred = _first_json(model_output_text)
    validity = int(isinstance(pred, dict) and set(RELEVANT_FIELDS).issubset(set(pred.keys())))

    # Field-level similarity (only where gold is non-empty)
    sims: Dict[str, float] = {}
    use_fields: List[str] = []
    if isinstance(pred, dict):
        for k in RELEVANT_FIELDS:
            g = gold.get(k)
            if g is None or (isinstance(g, str) and g.strip() == ""):
                sims[k] = 0.0
                continue
            p = pred.get(k)
            score = _sim(_norm(g), _norm(p))
            sims[k] = float(score)
            use_fields.append(k)
    field_mean = sum(sims.get(k, 0.0) for k in use_fields) / max(1, len(use_fields))

    # Constraints: dates in order if present
    constraints = 0
    if isinstance(pred, dict):
        fd = _parse_date(pred.get("filing_date"))
        pd = _parse_date(pred.get("date_published"))
        id_ = _parse_date(pred.get("patent_issue_date"))
        ok = True
        if fd and pd:
            ok = ok and (fd <= pd)
        if fd and id_:
            ok = ok and (fd <= id_)
        if pd and id_:
            ok = ok and (pd <= id_)
        constraints = int(ok)

    # Format bonus: exact keys + ISO dates if present
    fmt = 0.0
    if isinstance(pred, dict) and set(pred.keys()) == set(RELEVANT_FIELDS):
        iso_ok = True
        for k in ("filing_date", "date_published", "patent_issue_date", "abandon_date"):
            v = pred.get(k)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            try:
                datetime.strptime(str(v).strip(), "%Y-%m-%d")
            except Exception:
                iso_ok = False
                break
        if iso_ok:
            fmt = 0.1

    w1, w2, w3, w4 = weights
    total = w1 * validity + w2 * field_mean + w3 * constraints + w4 * fmt
    total = max(0.0, min(1.0, float(total)))

    return total, {
        "validity": float(validity),
        "field_mean": float(field_mean),
        "constraints": float(constraints),
        "format": float(fmt),
    }
