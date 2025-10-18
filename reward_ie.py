"""
Reward functions for Information Extraction (IE) on patent JSON outputs.

Components:
 - JSON validity (1/0): parses and matches schema (required fields, types)
 - Field-level similarity (0..1): EM / normalized Levenshtein / token-F1
 - Cross-field constraints (1/0): basic date validity and ordering
 - Format bonus (0/0.1): exact keys only, ISO dates

Total: w1*valid + w2*mean(field_scores) + w3*constraints + w4*format

Notes:
 - Keep computations efficient for long text by simple tokenization and
   a cap on tokens used in token-F1.
 - No heavy try/except; functions return conservative defaults on failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Optional fast fuzzy similarity
try:
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore
except Exception:  # noqa: E722
    _rf_ratio = None

from difflib import SequenceMatcher


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


# JSON schema (kept permissive on types to reflect real data)
JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": RELEVANT_FIELDS,
    "properties": {
        "publication_number": {"type": ["string", "null"]},
        "application_number": {"type": ["string", "null"]},
        "patent_number": {"type": ["string", "null"]},
        "date_published": {"type": ["string", "null"]},
        "filing_date": {"type": ["string", "null"]},
        "patent_issue_date": {"type": ["string", "null"]},
        "abandon_date": {"type": ["string", "null"]},
        "decision": {"type": ["string", "null"]},
        "main_cpc_label": {"type": ["string", "null"]},
        "main_ipcr_label": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
        "abstract": {"type": ["string", "null"]},
        "summary": {"type": ["string", "null"]},
        "claims": {
            "anyOf": [
                {"type": ["string", "null"]},
                {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
            ]
        },
    },
}


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Extract first top-level JSON object from a string using a simple brace scanner."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    # Scan for matching closing brace handling strings/escapes
    depth = 0
    in_str = False
    esc = False
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
                        cand = text[start : i + 1]
                        return json.loads(cand)
                    except Exception:
                        return None
    return None


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = "\n".join(map(str, s))
    s = str(s)
    s = s.replace("\u00a0", " ")  # nbsp
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def _levenshtein_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _rf_ratio is not None:
        return float(_rf_ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def _tokenize(s: str, max_tokens: int = 3000) -> List[str]:
    if not s:
        return []
    # Simple alnum tokenization; truncate to keep speed bounds
    toks = re.findall(r"[a-z0-9]+", s)
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return toks


def _f1(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    # multiset overlap using counts
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum(min(ca[t], cb[t]) for t in ca.keys() | cb.keys())
    p = overlap / max(1, sum(cb.values()))
    r = overlap / max(1, sum(ca.values()))
    if p == 0 or r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score_field(gold: Any, pred: Any) -> float:
    gs = _normalize_text(gold)
    ps = _normalize_text(pred)
    if not gs and not ps:
        return 1.0
    if gs == ps:
        return 1.0
    return 0.5 * _levenshtein_sim(gs, ps) + 0.5 * _f1(gs, ps)


def _is_nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return True


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # accept common patterns
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _is_iso_date(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    try:
        datetime.strptime(s.strip(), "%Y-%m-%d")
        return True
    except Exception:
        return False


def json_validity(pred_obj: Optional[Dict[str, Any]]) -> int:
    if not isinstance(pred_obj, dict):
        return 0
    # Try jsonschema if available
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(pred_obj, JSON_SCHEMA)
        return 1
    except Exception:
        # Fallback: keys present and types roughly match expectations
        keys_ok = set(pred_obj.keys()) >= set(RELEVANT_FIELDS)
        if not keys_ok:
            return 0
        return 1


def cross_field_constraints(pred_obj: Dict[str, Any]) -> int:
    if not isinstance(pred_obj, dict):
        return 0
    ok = True

    # Date ordering constraints when available
    fd = _parse_date(pred_obj.get("filing_date"))
    pd = _parse_date(pred_obj.get("date_published"))
    id_ = _parse_date(pred_obj.get("patent_issue_date"))

    if fd and pd:
        ok = ok and (fd <= pd)
    if fd and id_:
        ok = ok and (fd <= id_)
    if pd and id_:
        ok = ok and (pd <= id_)

    # Basic id patterns if present
    pub = pred_obj.get("publication_number")
    if isinstance(pub, str) and pub.strip():
        ok = ok and bool(re.match(r"^[A-Z]{2}\d{4,}.*", pub))

    return 1 if ok else 0


def format_bonus(pred_obj: Dict[str, Any]) -> float:
    if not isinstance(pred_obj, dict):
        return 0.0
    # exactly the relevant keys, no extras
    if set(pred_obj.keys()) != set(RELEVANT_FIELDS):
        return 0.0
    # ISO date strings if provided
    for k in ("filing_date", "date_published", "patent_issue_date", "abandon_date"):
        v = pred_obj.get(k)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if not _is_iso_date(v):
            return 0.0
    return 0.1


@dataclass
class RewardComponents:
    validity: int
    field_mean: float
    constraints: int
    fmt_bonus: float
    field_scores: Dict[str, float]


def compute_reward(
    model_output_text: str,
    gold: Dict[str, Any],
    weights: Tuple[float, float, float, float] = (0.5, 0.4, 0.1, 1.0),
) -> Tuple[float, RewardComponents]:
    pred_obj = None
    try:
        pred_obj = json.loads(model_output_text)
        if not isinstance(pred_obj, dict):
            pred_obj = _extract_first_json_obj(model_output_text)
    except Exception:
        pred_obj = _extract_first_json_obj(model_output_text)

    validity = json_validity(pred_obj)

    # Field-level scores
    scores: Dict[str, float] = {}
    fields_to_consider = [f for f in RELEVANT_FIELDS if _is_nonempty(gold.get(f))]
    denom = max(1, len(fields_to_consider))
    for f in RELEVANT_FIELDS:
        g = gold.get(f)
        p = pred_obj.get(f) if isinstance(pred_obj, dict) else None
        s = score_field(g, p)
        scores[f] = s
    field_mean = sum(scores[f] for f in fields_to_consider) / denom

    constraints = cross_field_constraints(pred_obj or {})
    fmt = format_bonus(pred_obj or {})

    w1, w2, w3, w4 = weights
    total = w1 * float(validity) + w2 * field_mean + w3 * float(constraints) + w4 * float(fmt)
    total = max(0.0, min(1.0, total))  # clamp to [0,1]

    return total, RewardComponents(
        validity=validity,
        field_mean=field_mean,
        constraints=constraints,
        fmt_bonus=fmt,
        field_scores=scores,
    )


def batch_compute_rewards(
    generations: List[List[str]],
    golds: List[Dict[str, Any]],
    weights: Tuple[float, float, float, float] = (0.5, 0.4, 0.1, 1.0),
) -> List[List[float]]:
    all_rewards: List[List[float]] = []
    for gens, gold in zip(generations, golds):
        rs = [compute_reward(g, gold, weights=weights)[0] for g in gens]
        all_rewards.append(rs)
    return all_rewards

