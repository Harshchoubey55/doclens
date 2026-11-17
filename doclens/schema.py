"""The receipt schema every method is evaluated on (a subset of CORD's 30 classes),
plus tolerant parsing of model output into that schema."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

MENU_KEYS = ["nm", "cnt", "price"]
SUB_TOTAL_KEYS = ["subtotal_price", "tax_price", "service_price", "discount_price"]
TOTAL_KEYS = ["total_price", "cashprice", "changeprice", "creditcardprice"]
SECTIONS = {"sub_total": SUB_TOTAL_KEYS, "total": TOTAL_KEYS}


class MenuItem(BaseModel):
    nm: Optional[str] = None
    cnt: Optional[str] = None
    price: Optional[str] = None


class SubTotal(BaseModel):
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    service_price: Optional[str] = None
    discount_price: Optional[str] = None


class Total(BaseModel):
    total_price: Optional[str] = None
    cashprice: Optional[str] = None
    changeprice: Optional[str] = None
    creditcardprice: Optional[str] = None


class Receipt(BaseModel):
    """Used as the structured-output schema for LLM APIs."""

    menu: List[MenuItem] = []
    sub_total: Optional[SubTotal] = None
    total: Optional[Total] = None


PROMPT = (
    "Extract this receipt as JSON with exactly these keys:\n"
    '  "menu": list of line items, each {"nm": item name, "cnt": quantity, "price": line price},\n'
    '  "sub_total": {"subtotal_price", "tax_price", "service_price", "discount_price"},\n'
    '  "total": {"total_price", "cashprice", "changeprice", "creditcardprice"}.\n'
    "Copy every value exactly as printed (keep separators like 60.000). "
    "Leave out any field that is not on the receipt. Output only the JSON."
)


def _clean(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        v = " ".join(str(x) for x in v if x is not None)
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


def restrict(parse: Optional[dict]) -> dict:
    """Map a CORD gt_parse (or any model output) onto the evaluation schema."""
    parse = parse or {}
    out: Dict[str, Any] = {"menu": []}
    menu = parse.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    for item in menu:
        if not isinstance(item, dict):
            continue
        clean = {k: _clean(item.get(k)) for k in MENU_KEYS}
        clean = {k: v for k, v in clean.items() if v}
        if clean:
            out["menu"].append(clean)
    for section, keys in SECTIONS.items():
        block = parse.get(section) or {}
        if isinstance(block, list):  # CORD occasionally repeats a block
            merged: dict = {}
            for b in block:
                if isinstance(b, dict):
                    for k, v in b.items():
                        merged.setdefault(k, v)
            block = merged
        if not isinstance(block, dict):
            continue
        clean = {k: _clean(block.get(k)) for k in keys}
        clean = {k: v for k, v in clean.items() if v}
        if clean:
            out[section] = clean
    return out


def to_target(parse: dict) -> str:
    """Compact JSON string used as the fine-tuning target."""
    return json.dumps(restrict(parse), ensure_ascii=False, separators=(",", ":"))


def parse_model_output(text: str) -> dict:
    """Best-effort JSON recovery from generated text (code fences, trailing commas, truncation)."""
    if not text:
        return restrict({})
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = t.find("{")
    if start < 0:
        return restrict({})
    t = t[start:]
    for candidate in (t, t[: t.rfind("}") + 1]):
        parsed = _loads(candidate)
        if parsed is not None:
            return restrict(parsed)
    # truncated generation (hit max_new_tokens): cut back to the last complete element, close brackets
    cuts = [i for i, ch in enumerate(t) if ch in "}],"][::-1][:300]
    for i in cuts:
        prefix = t[: i + 1] if t[i] in "}]" else t[:i]
        closers = _closers(prefix)
        if closers is None:
            continue
        parsed = _loads(prefix + closers)
        if parsed is not None:
            return restrict(parsed)
    return restrict({})


def _loads(s: str):
    try:
        v = json.loads(re.sub(r",\s*([}\]])", r"\1", s))
        return v if isinstance(v, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _closers(s: str) -> Optional[str]:
    """Brackets needed to close `s`, or None if it ends inside a string."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return None if in_str else "".join(reversed(stack))
