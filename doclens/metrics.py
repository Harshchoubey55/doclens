"""Field-level F1 (Donut-style flattened leaves) plus a pairing-aware line-item metric."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from doclens.schema import restrict


def norm_value(key: str, value: str) -> str:
    s = str(value).strip()
    if key.endswith("price") or key.endswith("cnt"):
        neg = s.startswith("-")
        digits = re.sub(r"[^0-9]", "", s)
        return ("-" if neg else "") + digits.lstrip("0") or "0"
    return re.sub(r"\s+", " ", s).upper()


def flatten(parse: dict) -> List[Tuple[str, str]]:
    d = restrict(parse)
    pairs = []
    for item in d.get("menu", []):
        for k, v in item.items():
            pairs.append((f"menu.{k}", norm_value(k, v)))
    for section in ("sub_total", "total"):
        for k, v in (d.get(section) or {}).items():
            pairs.append((f"{section}.{k}", norm_value(k, v)))
    return pairs


def items(parse: dict) -> List[Tuple[str, str]]:
    d = restrict(parse)
    return [(norm_value("nm", i.get("nm", "")), norm_value("price", i.get("price", ""))) for i in d.get("menu", [])]


def _prf(tp: int, n_pred: int, n_gt: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def evaluate(preds: Iterable[dict], gts: Iterable[dict]) -> dict:
    tp = n_pred = n_gt = 0
    it_tp = it_pred = it_gt = 0
    exact = docs = 0
    per_key: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])  # tp, pred, gt
    for pred, gt in zip(preds, gts):
        p, g = Counter(flatten(pred)), Counter(flatten(gt))
        inter = p & g
        tp += sum(inter.values())
        n_pred += sum(p.values())
        n_gt += sum(g.values())
        for (k, _), c in inter.items():
            per_key[k][0] += c
        for (k, _), c in p.items():
            per_key[k][1] += c
        for (k, _), c in g.items():
            per_key[k][2] += c
        pi, gi = Counter(items(pred)), Counter(items(gt))
        it_tp += sum((pi & gi).values())
        it_pred += sum(pi.values())
        it_gt += sum(gi.values())
        exact += p == g
        docs += 1
    out = {"documents": docs, **{f"field_{k}": v for k, v in _prf(tp, n_pred, n_gt).items()},
           "line_item_f1": _prf(it_tp, it_pred, it_gt)["f1"],
           "document_exact_match": round(exact / docs, 4) if docs else 0.0,
           "per_key_f1": {k: _prf(*v)["f1"] for k, v in sorted(per_key.items())}}
    return out
