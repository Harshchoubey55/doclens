"""CORD-v2 helpers: token-classification labels for LayoutLMv3 and decoding predictions
back into the receipt schema."""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple, Union

from doclens.schema import MENU_KEYS, SECTIONS

DATASET_ID = "naver-clova-ix/cord-v2"

ENTITY_TYPES = [f"menu.{k}" for k in MENU_KEYS] + [f"{s}.{k}" for s, keys in SECTIONS.items() for k in keys]
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_gt(ground_truth: Union[str, dict]) -> dict:
    return json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth


def words_boxes_labels(ground_truth: Union[str, dict]) -> Tuple[List[str], List[List[int]], List[str]]:
    """Words in reading order, boxes normalised to 0-1000 (LayoutLM convention), BIO labels."""
    gt = load_gt(ground_truth)
    size = gt["meta"]["image_size"]
    w_img, h_img = size["width"], size["height"]
    entries = []
    for line_idx, line in enumerate(gt["valid_line"]):
        cat = line["category"]
        ent = cat if cat in ENTITY_TYPES else None
        first_value = True
        for j, w in enumerate(line["words"]):
            text = (w.get("text") or "").strip()
            if not text:
                continue
            # CORD puts printed keys ("TOTAL", "TAX") in the same line as the value, flagged is_key=1
            is_value = ent is not None and not w.get("is_key", 0)
            q = w["quad"]
            x0, x1 = min(q["x1"], q["x4"]), max(q["x2"], q["x3"])
            y0, y1 = min(q["y1"], q["y2"]), max(q["y3"], q["y4"])
            box = [
                max(0, min(1000, int(1000 * x0 / w_img))),
                max(0, min(1000, int(1000 * y0 / h_img))),
                max(0, min(1000, int(1000 * x1 / w_img))),
                max(0, min(1000, int(1000 * y1 / h_img))),
            ]
            if is_value:
                label = ("B-" if first_value else "I-") + ent
                first_value = False
            else:
                label = "O"
            entries.append((line_idx, j, text, box, label))
    # reading order: each annotated line (one field) as a unit, sorted top-to-bottom in
    # ~8/1000-high bands, then left-to-right, so every labelled span stays contiguous
    groups: Dict[int, list] = {}
    for e in entries:
        groups.setdefault(e[0], []).append(e)
    ordered = sorted(groups.values(), key=lambda g: (min(e[3][1] for e in g) // 8, min(e[3][0] for e in g)))
    words, boxes, labels = [], [], []
    for g in ordered:
        for _, _, text, box, label in g:
            words.append(text)
            boxes.append(box)
            labels.append(label)
    return words, boxes, labels


def decode(words: Sequence[str], labels: Sequence[str]) -> dict:
    """BIO word labels -> receipt dict. A new line item starts when a menu field repeats."""
    spans: List[Tuple[str, List[str]]] = []
    for word, label in zip(words, labels):
        if label == "O":
            continue
        tag, ent = label.split("-", 1)
        if tag == "B" or not spans or spans[-1][0] != ent:
            spans.append((ent, [word]))
        else:
            spans[-1][1].append(word)

    out: dict = {"menu": []}
    current: Dict[str, str] = {}
    for ent, toks in spans:
        section, key = ent.split(".", 1)
        text = " ".join(toks)
        if section == "menu":
            if key in current:
                out["menu"].append(current)
                current = {}
            current[key] = text
        else:
            out.setdefault(section, {}).setdefault(key, text)
    if current:
        out["menu"].append(current)
    return out
