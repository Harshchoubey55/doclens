"""Compare every predictions/<method>_test.jsonl against gt_test.jsonl on the documents all
methods share, and write results/COMPARISON.md.

    python scripts/compare.py predictions results
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doclens.metrics import evaluate  # noqa: E402
from doclens.predictions import read_jsonl  # noqa: E402

NAMES = {
    "layoutlmv3": "LayoutLMv3-base fine-tuned (token classification, uses CORD's OCR tokens)",
    "vlm_zeroshot": "Qwen2.5-VL-3B 4-bit, zero-shot",
    "vlm_qlora": "Qwen2.5-VL-3B 4-bit + QLoRA (1 epoch)",
    "llm_zeroshot": "Gemini, zero-shot + JSON schema",
    "llm_fewshot": "Gemini, 2-shot + JSON schema",
}


def main(pred_dir: str = "predictions", out_dir: str = "results") -> None:
    pred_dir, out_dir = Path(pred_dir), Path(out_dir)
    gt = read_jsonl(pred_dir / "gt_test.jsonl")
    methods = {p.stem[: -len("_test")]: read_jsonl(p) for p in sorted(pred_dir.glob("*_test.jsonl")) if p.stem != "gt_test"}
    if not methods:
        sys.exit("no prediction files found")
    shared = sorted(set(gt).intersection(*[set(m) for m in methods.values()]), key=lambda d: int(d.split("-")[-1]))
    rows, report = [], {"documents": len(shared), "methods": {}}
    for name, preds in methods.items():
        s = evaluate([preds[d]["pred"] for d in shared], [gt[d]["gt"] for d in shared])
        lat = [preds[d].get("latency_s") for d in shared if preds[d].get("latency_s") is not None]
        s["median_latency_s"] = round(statistics.median(lat), 3) if lat else None
        report["methods"][name] = s
        rows.append((s["field_f1"], name, s))
    lines = [f"CORD-v2 test, {len(shared)} receipts shared by all methods. Field F1 = micro F1 over flattened "
             "(key, value) leaves; line-item F1 requires name and price to match together.", "",
             "| Method | Field F1 | Precision | Recall | Line-item F1 | Doc exact | Median latency (s) |",
             "|---|---|---|---|---|---|---|"]
    for _, name, s in sorted(rows, key=lambda r: -r[0]):
        lines.append(f"| {NAMES.get(name, name)} | {s['field_f1']:.3f} | {s['field_precision']:.3f} | "
                     f"{s['field_recall']:.3f} | {s['line_item_f1']:.3f} | {s['document_exact_match']:.3f} | "
                     f"{s['median_latency_s'] if s['median_latency_s'] is not None else '–'} |")
    keys = sorted({k for s in report["methods"].values() for k in s["per_key_f1"]})
    lines += ["", "Per-field F1:", "", "| Field | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{report['methods'][m]['per_key_f1'].get(k, 0):.2f}" for m in methods) + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
