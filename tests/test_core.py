import json
from pathlib import Path

import pytest

from doclens.cord import LABELS, decode, load_gt, words_boxes_labels
from doclens.metrics import evaluate, norm_value
from doclens.schema import parse_model_output, restrict, to_target

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "cord_gt_samples.json").read_text())


def test_restrict_handles_dict_menu_and_drops_other_keys():
    gt = load_gt(FIXTURES[0])["gt_parse"]
    r = restrict(gt)
    assert r["menu"] == [{"nm": "-TICKET CP", "cnt": "2", "price": "60.000"}]
    assert r["sub_total"] == {"subtotal_price": "60.000", "tax_price": "5.455", "discount_price": "-60.000"}
    assert "menuqty_cnt" not in r["total"]


@pytest.mark.parametrize("raw", [0, 1])
def test_bio_labels_decode_back_to_ground_truth(raw):
    gt = load_gt(FIXTURES[raw])
    words, boxes, labels = words_boxes_labels(gt)
    assert len(words) == len(boxes) == len(labels)
    assert all(l in LABELS for l in labels)
    assert all(0 <= v <= 1000 for b in boxes for v in b)
    score = evaluate([decode(words, labels)], [gt["gt_parse"]])
    assert score["field_f1"] == 1.0, score


def test_parse_model_output_recovers_fenced_and_truncated_json():
    fenced = '```json\n{"menu":[{"nm":"TEA","price":"5.000"}],"total":{"total_price":"5.000"},}\n```'
    assert parse_model_output(fenced) == {"menu": [{"nm": "TEA", "price": "5.000"}], "total": {"total_price": "5.000"}}
    truncated = '{"menu":[{"nm":"TEA","price":"5.000"},{"nm":"COF'
    assert parse_model_output(truncated)["menu"][0] == {"nm": "TEA", "price": "5.000"}
    assert parse_model_output("no json here") == {"menu": []}


def test_target_round_trip():
    gt = load_gt(FIXTURES[1])["gt_parse"]
    assert parse_model_output(to_target(gt)) == restrict(gt)


def test_metrics_normalise_prices_and_count_partial_credit():
    assert norm_value("price", "60.000") == norm_value("price", "60,000") == "60000"
    assert norm_value("nm", " j.stb  promo ") == "J.STB PROMO"
    gt = {"menu": [{"nm": "A", "price": "1.000"}, {"nm": "B", "price": "2.000"}], "total": {"total_price": "3.000"}}
    pred = {"menu": [{"nm": "A", "price": "1,000"}, {"nm": "B", "price": "9.000"}], "total": {"total_price": "3000"}}
    s = evaluate([pred], [gt])
    assert s["field_f1"] == 0.8  # 4 of 5 leaves right
    assert s["line_item_f1"] == 0.5
    assert s["per_key_f1"]["menu.price"] == 0.5
