"""Store, SQL guard, agent loop and API — with a fake embedder and a scripted fake LLM (no network)."""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from PIL import Image

from api.main import create_app
from doclens.agent import ask, build_agent
from doclens.store import Store, UnsafeSQL, guard_sql, parse_amount


def fake_embedder(texts):
    """Bag-of-characters vectors: enough to make 'teh' close to 'ICE TEH MANIS'."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for ch in t.lower():
            out[i, ord(ch) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture()
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'r.db'}", embedder=fake_embedder)
    s.add_receipt("r1", {"menu": [{"nm": "ICE TEH MANIS", "cnt": "2", "price": "10.000"},
                                  {"nm": "NASI GORENG", "cnt": "1", "price": "25.000"}],
                         "total": {"total_price": "35.000", "cashprice": "50.000", "changeprice": "15.000"}}, "ground_truth")
    s.add_receipt("r2", {"menu": [{"nm": "KOPI SUSU", "price": "18.000"}], "total": {"total_price": "18.000"}}, "ground_truth")
    return s


def test_parse_amount():
    assert parse_amount("60.000") == 60000
    assert parse_amount("60,000") == 60000
    assert parse_amount("-5.000") == -5000
    assert parse_amount("2.00") == 2
    assert parse_amount("x") is None


@pytest.mark.parametrize("bad", ["DELETE FROM receipts", "select 1; drop table items", "PRAGMA table_info(items)",
                                 "WITH x AS (SELECT 1) INSERT INTO items VALUES (1)", ""])
def test_guard_rejects_writes(bad):
    with pytest.raises(UnsafeSQL):
        guard_sql(bad)


def test_guard_adds_limit():
    assert guard_sql("SELECT * FROM items").endswith("LIMIT 200")
    assert guard_sql("select * from items limit 5") == "select * from items limit 5"


def test_queries_and_semantic_search(store):
    rows = store.read_only_query("SELECT SUM(total) AS t FROM receipts")
    assert rows == [{"t": 53000.0}]
    hits = store.search_items("teh manis", k=1)
    assert hits[0]["name"] == "ICE TEH MANIS"


def test_read_only_connection_blocks_writes_even_past_the_guard(store):
    with store.engine.connect() as c:
        c.exec_driver_sql("PRAGMA query_only = ON")
        with pytest.raises(Exception):
            c.exec_driver_sql("DELETE FROM items")


class ScriptedLLM:
    """Minimal chat model: first turn calls run_sql, second turn answers from the tool output."""

    def __init__(self):
        self.turn = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.turn += 1
        if self.turn % 2 == 1:
            return AIMessage(content="", tool_calls=[{"name": "run_sql", "id": f"c{self.turn}",
                                                      "args": {"query": "SELECT COUNT(*) AS n FROM receipts"}}])
        n = json.loads(messages[-1].content)[0]["n"]
        return AIMessage(content=f"There are {n} receipts.")


def test_agent_loop_uses_tools_and_keeps_thread_memory(store):
    agent = build_agent(store, llm=ScriptedLLM())
    out = ask(agent, "How many receipts?", thread_id="t1")
    assert out["answer"] == "There are 2 receipts."
    assert out["tool_calls"] == [{"tool": "run_sql", "args": {"query": "SELECT COUNT(*) AS n FROM receipts"}}]
    out2 = ask(agent, "Again?", thread_id="t1")
    assert out2["answer"] == "There are 2 receipts." and len(out2["tool_calls"]) == 1  # only new calls reported
    history = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert len(history) == 8  # 2 x (human, tool call, tool result, answer)


class FakeExtractor:
    def extract(self, img):
        return {"menu": [{"nm": "ES JERUK", "price": "12.000"}], "total": {"total_price": "12.000"}}


def test_api_extract_and_fetch(store):
    client = TestClient(create_app(store=store, extractors={"llm": FakeExtractor()}))
    buf = io.BytesIO()
    Image.new("RGB", (60, 80), "white").save(buf, format="PNG")
    r = client.post("/extract?method=llm", files={"file": ("r.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    rid = r.json()["receipt_id"]
    got = client.get(f"/receipts/{rid}").json()
    assert got["total"] == 12000 and got["items"][0]["name"] == "ES JERUK"
    assert client.post("/extract?method=nope", files={"file": ("r.png", buf.getvalue(), "image/png")}).status_code == 400
