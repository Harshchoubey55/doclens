"""Relational store for extracted receipts + semantic search over line items.

SQLite by default. With DATABASE_URL=postgresql+psycopg://... and the pgvector extension,
item embeddings live in a vector(384) column and similarity search runs in SQL.

CORD receipts carry no store name or date, so `store` and `purchase_date` are *simulated*
(deterministic from the receipt id) to make time/vendor questions possible in the demo.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import date, timedelta
from typing import Callable, List, Optional, Sequence

import numpy as np
from sqlalchemy import (Column, Date, Float, ForeignKey, Integer, MetaData, String, Table, create_engine, insert,
                        select, text)
from sqlalchemy.engine import Engine

from doclens.schema import restrict

metadata = MetaData()
receipts = Table(
    "receipts", metadata,
    Column("id", Integer, primary_key=True),
    Column("doc_id", String(64), unique=True),
    Column("source", String(32)),          # ground_truth | layoutlmv3 | vlm_qlora | llm | api
    Column("store", String(64)),           # simulated
    Column("purchase_date", Date),         # simulated
    Column("subtotal", Float), Column("tax", Float), Column("service", Float), Column("discount", Float),
    Column("total", Float), Column("cash", Float), Column("change", Float), Column("card", Float),
)
items = Table(
    "items", metadata,
    Column("id", Integer, primary_key=True),
    Column("receipt_id", Integer, ForeignKey("receipts.id")),
    Column("name", String(255)),
    Column("qty", Float),
    Column("price", Float),
)

SCHEMA_DOC = """Tables (amounts are in Indonesian rupiah, the currency of the CORD receipts):
receipts(id, doc_id, source, store, purchase_date DATE, subtotal, tax, service, discount, total, cash, change, card)
  -- store and purchase_date are simulated; card = amount paid by credit card
items(id, receipt_id -> receipts.id, name, qty, price)  -- price is the line total for that item"""

STORES = ["Kopi Senja", "Warung Nusantara", "Bakmi Ceria", "Roti Pagi Bakery", "Sate Merdeka",
          "Teh Tarik House", "Nasi Uduk 88", "Dimsum Kita"]


def parse_amount(s: Optional[str]) -> Optional[float]:
    """CORD style amounts: '60.000' and '60,000' mean 60000, '2.00' means 2, '-5.000' is negative.
    A final separator followed by 1-2 digits is a decimal point; every other one is a thousands mark."""
    if not s:
        return None
    neg = s.strip().startswith("-")
    m = re.sub(r"[^0-9.,]", "", s)
    if not re.search(r"\d", m):
        return None
    dec = re.search(r"[.,](\d{1,2})$", m)
    whole = re.sub(r"[.,]", "", m[: dec.start()] if dec else m) or "0"
    v = float(whole) + (float("0." + dec.group(1)) if dec else 0.0)
    return -v if neg else v


def _simulated(doc_id: str):
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    return STORES[h % len(STORES)], date(2025, 1, 1) + timedelta(days=h % 365)


class Embedder:
    """sentence-transformers MiniLM (384-d). Swappable for tests."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class Store:
    def __init__(self, url: Optional[str] = None, embedder: Optional[Callable] = None):
        self.url = url or os.getenv("DATABASE_URL", "sqlite:///doclens.db")
        self.engine: Engine = create_engine(self.url, future=True)
        self.is_pg = self.engine.dialect.name == "postgresql"
        metadata.create_all(self.engine)
        self._embedder = embedder
        self._matrix: Optional[np.ndarray] = None
        self._item_ids: List[int] = []
        if self.is_pg:
            with self.engine.begin() as c:
                c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                c.execute(text("ALTER TABLE items ADD COLUMN IF NOT EXISTS embedding vector(384)"))

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    # ---------------------------------------------------------------- writes
    def add_receipt(self, doc_id: str, parse: dict, source: str) -> int:
        d = restrict(parse)
        st, dt = _simulated(doc_id)
        sub, tot = d.get("sub_total", {}), d.get("total", {})
        row = dict(doc_id=doc_id, source=source, store=st, purchase_date=dt,
                   subtotal=parse_amount(sub.get("subtotal_price")), tax=parse_amount(sub.get("tax_price")),
                   service=parse_amount(sub.get("service_price")), discount=parse_amount(sub.get("discount_price")),
                   total=parse_amount(tot.get("total_price")), cash=parse_amount(tot.get("cashprice")),
                   change=parse_amount(tot.get("changeprice")), card=parse_amount(tot.get("creditcardprice")))
        with self.engine.begin() as c:
            c.execute(receipts.delete().where(receipts.c.doc_id == doc_id))
            rid = c.execute(insert(receipts).values(**row)).inserted_primary_key[0]
            for it in d.get("menu", []):
                qty = parse_amount(it.get("cnt"))
                c.execute(insert(items).values(receipt_id=rid, name=it.get("nm") or "", qty=qty or 1.0,
                                               price=parse_amount(it.get("price"))))
        self._matrix = None
        return rid

    def build_index(self) -> int:
        """Embed all item names (pgvector column on Postgres, in-memory matrix otherwise)."""
        with self.engine.connect() as c:
            rows = c.execute(select(items.c.id, items.c.name)).all()
        if not rows:
            return 0
        vecs = self.embedder([r.name for r in rows])
        if self.is_pg:
            with self.engine.begin() as c:
                for r, v in zip(rows, vecs):
                    c.execute(text("UPDATE items SET embedding = CAST(:v AS vector) WHERE id = :id"),
                              {"v": "[" + ",".join(f"{x:.6f}" for x in v) + "]", "id": r.id})
        self._item_ids, self._matrix = [r.id for r in rows], vecs
        return len(rows)

    # ---------------------------------------------------------------- reads
    def search_items(self, query: str, k: int = 5) -> List[dict]:
        q = self.embedder([query])[0]
        if self.is_pg:
            sql = text("""SELECT i.id, i.name, i.qty, i.price, r.store, r.purchase_date,
                                 1 - (i.embedding <=> CAST(:q AS vector)) AS score
                          FROM items i JOIN receipts r ON r.id = i.receipt_id
                          WHERE i.embedding IS NOT NULL ORDER BY i.embedding <=> CAST(:q AS vector) LIMIT :k""")
            with self.engine.connect() as c:
                rows = c.execute(sql, {"q": "[" + ",".join(f"{x:.6f}" for x in q) + "]", "k": k}).mappings().all()
            return [dict(r) for r in rows]
        if self._matrix is None:
            self.build_index()
        if self._matrix is None:
            return []
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        ids = [self._item_ids[i] for i in top]
        with self.engine.connect() as c:
            rows = {r.id: r for r in c.execute(
                select(items.c.id, items.c.name, items.c.qty, items.c.price, receipts.c.store, receipts.c.purchase_date)
                .join(receipts, receipts.c.id == items.c.receipt_id).where(items.c.id.in_(ids))).all()}
        return [{**rows[i]._asdict(), "score": round(float(scores[j]), 4)} for i, j in zip(ids, top) if i in rows]

    def read_only_query(self, sql: str, max_rows: int = 200) -> List[dict]:
        safe = guard_sql(sql, max_rows)
        with self.engine.connect() as c:
            if self.is_pg:
                c.execute(text("SET TRANSACTION READ ONLY"))
            elif self.engine.dialect.name == "sqlite":
                c.exec_driver_sql("PRAGMA query_only = ON")
            try:
                rows = c.execute(text(safe)).mappings().all()
            finally:
                if self.engine.dialect.name == "sqlite":
                    c.exec_driver_sql("PRAGMA query_only = OFF")
            return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()} for r in rows]


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|grant|"
                        r"revoke|copy|vacuum|reindex|call|execute)\b", re.IGNORECASE)


class UnsafeSQL(ValueError):
    pass


def guard_sql(sql: str, max_rows: int = 200) -> str:
    """Allow one read-only SELECT/WITH statement; add a LIMIT if missing."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQL("empty query")
    if ";" in s:
        raise UnsafeSQL("only one statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise UnsafeSQL("only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise UnsafeSQL("query contains a forbidden keyword")
    if not re.search(r"\blimit\s+\d+\s*$", s, re.IGNORECASE):
        s = f"{s} LIMIT {max_rows}"
    return s
