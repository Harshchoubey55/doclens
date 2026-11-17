"""FastAPI service: extract receipts and answer questions about them.

    uvicorn api.main:app --reload
    curl -F file=@receipt.jpg "http://127.0.0.1:8000/extract?method=llm"
    curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Average total per store?"}'

method=llm needs GOOGLE_API_KEY; method=vlm needs a GPU (+ VLM_ADAPTER for the fine-tuned model).
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doclens.store import Store, items, receipts  # noqa: E402


class AskRequest(BaseModel):
    question: str
    thread_id: str = "default"


def create_app(store: Optional[Store] = None, extractors: Optional[Dict[str, Callable]] = None, agent=None) -> FastAPI:
    app = FastAPI(title="DocLens", version="0.1.0")
    state = {"store": store, "agent": agent, "extractors": dict(extractors or {})}

    def get_store() -> Store:
        if state["store"] is None:
            state["store"] = Store()
        return state["store"]

    def get_extractor(method: str):
        if method not in state["extractors"]:
            if method == "llm":
                from doclens.extractors import GeminiExtractor

                state["extractors"][method] = GeminiExtractor()
            elif method == "vlm":
                from doclens.extractors import VLMExtractor

                state["extractors"][method] = VLMExtractor()
            else:
                raise HTTPException(400, f"unknown method {method!r}; use llm or vlm")
        return state["extractors"][method]

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded_extractors": sorted(state["extractors"])}

    @app.post("/extract")
    async def extract(file: UploadFile = File(...), method: str = Query("llm")):
        data = await file.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(400, "file is not an image")
        extractor = get_extractor(method)
        t0 = time.perf_counter()
        parse = extractor.extract(img)
        latency = time.perf_counter() - t0
        doc_id = "api-" + hashlib.sha256(data).hexdigest()[:16]
        rid = get_store().add_receipt(doc_id, parse, source=f"api:{method}")
        return {"receipt_id": rid, "doc_id": doc_id, "method": method, "latency_s": round(latency, 3), "data": parse}

    @app.get("/receipts/{rid}")
    def get_receipt(rid: int):
        s = get_store()
        with s.engine.connect() as c:
            r = c.execute(select(receipts).where(receipts.c.id == rid)).mappings().first()
            if r is None:
                raise HTTPException(404, "not found")
            its = c.execute(select(items).where(items.c.receipt_id == rid)).mappings().all()
        return {**{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()},
                "items": [dict(i) for i in its]}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        from doclens.agent import ask, build_agent

        if state["agent"] is None:
            state["agent"] = build_agent(get_store())
        return ask(state["agent"], req.question, thread_id=req.thread_id)

    return app


app = create_app()
