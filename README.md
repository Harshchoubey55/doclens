# DocLens: three ways to turn a receipt photo into JSON, compared, plus an agent that queries the results

Intelligent document processing has three competing approaches. DocLens implements all three on the same
data and measures them with the same metric:

| Approach | Model | Reads | Training |
|---|---|---|---|
| Layout-aware token classification | **LayoutLMv3-base** (125M) | OCR words + 2-D boxes + image | full fine-tune, 12 epochs |
| OCR-free vision-language model | **Qwen2.5-VL-3B**, 4-bit NF4 + **QLoRA** (r=16) | pixels | 1 epoch, free Colab T4 |
| Prompted LLM API | **Gemini** with a JSON schema (structured output) | pixels | none (zero- / 2-shot) |

Extracted receipts land in SQL, and a **LangGraph** agent answers natural-language questions with
**Text-to-SQL** and **embedding search** over line items, with per-thread conversation memory.

```
receipt image -+- LayoutLMv3 (BIO tags > spans > items) --+
               +- Qwen2.5-VL + LoRA adapter (JSON) -------+-> schema-validated JSON -> SQLite / Postgres+pgvector
               +- Gemini + response_schema (JSON) --------+                                |
                                                       FastAPI  /extract   /ask <-- LangGraph agent
                                                                                   +- run_sql (read-only, guarded)
                                                                                   +- search_items (MiniLM embeddings)
```

## Data
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) has 1,000 Indonesian restaurant receipts
(800 train / 100 val / 100 test) with word boxes and a nested JSON parse. Everything is evaluated on a
fixed 11-field subset: line items (`nm`, `cnt`, `price`), `sub_total.{subtotal, tax, service, discount}_price`
and `total.{total_price, cashprice, changeprice, creditcardprice}`.

## Metrics
* **Field F1**: micro F1 over flattened `(key, value)` leaves (Donut-style). Prices are compared on their digits, so `60.000` and `60,000` match.
* **Line-item F1**: an item counts only if its name **and** price match together, so mixing up which price belongs to which item is penalised.
* **Document exact match** and **median latency per receipt**.

## Notes on the comparison
The VLM and LLM runs use the first 50 test receipts because generation is slow on a T4. `scripts/compare.py`
scores every method on the documents they all share.

* LayoutLMv3 is given CORD's *ground-truth* OCR words, as in the LayoutLMv3 paper, so it gets perfect text for
  free. The VLM and Gemini read raw pixels. A production LayoutLM pipeline would add OCR error on top.
* `notebooks/DocLens_1` prints a **decoder upper bound** (gold BIO labels -> JSON). It is below 1.0 when a
  receipt's items can't be recovered from reading order alone, which caps LayoutLMv3's score.
* LayoutLMv3 weights are CC BY-NC-SA, which is fine for research and portfolio use but not for commercial use.

## The agent
Two tools: `run_sql` for aggregates and filters, `search_items` for fuzzy item lookups over MiniLM embeddings. Conversation state is kept per thread, so follow-ups like "what about coffee?" work.

Safety: `run_sql` accepts exactly one `SELECT`/`WITH` statement, rejects DDL/DML keywords and adds a `LIMIT`.
The connection itself is also read-only (`PRAGMA query_only` on SQLite, `SET TRANSACTION READ ONLY` on
Postgres), so a prompt-injected query fails even if it gets past the guard.

CORD has no store names or dates, so `store` and `purchase_date` are **simulated** (deterministic per receipt)
so that vendor and time questions have something to work with.

## Run it

**Colab (free T4)**: each notebook clones this repo in its first cell. Run in order:
1. `notebooks/DocLens_1_LayoutLMv3.ipynb`: T4, ~30 min
2. `notebooks/DocLens_2_QLoRA_VLM.ipynb`: T4, ~75 min (zero-shot eval, QLoRA, fine-tuned eval)
3. `notebooks/DocLens_3_LLM_Compare_Agent.ipynb`: CPU, ~20 min; needs a free Gemini key in Colab Secrets as `GOOGLE_API_KEY`

Notebooks 1 and 2 need the GPU; notebook 3 can run at the same time on a CPU runtime. They share files through
`MyDrive/doclens/`. Add `WANDB_API_KEY` to Colab Secrets to log runs to Weights & Biases.

**Locally (API + agent, no GPU)**
```bash
pip install -r requirements.txt
pytest -q                                   # 17 tests: schema, metrics, CORD decoding, SQL guard, agent loop, API
export GOOGLE_API_KEY=...                   # PowerShell: $env:GOOGLE_API_KEY="..."
uvicorn api.main:app --reload               # POST /extract?method=llm (file=@receipt.jpg), POST /ask {"question": ...}
```
`method=vlm` serves the fine-tuned adapter (`VLM_ADAPTER=path/to/doclens-qwen25vl-lora`) and needs a CUDA GPU.
For Postgres + pgvector: `docker compose up -d` and `DATABASE_URL=postgresql+psycopg://doclens:doclens@localhost:5432/doclens`.

## Layout
```
doclens/schema.py      evaluation schema, prompt, tolerant JSON recovery (code fences, truncated output)
doclens/cord.py        CORD > words/boxes/BIO labels for LayoutLMv3; BIO predictions > JSON
doclens/metrics.py     field F1, line-item F1, per-field F1
doclens/extractors.py  GeminiExtractor (structured output, few-shot, rate limiting), VLMExtractor (4-bit + LoRA)
doclens/store.py       SQLAlchemy store, amount parsing, embeddings (numpy or pgvector), read-only SQL guard
doclens/agent.py       LangGraph agent: tools, conditional edges, MemorySaver checkpointer
api/main.py            FastAPI service
scripts/compare.py     builds results/COMPARISON.md from predictions/*.jsonl
```
