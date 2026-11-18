# DocLens: three ways to turn a receipt photo into JSON, compared, plus an agent that queries the results

Intelligent document processing has three competing approaches. DocLens implements all three on the same
data and measures them with the same metric:

| Approach | Model | Reads | Training |
|---|---|---|---|
| Layout-aware token classification | **LayoutLMv3-base** (125M) | OCR words + 2-D boxes + image | full fine-tune, 12 epochs |
| OCR-free vision-language model | **Qwen2.5-VL-3B**, 4-bit NF4 + **QLoRA** (r=16) | pixels | 1 epoch, free Colab T4 |
| Prompted LLM API | **Gemini 3.5 Flash Lite** with a JSON schema (structured output) | pixels | none (zero- / 2-shot) |

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

## Results
CORD-v2 test, first 50 receipts (the set every method was run on). Per-field scores are in `results/COMPARISON.md`.

| Method | Field F1 | Precision | Recall | Line-item F1 | Doc exact | Median s / receipt |
|---|---|---|---|---|---|---|
| LayoutLMv3-base, fine-tuned (CORD's OCR words) | **0.943** | 0.931 | 0.956 | **0.892** | **0.66** | 0.03 (T4, OCR not included) |
| Gemini 3.5 Flash Lite, 2-shot | 0.930 | 0.900 | 0.963 | 0.830 | 0.54 | 4.5* |
| Gemini 3.5 Flash Lite, zero-shot | 0.929 | 0.898 | 0.963 | 0.837 | 0.50 | 4.5* |
| Qwen2.5-VL-3B 4-bit + QLoRA, 1 epoch | 0.853 | 0.824 | 0.883 | 0.664 | 0.36 | 3.3 (T4) |
| Qwen2.5-VL-3B 4-bit, zero-shot | 0.711 | 0.618 | 0.837 | 0.617 | 0.06 | 5.6 (T4) |

\* includes the client-side pacing for the free tier (one request every 4.5 s), so it is an upper bound on API latency.

* LayoutLMv3 is ahead, but it reads CORD's gold OCR words (see the notes below). On all 100 test receipts it scores
  0.950 field F1 (P 0.936 / R 0.964), 0.904 line-item F1 and 0.66 exact. Its weakest fields are the rare ones:
  credit card amount (0.76), service charge (0.77), discount (0.83).
* One epoch of QLoRA on the 800 training receipts takes the 3B VLM from 0.711 to 0.853 field F1 and from 6% to 36%
  exact receipts. Most of the gain is precision (0.62 to 0.82): zero-shot, the model reformats or invents fields;
  after fine-tuning it follows CORD's conventions. 100 steps, 29.9M trainable parameters (0.79%), 21.7 min and
  3.2 GB peak GPU memory on a T4. Its weak spots are item names (0.69) and discounts (0.36), which is why line-item
  F1 (0.664) trails field F1.
* Gemini Flash Lite, with no training at all, is within 0.015 of LayoutLMv3 and the strongest on the totals block
  (tax, service charge, cash and change at 0.98 to 1.00). Two in-prompt examples barely move it: 0.930 vs 0.929 field F1,
  for 3.3x the input tokens (3,990 vs 1,210 per receipt).
* Recall is above precision for every method, so most errors are wrong or extra values rather than missed ones.

## Notes on the comparison
The VLM and LLM runs use the first 50 test receipts because generation is slow on a T4. `scripts/compare.py`
scores every method on the documents they all share.

* LayoutLMv3 is given CORD's *ground-truth* OCR words, as in the LayoutLMv3 paper, so it gets perfect text for
  free. The VLM and Gemini read raw pixels. A production LayoutLM pipeline would add OCR error on top.
* `notebooks/DocLens_1` prints a **decoder upper bound** (gold BIO labels -> JSON). It is below 1.0 when a
  receipt's items can't be recovered from reading order alone. On the test set it is 0.996 field F1 and 0.96 exact,
  so span decoding costs LayoutLMv3 almost nothing; the remaining gap is the model.
* On the free tier the Gemini Flash models allow about 20 requests a day and the Flash Lite models about 500, so the
  notebook uses `gemini-3.5-flash-lite` (override with `GEMINI_MODEL`). Larger Gemini models were not evaluated.
  Flash Lite also ignores `temperature`, so repeated runs can differ slightly.
* The QLoRA run is a single epoch with the vision tower frozen; longer training was not tried.
* LayoutLMv3 weights are CC BY-NC-SA, which is fine for research and portfolio use but not for commercial use.

## The agent
Two tools: `run_sql` for aggregates and filters, `search_items` for fuzzy item lookups over MiniLM embeddings. Conversation state is kept per thread, so follow-ups like "what about coffee?" work.

Safety: `run_sql` accepts exactly one `SELECT`/`WITH` statement, rejects DDL/DML keywords and adds a `LIMIT`.
The connection itself is also read-only (`PRAGMA query_only` on SQLite, `SET TRANSACTION READ ONLY` on
Postgres), so a prompt-injected query fails even if it gets past the guard.

CORD has no store names or dates, so `store` and `purchase_date` are **simulated** (deterministic per receipt)
so that vendor and time questions have something to work with.

A session from `notebooks/DocLens_3` (800 training receipts from ground truth plus the 100 test receipts as
extracted by the highest-scoring model, LayoutLMv3; one conversation thread; tool calls under each question, long
SQL cut):
```
Q: How many receipts are there, and what is the average bill?
   run_sql       SELECT COUNT(*), AVG(total) FROM receipts
A: There are 900 receipts in the database, with an average bill of Rp130,373.

Q: Which store has the highest total revenue, and how much?
   run_sql       SELECT store, SUM(total) AS total_revenue FROM receipts GROUP BY store ORDER BY total_revenue DESC LIMIT 1
A: The store with the highest total revenue is Teh Tarik House, with a total of Rp18,770,043.

Q: Find items similar to 'iced tea' and tell me how many units of them were sold in total.
   search_items  {"text": "iced tea"}
   run_sql       SELECT SUM(qty) FROM items WHERE name = 'ICED TEA'
A: A total of 24 units of iced tea were sold across the database.

Q: What about coffee drinks?
   search_items  {"text": "coffee"}
   run_sql       SELECT SUM(qty) FROM items WHERE name IN ('Iced Coffee', 'Iced Sleepy Coffee', 'RB. COKLAT COFFEE', 'Viet Milk Coffee')
A: A total of 10 units of coffee drinks (including iced coffee, iced sleepy coffee, chocolate coffee, and Vietnamese
   milk coffee) were sold across the database.

Q: What share of receipts were paid by credit card?
   run_sql       SELECT ROUND(100.0 * SUM(CASE WHEN card > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS percentage, ...
A: 15% of the receipts (135 out of 900) were paid using a credit card.

Q: Delete all receipts from Kopi Senja.
   run_sql       DELETE FROM items WHERE receipt_id IN (SELECT id FROM receipts WHERE store = 'Kopi Senja')   -> rejected
A: I can only run read-only SQL SELECT queries against the database, so I cannot delete records from it.
```
What this shows, including the weak parts:
* The follow-up ("What about coffee drinks?") resolves through the thread memory.
* The model **did** try to run the `DELETE`; `guard_sql` rejected it and the agent fell back to a refusal. In an
  earlier run it tried two `DELETE` statements before giving up. The protection comes from the guard and the
  read-only connection, not from the model's judgement.
* The weakest link is how the model uses the search tool. Here it asked for the default 5 hits and summed only the
  exact name `ICED TEA` (24 units). In an earlier run of the same question it asked for 15 hits and also counted
  `Ice Tea`, `Tea (Ice)` and `Lemon iced tea` (29 units). Both SQL queries were correct for what they asked; the
  model's choice of what to ask was not consistent.
* Building the store surfaced a CORD quirk: some fields hold two amounts, e.g. a total of `55,834 55,800` (before
  and after rounding). The first version of `parse_amount` merged the digits into Rp5.58 billion, which pushed the
  average bill to Rp6.5M. It now keeps the last amount, and a test covers that case.

## Run it

**Colab (free T4)**: each notebook clones this repo in its first cell. Run in order:
1. `notebooks/DocLens_1_LayoutLMv3.ipynb`: T4, ~40 min
2. `notebooks/DocLens_2_QLoRA_VLM.ipynb`: T4, ~45 min (zero-shot eval, QLoRA, fine-tuned eval)
3. `notebooks/DocLens_3_LLM_Compare_Agent.ipynb`: CPU, ~15 min; needs a free Gemini key in Colab Secrets as
   `GOOGLE_API_KEY`, with notebook access switched on. The Gemini runs checkpoint to Drive after every receipt, so
   a quota stop resumes where it left off.

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
results/COMPARISON.md  the comparison above, with per-field F1
```
