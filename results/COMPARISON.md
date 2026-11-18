CORD-v2 test, 50 receipts shared by all methods. Field F1 = micro F1 over flattened (key, value) leaves; line-item F1 requires name and price to match together.

| Method | Field F1 | Precision | Recall | Line-item F1 | Doc exact | Median latency (s) |
|---|---|---|---|---|---|---|
| LayoutLMv3-base fine-tuned (token classification, uses CORD's OCR tokens) | 0.943 | 0.931 | 0.956 | 0.892 | 0.660 | 0.033 |
| Gemini, 2-shot + JSON schema | 0.930 | 0.900 | 0.963 | 0.830 | 0.540 | 4.465 |
| Gemini, zero-shot + JSON schema | 0.929 | 0.898 | 0.963 | 0.837 | 0.500 | 4.512 |
| Qwen2.5-VL-3B 4-bit + QLoRA (1 epoch) | 0.853 | 0.824 | 0.883 | 0.664 | 0.360 | 3.349 |
| Qwen2.5-VL-3B 4-bit, zero-shot | 0.711 | 0.618 | 0.837 | 0.617 | 0.060 | 5.629 |

Per-field F1:

| Field | layoutlmv3 | llm_fewshot | llm_zeroshot | vlm_qlora | vlm_zeroshot |
|---|---|---|---|---|---|
| menu.cnt | 0.99 | 0.96 | 0.95 | 0.87 | 0.82 |
| menu.nm | 0.93 | 0.84 | 0.84 | 0.69 | 0.66 |
| menu.price | 0.96 | 0.95 | 0.95 | 0.91 | 0.84 |
| sub_total.discount_price | 0.89 | 0.77 | 0.77 | 0.36 | 0.18 |
| sub_total.service_price | 0.92 | 1.00 | 1.00 | 0.77 | 0.10 |
| sub_total.subtotal_price | 0.94 | 0.94 | 0.95 | 0.94 | 0.67 |
| sub_total.tax_price | 0.90 | 1.00 | 1.00 | 0.86 | 0.56 |
| total.cashprice | 0.90 | 0.99 | 0.99 | 0.99 | 0.76 |
| total.changeprice | 0.94 | 0.98 | 1.00 | 0.94 | 0.70 |
| total.creditcardprice | 0.78 | 0.94 | 0.88 | 0.62 | 0.34 |
| total.total_price | 0.95 | 0.97 | 0.97 | 0.96 | 0.88 |
