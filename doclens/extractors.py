"""Extractors that read a receipt image and return the schema dict.

GeminiExtractor - prompted LLM (zero- or few-shot) with JSON-schema structured output.
VLMExtractor    - local open VLM (Qwen2.5-VL-3B by default) + optional LoRA adapter, 4-bit.
"""
from __future__ import annotations

import io
import os
import time
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from doclens.schema import PROMPT, Receipt, parse_model_output, restrict, to_target


def resize_max(img: Image.Image, max_side: int = 896) -> Image.Image:
    img = img.convert("RGB")
    s = max_side / max(img.size)
    return img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC) if s < 1 else img


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    resize_max(img, 1280).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class GeminiExtractor:
    name = "llm"

    def __init__(self, model: Optional[str] = None, few_shot: Sequence[Tuple[Image.Image, dict]] = (),
                 min_interval_s: float = 0.0):
        from google import genai

        self.genai = genai
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.few_shot = list(few_shot)
        self.min_interval_s = min_interval_s  # free-tier rate limits
        self._last = 0.0
        self.last_usage: dict = {}

    def _contents(self, img: Image.Image) -> list:
        from google.genai import types

        parts: List = []
        for i, (ex_img, ex_parse) in enumerate(self.few_shot, 1):
            parts += [f"Example {i}:", types.Part.from_bytes(data=_jpeg(ex_img), mime_type="image/jpeg"),
                      f"Example {i} JSON: {to_target(ex_parse)}"]
        parts += [PROMPT, types.Part.from_bytes(data=_jpeg(img), mime_type="image/jpeg")]
        return parts

    def extract(self, img: Image.Image, retries: int = 4) -> dict:
        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=Receipt)
        for attempt in range(retries):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=self._contents(img), config=cfg)
                um = getattr(resp, "usage_metadata", None)
                self.last_usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                                   "output_tokens": getattr(um, "candidates_token_count", None)}
                if resp.parsed is not None:
                    return restrict(resp.parsed.model_dump(exclude_none=True))
                return parse_model_output(resp.text or "")
            except Exception as e:  # 429 / transient errors: back off and retry
                code = getattr(e, "code", None)
                if attempt == retries - 1 or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                    raise  # bad request, auth or unknown model: retrying won't help
                time.sleep(10 * (attempt + 1))
                print(f"retrying after error: {e}")
        return restrict({})


class VLMExtractor:
    """Plain transformers + peft inference (for the API). Needs a CUDA GPU for reasonable speed."""

    name = "vlm"

    def __init__(self, model_id: Optional[str] = None, adapter: Optional[str] = None, load_4bit: bool = True,
                 max_new_tokens: int = 512):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.torch = torch
        model_id = model_id or os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        adapter = adapter or os.getenv("VLM_ADAPTER")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_use_double_quant=True) if load_4bit else None
        self.model = AutoVLM.from_pretrained(model_id, torch_dtype=dtype, quantization_config=quant, device_map="auto")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens

    def extract(self, img: Image.Image) -> dict:
        img = resize_max(img)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_model_output(text)
