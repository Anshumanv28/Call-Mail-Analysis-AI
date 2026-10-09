"""LLM backends behind one tiny interface: chat_batch(list[messages]) -> list[str].

hf     : in-process transformers, batched greedy decoding. No server, nothing to
         install beyond transformers — the light option for a Kaggle T4.
openai : any OpenAI-compatible server (vLLM, SGLang, llama.cpp). Passes the JSON schema
         through as guided decoding where the server supports it.
llama  : the openai backend preset for a local llama.cpp server (bin/llama-server.exe, port
         8080): JSON-schema-constrained decoding, the tree prompt cached in KV across calls.
"""

from __future__ import annotations

import json
import os
import time

import httpx

Messages = list[dict[str, str]]


RETRY_WINDOW_S = int(os.environ.get("CALLAI_LLM_RETRY_S", 2 * 3600))  # keep long runs alive through server restarts


class HFBackend:
    name = "hf"

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507", batch_size: int = 3,
                 max_input_tokens: int = 2300, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {}
        if torch.cuda.is_available():
            # T4 has no bf16; fp16 is fine for a 4B instruct model
            kw["dtype"] = torch.float16
            kw["device_map"] = "cuda:0"
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        self.model.eval()
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.token_budget = batch_size * max_input_tokens
        self.model_id = model

    def _fit(self, messages: Messages) -> str:
        """Render the chat template; if the prompt is too long, cut the LAST user message
        from its middle until it fits. The template tail (the instruction to answer and
        the assistant turn) is never truncated — a cut tail makes the model continue the
        transcript instead of answering."""
        def render(msgs):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        prompt = render(messages)
        n = len(self.tok(prompt)["input_ids"])
        if n <= self.max_input_tokens:
            return prompt
        msgs = [dict(m) for m in messages]
        k = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        text = msgs[k]["content"]
        for _ in range(8):
            keep = max(0.2, (self.max_input_tokens - 40) / n)  # fraction of the text to keep
            cut = int(len(text) * keep)
            head, tail = text[: cut * 2 // 3], text[-(cut // 3) :] if cut // 3 else ""
            text = head + "\n[...]\n" + tail
            msgs[k]["content"] = text
            prompt = render(msgs)
            n = len(self.tok(prompt)["input_ids"])
            if n <= self.max_input_tokens:
                break
        return prompt

    def chat_batch(self, batch: list[Messages], max_new_tokens: int = 300, json_schema: dict | None = None,
                   temperature: float = 0.0) -> list[str]:
        import torch

        prompts_all = [self._fit(m) for m in batch]
        lens = [len(self.tok(p)["input_ids"]) for p in prompts_all]
        # similar lengths together -> less padding; batch by token budget, not by count,
        # so short calls go six at a time and long ones alone (T4: 15 GB, 150k-vocab logits)
        order = sorted(range(len(prompts_all)), key=lambda i: lens[i])
        outs_by_idx: dict[int, str] = {}
        groups: list[list[int]] = []
        for j in order:
            if groups and (len(groups[-1]) + 1) * max(lens[k] for k in groups[-1] + [j]) <= self.token_budget \
                    and len(groups[-1]) < 2 * self.batch_size:
                groups[-1].append(j)
            else:
                groups.append([j])
        for idx in groups:
            prompts = [prompts_all[j] for j in idx]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            gen_kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                gen_kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                gen_kw.update(do_sample=False)
            with torch.no_grad():
                gen = self.model.generate(**enc, **gen_kw)
            new = gen[:, enc["input_ids"].shape[1] :]
            for j, o in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                outs_by_idx[j] = o.strip()
            del enc, gen, new
        return [outs_by_idx[i] for i in range(len(batch))]


class OpenAIBackend:
    name = "openai"

    def __init__(self, base_url: str | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 api_key: str | None = None, guided: str = "auto", timeout: float = 120.0, parallel: int = 2):
        import httpx

        self.base_url = (base_url or os.environ.get("CALLAI_LLM_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model_id = model
        self.key = api_key or os.environ.get("CALLAI_LLM_KEY", "none")
        self.guided = guided  # "vllm" | "openai" | "none" | "auto"
        self.parallel = max(1, parallel)  # match the server's slot count (llama-server -np N)
        self.http = httpx.Client(timeout=timeout)

    def _one(self, messages: Messages, max_new_tokens: int, json_schema: dict | None, temperature: float) -> str:
        body = {"model": self.model_id, "messages": messages, "max_tokens": max_new_tokens,
                "temperature": temperature}
        if json_schema is not None:
            if self.guided in ("vllm", "auto"):
                body["extra_body"] = {"guided_json": json_schema}
            if self.guided in ("openai", "auto"):
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "out", "schema": json_schema, "strict": True}}
        # a busy shared server (several clients, long queue) or a briefly stalled one must not sink an
        # hours-long run: keep retrying transient failures for up to RETRY_WINDOW_S; give up at once on 4xx
        deadline = time.time() + RETRY_WINDOW_S
        attempt = 0
        while True:
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body,
                                   headers={"Authorization": f"Bearer {self.key}"})
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()  # our own mistake (prompt too long, bad schema): retrying cannot help
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                err = e
            except (httpx.TimeoutException, httpx.TransportError, KeyError, ValueError) as e:
                err = e
            attempt += 1
            if time.time() > deadline:
                raise RuntimeError(f"LLM server unavailable for {RETRY_WINDOW_S // 60} min: {err!r}") from err
            time.sleep(min(60.0, 2.0 * attempt))

    def chat_batch(self, batch: list[Messages], max_new_tokens: int = 300, json_schema: dict | None = None,
                   temperature: float = 0.0) -> list[str]:
        if self.parallel == 1 or len(batch) == 1:
            return [self._one(m, max_new_tokens, json_schema, temperature) for m in batch]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(lambda m: self._one(m, max_new_tokens, json_schema, temperature), batch))


def make_llm(kind: str, **kw):
    if kind == "hf":
        return HFBackend(**kw)
    if kind == "openai":
        return OpenAIBackend(**kw)
    if kind == "llama":  # scripts/llm_server.cmd -> llama.cpp on :8080; the schema goes as response_format
        kw = {"base_url": os.environ.get("CALLAI_LLM_URL", "http://127.0.0.1:8080/v1"), "model": "local",
              "guided": "openai", "timeout": 600.0,
              "parallel": int(os.environ.get("CALLAI_LLM_PARALLEL", "2")), **kw}   # = the server's -np slots
        return OpenAIBackend(**kw)
    raise ValueError(f"unknown LLM backend {kind!r}")


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter).
    Uses the real JSON decoder from each "{", so braces inside strings cannot cut the object short."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    dec = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = dec.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None
