"""Model backends. One interface; mlx and hf implement it.

No task knowledge here. Callers pass already-formatted prompts.

Cache-aware decode lives in DecodeSession (session.py). Backends only
construct a session. --no-cache forces StatelessSession so old timings
can be reproduced.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from early_exit.session import DecodeSession, HfCacheSession, MlxCacheSession, StatelessSession


DEFAULT_MODEL_MLX = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FALLBACK_MODEL_MLX = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_MODEL_HF = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL_HF = "Qwen/Qwen2.5-1.5B-Instruct"


class Backend:
    name = "base"
    model_id = "unset"
    load_s = 0.0
    use_cache = False
    cache_enabled = False

    def format_chat(self, user_text: str) -> str:
        return user_text

    def encode_text(self, text: str, special: bool = False) -> list[int]:
        tok = getattr(self, "tokenizer", None)
        if tok is None or not hasattr(tok, "encode"):
            return []
        try:
            return list(tok.encode(text, add_special_tokens=special))
        except TypeError:
            return list(tok.encode(text))

    def decode_ids(self, ids: list[int]) -> str:
        tok = getattr(self, "tokenizer", None)
        if tok is None or not ids:
            return ""
        try:
            return tok.decode(ids, skip_special_tokens=True)
        except TypeError:
            return tok.decode(ids)

    def count_tokens(self, text: str) -> int:
        ids = self.encode_text(text, special=False)
        if ids:
            return len(ids)
        return max(1, len(text.split())) if text else 0

    def is_eos(self, token_id: int) -> bool:
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return False
        eos_ids = getattr(tok, "eos_token_ids", None)
        if isinstance(eos_ids, int) and token_id == eos_ids:
            return True
        if eos_ids is not None and not isinstance(eos_ids, int):
            try:
                if token_id in eos_ids:
                    return True
            except TypeError:
                pass
        eos = getattr(tok, "eos_token_id", None)
        return eos is not None and token_id == eos

    def warmup(self) -> None:
        try:
            self.generate("Reply with exactly: OK", max_tokens=4)
        except Exception as e:
            print(f"WARNING: warmup failed: {e!r}", file=sys.stderr)

    def start_session(self, prompt: str) -> DecodeSession:
        if self.use_cache and self.cache_enabled:
            return self._cached_session(prompt)
        return StatelessSession(self, prompt)

    def _cached_session(self, prompt: str) -> DecodeSession:
        return StatelessSession(self, prompt)

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        raise NotImplementedError

    def generate_continue(self, prefix_prompt: str, max_tokens: int) -> tuple[str, int, float]:
        return self.generate(prefix_prompt, max_tokens)

    def stream_continue(self, prefix_prompt: str, max_tokens: int):
        """Yield (piece, n_tok_so_far, dt_so_far) one token at a time."""
        text, n_tok, dt = self.generate_continue(prefix_prompt, max_tokens)
        yield text, n_tok, dt


class MlxBackend(Backend):
    name = "mlx"

    def __init__(self, model_id: str, temperature: float = 0.0, use_cache: bool = True):
        from mlx_lm import load, generate, stream_generate

        self._generate = generate
        self._stream_generate = stream_generate
        self.temperature = temperature
        self.use_cache = use_cache
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(temp=temperature)
        except Exception:
            self._sampler = None
        try:
            from mlx_lm.models.cache import make_prompt_cache  # noqa: F401

            self.cache_enabled = True
        except Exception:
            self.cache_enabled = False
            if use_cache:
                print(
                    "WARNING: mlx_lm.models.cache.make_prompt_cache not available; "
                    "falling back to cold generate",
                    file=sys.stderr,
                )

        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        self.load_s = time.perf_counter() - t0
        self.model_id = model_id

    def _cached_session(self, prompt: str) -> DecodeSession:
        return MlxCacheSession(self, prompt)

    def format_chat(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return user_text

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        return self._run(self.format_chat(prompt), max_tokens)

    def generate_continue(self, prefix_prompt: str, max_tokens: int) -> tuple[str, int, float]:
        return self._run(prefix_prompt, max_tokens)

    def _kwargs(self, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        else:
            kwargs["temp"] = self.temperature
        return kwargs

    def _run(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        kwargs = self._kwargs(max_tokens)
        t0 = time.perf_counter()
        text_parts: list[str] = []
        n_tok = 0
        try:
            for resp in self._stream_generate(self.model, self.tokenizer, prompt, **kwargs):
                piece = getattr(resp, "text", None)
                if piece is None:
                    piece = str(resp)
                text_parts.append(piece)
                n_tok += 1
        except TypeError as e:
            print(
                f"WARNING: stream_generate raised TypeError, falling back "
                f"to non-streaming generate(): {e!r}",
                file=sys.stderr,
            )
            text = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
            dt = time.perf_counter() - t0
            n_tok = self.count_tokens(text) or max(1, len(text.split()))
            return text, n_tok, dt
        return "".join(text_parts), n_tok, time.perf_counter() - t0

    def stream_continue(self, prefix_prompt: str, max_tokens: int):
        kwargs = self._kwargs(max_tokens)
        t0 = time.perf_counter()
        n_tok = 0
        acc = ""
        try:
            for resp in self._stream_generate(
                self.model, self.tokenizer, prefix_prompt, **kwargs
            ):
                piece = getattr(resp, "text", None)
                if piece is None:
                    piece = str(resp)
                acc += piece
                n_tok += 1
                yield piece, n_tok, time.perf_counter() - t0
        except TypeError as e:
            print(
                f"WARNING: stream_generate raised TypeError, falling back "
                f"to non-streaming generate(): {e!r}",
                file=sys.stderr,
            )
            text, n_tok, dt = self._run(prefix_prompt, max_tokens)
            yield text, n_tok, dt


class HfBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, temperature: float = 0.0, use_cache: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_enabled = True
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self.model_id = model_id

    def _cached_session(self, prompt: str) -> DecodeSession:
        return HfCacheSession(self, prompt)

    def format_chat(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return user_text

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        return self._run(self.format_chat(prompt), max_tokens)

    def generate_continue(self, prefix_prompt: str, max_tokens: int) -> tuple[str, int, float]:
        return self._run(prefix_prompt, max_tokens)

    def _run(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, int(new_tokens.shape[0]), dt


def default_model_for(backend: str) -> str:
    return DEFAULT_MODEL_MLX if backend == "mlx" else DEFAULT_MODEL_HF


def fallback_model_for(backend: str) -> str:
    return FALLBACK_MODEL_MLX if backend == "mlx" else FALLBACK_MODEL_HF


def build_backend(
    backend: str,
    model_id: str,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> Backend:
    if backend == "mlx":
        return MlxBackend(model_id, temperature=temperature, use_cache=use_cache)
    if backend == "hf":
        return HfBackend(model_id, temperature=temperature, use_cache=use_cache)
    if backend == "fake":
        raise ValueError("fake backend is constructed by tests, not build_backend")
    raise ValueError(f"unknown backend {backend!r}")
