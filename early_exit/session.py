"""Decode sessions: one prefill, then cached decode, with a discarded probe fork.

StatelessSession is the old driver (every call re-encodes the full prefix).
Cached sessions keep a KV cache and only run the model on new tokens.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Iterator

from early_exit.timing import GenerateResult

if TYPE_CHECKING:
    from early_exit.backends import Backend


class DecodeSession:
    cache_enabled: bool = False

    def continue_decode(self, max_new: int) -> GenerateResult:
        raise NotImplementedError

    def probe(self, suffix: str, max_new: int) -> GenerateResult:
        raise NotImplementedError

    def stream_decode(self, max_new: int) -> Iterator[tuple[str, int, float]]:
        res = self.continue_decode(max_new)
        yield res.text, res.n_tokens, res.latency_s


class StatelessSession(DecodeSession):
    """Re-encodes chat0+reason on every call. Reproduces the original timings."""

    cache_enabled = False

    def __init__(self, backend: Backend, prompt: str):
        self.backend = backend
        self.prefix = prompt

    def continue_decode(self, max_new: int) -> GenerateResult:
        t0 = time.perf_counter()
        text, n_tok, dt = self.backend.generate_continue(self.prefix, max_new)
        self.prefix += text
        prefill = self.backend.count_tokens(self.prefix[: max(0, len(self.prefix) - len(text))])
        return GenerateResult(
            text=text,
            n_tokens=n_tok,
            latency_s=dt if dt else time.perf_counter() - t0,
            prefill_tokens=prefill,
            decode_tokens=n_tok,
            prefill_s=dt,
            decode_s=0.0,
            n_calls=1,
            cache_hit=False,
            kind="cold",
        )

    def probe(self, suffix: str, max_new: int) -> GenerateResult:
        t0 = time.perf_counter()
        text, n_tok, dt = self.backend.generate_continue(self.prefix + suffix, max_new)
        prefill = self.backend.count_tokens(self.prefix + suffix)
        return GenerateResult(
            text=text,
            n_tokens=n_tok,
            latency_s=dt if dt else time.perf_counter() - t0,
            prefill_tokens=prefill,
            decode_tokens=n_tok,
            prefill_s=dt,
            decode_s=0.0,
            n_calls=1,
            cache_hit=False,
            kind="cold",
        )

    def stream_decode(self, max_new: int) -> Iterator[tuple[str, int, float]]:
        n = 0
        t0 = time.perf_counter()
        acc = ""
        for piece, n, _dt in self.backend.stream_continue(self.prefix, max_new):
            acc += piece
            yield piece, n, time.perf_counter() - t0
        self.prefix += acc


class MlxCacheSession(DecodeSession):
    cache_enabled = True

    def __init__(self, backend: Backend, prompt: str):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self.mx = mx
        self.backend = backend
        self.model = backend.model
        self.cache = make_prompt_cache(self.model)
        self.token_ids = backend.encode_text(prompt)
        t0 = time.perf_counter()
        self._last_logits = self._forward(self.token_ids, self.cache)
        self.prompt_prefill_s = time.perf_counter() - t0
        self.prompt_prefill_tokens = len(self.token_ids)
        self._prompt_prefill_logged = False

    def _forward(self, token_ids: list[int], cache):
        if not token_ids:
            raise ValueError("forward requires at least one token")
        y = self.mx.array(token_ids)[None]
        logits = self.model(y, cache=cache)
        last = logits[:, -1, :]
        self.mx.eval(last)
        self.mx.eval([c.state for c in cache])
        return last

    def _sample(self, logits) -> int:
        tok = self.mx.argmax(logits, axis=-1)
        self.mx.eval(tok)
        return int(tok.item())

    def _consume_prefill_log(self) -> tuple[int, float]:
        if self._prompt_prefill_logged:
            return 0, 0.0
        self._prompt_prefill_logged = True
        return self.prompt_prefill_tokens, self.prompt_prefill_s

    def continue_decode(self, max_new: int) -> GenerateResult:
        extra_prefill_tok, extra_prefill_s = self._consume_prefill_log()
        t0 = time.perf_counter()
        new_ids: list[int] = []
        for _ in range(max_new):
            tid = self._sample(self._last_logits)
            if self.backend.is_eos(tid):
                break
            new_ids.append(tid)
            self.token_ids.append(tid)
            self._last_logits = self._forward([tid], self.cache)
        dt = time.perf_counter() - t0
        text = self.backend.decode_ids(new_ids)
        return GenerateResult(
            text=text,
            n_tokens=len(new_ids),
            latency_s=dt + extra_prefill_s,
            prefill_tokens=extra_prefill_tok,
            decode_tokens=len(new_ids),
            prefill_s=extra_prefill_s,
            decode_s=dt,
            n_calls=1,
            cache_hit=extra_prefill_tok == 0,
            kind="decode" if extra_prefill_tok == 0 else "prefill",
            token_ids=new_ids,
        )

    def probe(self, suffix: str, max_new: int) -> GenerateResult:
        t0 = time.perf_counter()
        forked = copy.deepcopy(self.cache)
        suffix_ids = self.backend.encode_text(suffix, special=False)
        tp = time.perf_counter()
        last = self._forward(suffix_ids, forked) if suffix_ids else self._last_logits
        prefill_s = time.perf_counter() - tp
        new_ids: list[int] = []
        td = time.perf_counter()
        for _ in range(max_new):
            tid = self._sample(last)
            if self.backend.is_eos(tid):
                break
            new_ids.append(tid)
            last = self._forward([tid], forked)
        decode_s = time.perf_counter() - td
        text = self.backend.decode_ids(new_ids)
        return GenerateResult(
            text=text,
            n_tokens=len(new_ids),
            latency_s=time.perf_counter() - t0,
            prefill_tokens=len(suffix_ids),
            decode_tokens=len(new_ids),
            prefill_s=prefill_s,
            decode_s=decode_s,
            n_calls=1,
            cache_hit=True,
            kind="probe",
            token_ids=new_ids,
        )

    def stream_decode(self, max_new: int) -> Iterator[tuple[str, int, float]]:
        extra_prefill_tok, extra_prefill_s = self._consume_prefill_log()
        del extra_prefill_tok
        t0 = time.perf_counter() - extra_prefill_s
        n = 0
        for _ in range(max_new):
            tid = self._sample(self._last_logits)
            if self.backend.is_eos(tid):
                break
            self.token_ids.append(tid)
            self._last_logits = self._forward([tid], self.cache)
            n += 1
            yield self.backend.decode_ids([tid]), n, time.perf_counter() - t0


class HfCacheSession(DecodeSession):
    cache_enabled = True

    def __init__(self, backend: Backend, prompt: str):
        self.backend = backend
        self.torch = backend.torch
        self.model = backend.model
        self.token_ids = backend.encode_text(prompt)
        t0 = time.perf_counter()
        self._last_logits, self.past = self._forward(self.token_ids, None)
        self.prompt_prefill_s = time.perf_counter() - t0
        self.prompt_prefill_tokens = len(self.token_ids)
        self._prompt_prefill_logged = False

    def _clone_past(self, past):
        if past is None:
            return None
        if hasattr(past, "clone"):
            return past.clone()
        try:
            return copy.deepcopy(past)
        except Exception:
            return tuple(tuple(t.clone() for t in layer) for layer in past)

    def _forward(self, token_ids: list[int], past):
        device = self.model.device
        x = self.torch.tensor([token_ids], device=device)
        with self.torch.no_grad():
            out = self.model(x, past_key_values=past, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values

    def _sample(self, logits) -> int:
        if self.backend.temperature and self.backend.temperature > 0:
            probs = self.torch.softmax(logits.float() / self.backend.temperature, dim=-1)
            return int(self.torch.multinomial(probs, 1).item())
        return int(self.torch.argmax(logits, dim=-1).item())

    def _consume_prefill_log(self) -> tuple[int, float]:
        if self._prompt_prefill_logged:
            return 0, 0.0
        self._prompt_prefill_logged = True
        return self.prompt_prefill_tokens, self.prompt_prefill_s

    def continue_decode(self, max_new: int) -> GenerateResult:
        extra_prefill_tok, extra_prefill_s = self._consume_prefill_log()
        t0 = time.perf_counter()
        new_ids: list[int] = []
        for _ in range(max_new):
            tid = self._sample(self._last_logits)
            if self.backend.is_eos(tid):
                break
            new_ids.append(tid)
            self.token_ids.append(tid)
            self._last_logits, self.past = self._forward([tid], self.past)
        dt = time.perf_counter() - t0
        text = self.backend.decode_ids(new_ids)
        return GenerateResult(
            text=text,
            n_tokens=len(new_ids),
            latency_s=dt + extra_prefill_s,
            prefill_tokens=extra_prefill_tok,
            decode_tokens=len(new_ids),
            prefill_s=extra_prefill_s,
            decode_s=dt,
            n_calls=1,
            cache_hit=extra_prefill_tok == 0,
            kind="decode" if extra_prefill_tok == 0 else "prefill",
            token_ids=new_ids,
        )

    def probe(self, suffix: str, max_new: int) -> GenerateResult:
        t0 = time.perf_counter()
        past = self._clone_past(self.past)
        suffix_ids = self.backend.encode_text(suffix, special=False)
        tp = time.perf_counter()
        last, past = (
            self._forward(suffix_ids, past) if suffix_ids else (self._last_logits, past)
        )
        prefill_s = time.perf_counter() - tp
        new_ids: list[int] = []
        td = time.perf_counter()
        for _ in range(max_new):
            tid = self._sample(last)
            if self.backend.is_eos(tid):
                break
            new_ids.append(tid)
            last, past = self._forward([tid], past)
        decode_s = time.perf_counter() - td
        return GenerateResult(
            text=self.backend.decode_ids(new_ids),
            n_tokens=len(new_ids),
            latency_s=time.perf_counter() - t0,
            prefill_tokens=len(suffix_ids),
            decode_tokens=len(new_ids),
            prefill_s=prefill_s,
            decode_s=decode_s,
            n_calls=1,
            cache_hit=True,
            kind="probe",
            token_ids=new_ids,
        )

    def stream_decode(self, max_new: int) -> Iterator[tuple[str, int, float]]:
        extra_prefill_tok, extra_prefill_s = self._consume_prefill_log()
        del extra_prefill_tok
        t0 = time.perf_counter() - extra_prefill_s
        n = 0
        for _ in range(max_new):
            tid = self._sample(self._last_logits)
            if self.backend.is_eos(tid):
                break
            self.token_ids.append(tid)
            self._last_logits, self.past = self._forward([tid], self.past)
            n += 1
            yield self.backend.decode_ids([tid]), n, time.perf_counter() - t0
