# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 chain speculative decoding on sglang spec-V2 (ADR-0006, Strategy B).

A small RWKV-7 *draft* proposes K tokens by running K greedy decode steps on its
own recurrent state; the *target* verifies them in ONE ``TARGET_VERIFY`` forward.
Upstream main already implements recurrent-model spec verify+commit — the RWKV-7
linear-attention backend captures per-draft-token state in
``MambaPool.SpeculativeState.intermediate_ssm`` and
``spec_utils.commit_mamba_states_after_verify`` commits the accepted-length state
(``accept_lens-1`` for topk==1 == chain accept). So this worker only supplies the
draft; the verify/rollback machinery is upstream's.

Why not subclass ``EAGLEWorkerV2`` / ``StandaloneWorkerV2``: their draft path is
transformer-shaped (draft attention backend, draft cuda-graphs, tree mask) — a
recurrent RWKV draft has no attention or KV, so it cannot ride that machinery.
Instead we mirror ``NGRAMWorker`` (subclass ``BaseSpecWorker`` directly, build a
verify input from a non-EAGLE draft source, reuse the target verify), with the
draft source being a real 0.1B RWKV model instead of an n-gram corpus.

Exact-by-construction at temperature 0: only tokens the target would greedily
emit are committed. Registered as the plugin algorithm ``RWKV_SPEC``; run with
``--disable-overlap-schedule`` (the recurrent verify is a synchronous V2 path).

Status: increment (i) — bsz1 build, iterated on the tower main container.
"""

import logging
from typing import List, Optional

import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

logger = logging.getLogger(__name__)


class RwkvSpecAlgo(CustomSpecAlgo):
    """Plugin descriptor: a recurrent draft with NO draft KV cache (like NGRAM),
    so the KV reserve must not be inflated for a draft-KV path."""

    def has_draft_kv(self) -> bool:
        return False

    def carries_draft_hidden_states(self) -> bool:
        # The recurrent draft is a separate model with its own O(1) state; it
        # does not hand hidden states to the target across a P/D boundary
        # (that is an EAGLE-family notion). Not in CustomSpecAlgo's defaults.
        return False

    def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens=True):
        # Mirror SpeculativeAlgorithm.create_future_map (also absent from
        # CustomSpecAlgo). init_overlap constructs it unconditionally; the sync
        # (--disable-overlap-schedule) forward path we run does not consume it.
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)


def _validate_server_args(server_args: ServerArgs) -> None:
    if not server_args.speculative_draft_model_path:
        raise ValueError(
            "RWKV_SPEC needs --speculative-draft-model-path <0.1B RWKV-7 dir>."
        )
    if not server_args.disable_overlap_schedule:
        # create_worker also guards this; make the message actionable early.
        raise ValueError("RWKV_SPEC requires --disable-overlap-schedule (V2 sync path).")


@SpeculativeAlgorithm.register(
    "RWKV_SPEC",
    supports_overlap=False,
    validate_server_args=_validate_server_args,
    spec_class=RwkvSpecAlgo,
)
def _rwkv_spec_factory(server_args: ServerArgs):
    # Lazy: importing the worker class pulls in TpModelWorker/ScheduleBatch/etc.,
    # which must NOT happen at plugin-registration (import) time (circular).
    return RwkvSpecWorker


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #


class RwkvSpecWorker(BaseSpecWorker):
    """Bespoke recurrent spec worker (BaseSpecWorker surface, NGRAM-shaped).

    Subclassing BaseSpecWorker inherits no-op init_attention_backends /
    init_cuda_graphs / on_verify_complete_cpu / activate_step_by_batch — exactly
    right for an eager recurrent draft (no separate attn backend, no draft graphs
    in increment (i)). target_worker / draft_worker / clear_cache_pool are the
    abstract methods we implement.

    The scheduler sets ``self.model_worker = draft_worker`` and drives
    ``forward_batch_generation(batch)`` each step on the non-overlap V2 path.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker,
    ):
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.server_args = server_args
        self._target_worker = target_worker
        self.target_runner = target_worker.model_runner
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.k = int(server_args.speculative_num_draft_tokens or 4)
        self.speculative_num_draft_tokens = self.k

        # The draft loads speculative_draft_model_path (TpModelWorker resolves it
        # via is_draft_worker). Force its context to the target's; keep it eager
        # for increment (i) — draft-decode cuda graphs are the speed increment.
        server_args.context_length = self.target_runner.model_config.context_len
        with _preserve(server_args, "disable_cuda_graph", True):
            self._draft = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )
        self.draft_runner = self._draft.model_runner

        d_cfg = self.draft_runner.model_config.hf_config
        t_cfg = self.target_runner.model_config.hf_config
        self.draft_layers = list(range(d_cfg.num_hidden_layers))
        self.target_layers = list(range(t_cfg.num_hidden_layers))
        self._draft_slot = {}  # rid -> draft pool req index
        self._draft_shim = {}  # rid -> shim request object (owns draft pool bindings)
        self._rounds = 0
        self._accept_sum = 0
        logger.info(
            "RWKV_SPEC up: draft=%s K=%d (increment i, eager, --disable-overlap-schedule)",
            server_args.speculative_draft_model_path,
            self.k,
        )

    # ---- BaseSpecWorker surface -------------------------------------------- #

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # Target pools exist by the time the scheduler calls this; keep refs.
        # The draft builds its OWN fresh pool (req_to_token_pool=None) so its
        # recurrent state never aliases the target's — but is_draft_worker does
        # not self-profile, so it needs the scheduler-sized memory_pool_config.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self._draft.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        )
        self.draft_pool = self.draft_runner.req_to_token_pool
        self.target_pool = self.target_runner.req_to_token_pool

    @property
    def model_runner(self):  # scheduler reads draft_worker.model_runner.*
        return self.draft_runner

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        # Not an EagleDraftWorker — we own the (recurrent) draft ourselves.
        return None

    def clear_cache_pool(self):
        self._draft_slot.clear()
        self._draft_shim.clear()

    def update_weights_from_tensor(self, recv_req):
        return self._target_worker.update_weights_from_tensor(recv_req)

    # ---- the V2 entry point ------------------------------------------------ #

    def forward_batch_generation(self, batch, on_publish=None):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        if not batch.forward_mode.is_decode():
            # Extend/prefill: target runs normally; mirror the chunk into the
            # draft's own state so the round invariant holds at first decode.
            batch_result = self._target_worker.forward_batch_generation(batch)
            self._mirror_prefill_into_draft(batch)
            return batch_result

        return self._decode_round(batch, on_publish)

    # ---- prefill mirror ---------------------------------------------------- #

    def _mirror_prefill_into_draft(self, batch):
        """Run the same prompt chunk through the draft so both models have
        consumed the prompt (radix off for increment (i))."""
        for req in batch.reqs:
            # NOTE: keyed off `prefix_indices` (the radix-cache hit length), NOT
            # `extend_range` — this is a hard invariant check ("radix must be off"),
            # independent of chunked-prefill bookkeeping below.
            radix_prefix_len = (
                len(req.prefix_indices) if req.prefix_indices is not None else 0
            )
            if req.rid not in self._draft_slot:
                if radix_prefix_len != 0:
                    raise RuntimeError(
                        "RWKV_SPEC: radix prefix hit on an unseen request — run with "
                        "--disable-radix-cache (increment (i) scope)."
                    )
                self._ensure_draft_slot(req)
            # `Req.fill_ids` doesn't exist on current main; `get_fill_ids()` returns
            # `full_untruncated_fill_ids[:extend_range.end]`. Slice by
            # `extend_range.start:end` (the chunk THIS call actually extended), not
            # by `radix_prefix_len` — with radix off, `prefix_indices` stays empty
            # across chunked-prefill continuations of the same request, so slicing
            # by it would re-feed the whole accumulated prompt into the draft on
            # every chunk after the first.
            chunk_start = req.extend_range.start
            chunk = list(req.get_fill_ids()[chunk_start : req.extend_range.end])
            if chunk:
                self._draft_extend(req, chunk, chunk_start)

    # ---- the chain round --------------------------------------------------- #

    def _decode_round(self, batch, on_publish):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        flat_tokens: List[int] = []
        accept_lens: List[int] = []
        for i, req in enumerate(batch.reqs):
            appended = self._round_one(batch, i, req)
            flat_tokens.extend(appended)
            accept_lens.append(len(appended) - 1)  # drafts accepted (excl. the bonus)
            self._rounds += 1
            self._accept_sum += len(appended)
            if req.finished():
                self._release_draft_slot(req)

        self._sync_batch_lens(batch)
        dev = batch.seq_lens.device
        next_token_ids = torch.tensor(flat_tokens, dtype=torch.int64, device=dev)
        accept_lens_t = torch.tensor(accept_lens, dtype=torch.int64, device=dev)
        new_seq_lens = batch.seq_lens  # already synced to include committed tokens
        if on_publish is not None:
            on_publish(new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            num_accepted_tokens=int(sum(accept_lens)),
            accept_lens=accept_lens_t,
            num_correct_drafts_per_req_cpu=accept_lens,
            new_seq_lens=new_seq_lens,
            can_run_cuda_graph=False,
        )

    def _round_one(self, batch, i, req) -> List[int]:
        target_req_idx = int(batch.req_pool_indices[i])
        draft_req_idx = self._draft_slot[req.rid]
        t_mslot = self._mamba_idx(self.target_pool, target_req_idx)
        d_mslot = self._mamba_idx(self.draft_pool, draft_req_idx)
        seq_committed = len(req.origin_input_ids) + len(req.output_ids)
        t_last = req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]

        snap_t = _snapshot(self.target_pool, self.target_layers, t_mslot)
        snap_d = _snapshot(self.draft_pool, self.draft_layers, d_mslot)

        # 1) draft proposes K tokens (K eager decode steps on its own pool).
        drafts = self._draft_decode_k(req, draft_req_idx, t_last, seq_committed)

        # 2) chain-verify: ONE target extend over [t_last, d0..d_{K-2}] -> K logits.
        verify_in = [t_last] + drafts[:-1]
        target_argmax = self._target_verify(req, target_req_idx, verify_in, seq_committed - 1)

        # 3) accept longest greedy-matching prefix.
        j = 0
        while j < self.k and int(target_argmax[j]) == drafts[j]:
            j += 1
        if j >= self.k:
            committed = list(drafts)  # both states already correct; keep
        else:
            bonus = int(target_argmax[j])
            committed = drafts[:j] + [bonus]
            _restore(self.target_pool, self.target_layers, snap_t)
            _restore(self.draft_pool, self.draft_layers, snap_d)
            commit_in = [t_last] + drafts[:j]  # J+1 tokens; bonus stays pending
            self._target_verify(req, target_req_idx, commit_in, seq_committed - 1, capture=False)
            self._draft_extend(req, commit_in, seq_committed - 1)

        appended: List[int] = []
        for tok in committed:
            req.output_ids.append(tok)
            appended.append(tok)
            req.check_finished()
            if req.finished():
                break
        req.spec_verify_ct = getattr(req, "spec_verify_ct", 0) + 1
        return appended

    # ---- draft pool slot mgmt --------------------------------------------- #

    def _ensure_draft_slot(self, req):
        import types

        shim = types.SimpleNamespace(
            rid=req.rid, req_pool_idx=None, mamba_pool_idx=None,
        )
        idx = self.draft_pool.alloc([shim])
        assert idx is not None, "draft state pool exhausted"
        self._draft_slot[req.rid] = int(idx[0])
        self._draft_shim[req.rid] = shim

    def _release_draft_slot(self, req):
        idx = self._draft_slot.pop(req.rid, None)
        shim = self._draft_shim.pop(req.rid, None)
        if idx is None or shim is None:
            return
        try:
            if getattr(shim, "mamba_pool_idx", None) is not None:
                self.draft_pool.free_mamba_cache(shim)
            self.draft_pool.free(shim)
        except Exception:
            logger.warning("draft slot %s free failed", idx, exc_info=True)

    def _mamba_idx(self, pool, req_pool_idx):
        return int(
            pool.get_mamba_indices(
                torch.tensor([req_pool_idx], dtype=torch.int64, device=pool.device)
            )[0]
        )

    def _sync_batch_lens(self, batch):
        lens = [len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs]
        batch.seq_lens = torch.tensor(lens, dtype=batch.seq_lens.dtype, device=batch.seq_lens.device)
        if getattr(batch, "seq_lens_cpu", None) is not None:
            batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
        batch.seq_lens_sum = int(sum(lens))

    # ---- the model forwards (TOWER-VALIDATED SECTION) --------------------- #
    # These build single-request forwards on the draft / target runners outside
    # the scheduler's normal batch flow. The exact ForwardBatch construction on
    # main is what the tower boot-test pins down; see _build_forward.

    def _draft_decode_k(self, req, draft_req_idx, t_last, seq_committed) -> List[int]:
        """K eager greedy decode steps on the draft; returns K token ids."""
        dev = self.draft_runner.device
        vocab = self.draft_runner.model_config.vocab_size
        drafts: List[int] = []
        tok = t_last
        pos = seq_committed  # position of the token being fed (0-indexed length)
        for step in range(self.k):
            out = self._run_decode(self.draft_runner, req, draft_req_idx, tok, pos)
            logits = out.next_token_logits.float()
            nxt = int(logits[0, :vocab].argmax(dim=-1))
            drafts.append(nxt)
            tok = nxt
            pos += 1
        return drafts

    def _target_verify(self, req, target_req_idx, tokens, prefix_len, capture=True):
        out = self._run_extend(self.target_runner, req, target_req_idx, tokens, prefix_len, capture)
        if not capture:
            return None
        return self._lm_head_argmax(out.hidden_states)

    def _draft_extend(self, req, tokens, prefix_len):
        self._run_extend(self.draft_runner, req, self._draft_slot[req.rid], tokens, prefix_len, False)

    def _lm_head_argmax(self, hidden):
        w = self.target_runner.model.lm_head.weight
        vocab = self.target_runner.model_config.vocab_size
        h = hidden.to(w.dtype)
        # per-row [1,H]@[H,V] on purpose: same reduction order as the M=1 decode
        # logits matmul, so near-tie argmaxes match the plain baseline (F0031).
        return torch.cat([
            torch.matmul(h[i : i + 1], w.t()).float()[:, :vocab].argmax(dim=-1)
            for i in range(h.shape[0])
        ])

    def _run_decode(self, runner, req, req_pool_idx, token, pos):
        fb = self._build_forward(runner, req, req_pool_idx, [token], pos, extend=False, capture=False)
        return runner.forward(fb).next_token_logits_holder()  # placeholder; see note

    def _run_extend(self, runner, req, req_pool_idx, tokens, prefix_len, capture):
        fb = self._build_forward(runner, req, req_pool_idx, tokens, prefix_len, extend=True, capture=capture)
        return runner.forward(fb).logits_output

    def _build_forward(self, runner, req, req_pool_idx, tokens, pos_or_prefix, *, extend, capture):
        # TOWER TODO: construct a ForwardBatch for a single-request extend/decode
        # on main. main's ForwardBatch.init_new(batch: ScheduleBatch, runner)
        # consumes a ScheduleBatch — build a minimal one here, pinned against the
        # live API during the tower boot-test.
        raise NotImplementedError("draft/target forward construction — pin on tower")


# --------------------------------------------------------------------------- #
# O(1) recurrent-state snapshot / restore (draft's own rollback; the target's
# is handled upstream by commit_mamba_states_after_verify, but the re-run commit
# path here also restores the target around the verify).
# --------------------------------------------------------------------------- #


def _snapshot(pool, layer_ids, slot):
    conv0, conv1, temporal = [], [], []
    for lid in layer_ids:
        cache = pool.mamba2_layer_cache(lid)
        conv0.append(cache.conv[0][slot].clone())
        conv1.append(cache.conv[1][slot].clone())
        temporal.append(cache.temporal[slot].clone())
    return (slot, torch.stack(conv0), torch.stack(conv1), torch.stack(temporal))


def _restore(pool, layer_ids, snap):
    slot, conv0, conv1, temporal = snap
    for i, lid in enumerate(layer_ids):
        cache = pool.mamba2_layer_cache(lid)
        cache.conv[0][slot].copy_(conv0[i])
        cache.conv[1][slot].copy_(conv1[i])
        cache.temporal[slot].copy_(temporal[i])


class _preserve:
    """Temporarily set a server_args attribute, restore on exit."""

    def __init__(self, obj, attr, value):
        self.obj, self.attr, self.value = obj, attr, value

    def __enter__(self):
        self.old = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.attr, self.old)
