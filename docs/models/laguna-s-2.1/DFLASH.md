# Laguna-S-2.1 DFlash drafter in gufo (phase 6 spec)

Status 2026-10-26: spec, the CPU config and weight index (6a), the CPU oracle
(6b, checked against Poolside's fork), the verify-cost decision data (6b-0),
the drafter's GPU operators with a composed GPU drafter forward (6c, see
"What phase 6c added"; `src/models/laguna/kernels/dflash_ops.*`, tests
`laguna.dflash_ops` and `laguna.dflash_gpu_real`) and the target side (6d,
see "What phase 6d added"; `kernels/dflash_session.*`, tests
`laguna.dflash_target` and `laguna.dflash_target_real`) and the speculative decode loop with serve and
prompt wiring (6e, see "What phase 6e added"; `speculative.*`, tests
`laguna.speculative`, `laguna.speculative_real`, `laguna.serving`).
Acceptance and speed tuning against the fork (6f) is not done. Target
phases 1 to 5 are complete (PLAN.md); this document is the "own feature-layer
and mask verification" that PLAN.md item 6 asks for.

## Evidence and confidence

| Claim | Source | Confidence |
| --- | --- | --- |
| Drafter shapes, tensors, quant types, metadata | `laguna-s-2.1-DFlash-Q4_K.gguf` header, read directly (own parser plus `laguna.dflash`) | confirmed |
| Encoder, K/V injection, block graph, mask, taps, mask token | Poolside's llama.cpp fork, branch `laguna`, 06f8cebd7 (2026-08-04), shallow clone at `/server/staging/laguna/dflash-ref/poolside-llama.cpp`: `src/models/dflash.cpp`, `src/models/laguna.cpp`, `common/speculative.cpp`, `conversion/laguna.py` | confirmed by reading the code, never run |
| Tap numbering, causal block, mask token, per-head gate | HF `poolside/Laguna-S-2.1-DFlash-INT4` `config.json`, `config.py`, `README.md` (read through the HF connector, no weights downloaded) | confirmed |
| Position pairing (feature of token p goes to draft position p; anchor at `n_past`) and the cache clean-up order | `tools/server/server-context.cpp` and `common/speculative.cpp` of the fork | read, not run |
| Numerical behavior of any of it | nothing | not confirmed |

Not confirmed: no drafter forward was executed anywhere (no CPU oracle, no
GPU, no fork run), so the arithmetic below is a reading of the fork's graph.
The training code (Poolside's `speculators` package; vLLM #46853, SGLang
#29446, TRT-LLM #15666 per the card) was not read; whether training used the
same feature-to-position alignment as the fork is inferred from the fork
working at all (27.8% acceptance on a generic prompt, Myric's measurement). The
upstream ggml-org llama.cpp (d854d98, `/server/programming/llama.cpp-laguna`)
has `dflash.cpp` but without `decoder_arch`, `enc.aux_norm`, the gate or the
Laguna K/V norm, and its `laguna.cpp` has no residual taps, so it cannot run
this drafter (the model card says the same).

## Drafter architecture (header, 76 tensors, 652,160,384 bytes)

Arch `dflash`, `dflash.decoder_arch = laguna`, `general.file_type` 15 (Q4_K_M
mix: 27 F32, 45 Q4_K, 4 Q6_K).

| Item | Value |
| --- | --- |
| Layers | 6, all sliding-window (`sliding_window_pattern` all 1, window 512) |
| Hidden / FFN | 3072 / 12288, dense SwiGLU, no MoE |
| Heads | 72 query, 8 KV (group 9), head dim 128 on every layer |
| RoPE | one rope: NeoX, all 128 channels, theta 500000, `rope.scaling.type none` (no YaRN). Differs from both target layer kinds |
| Block | `block_size` 16: rows `[anchor, MASK x 15]`; at most 15 drafts |
| Mask token | `tokenizer.ggml.mask_token_id` 12 = `〈|MASK|〉` (the header manifest in `outputs/` omits this key) |
| Vocabulary | the drafter ships the target's 100352-token vocabulary and has no `output.weight` or `token_embd.weight` |
| Taps | `dflash.target_layers` = [2, 11, 20, 30, 39, 48] |
| Norm eps | 1e-6 |

Tensors: global `enc.aux_norm.weight` [3072, 6] F32, `enc.output_norm.weight`
[3072] F32, `fc.weight` [18432, 3072] Q4_K, `output_norm.weight` [3072] F32;
per layer `attn_norm`, `attn_q` [3072, 9216], `attn_k`/`attn_v` [3072, 1024],
`attn_output` [9216, 3072], `attn_q_norm`/`attn_k_norm` [128], `attn_gate`
[3072, 72] (Q4_K; the target's v2 gate is F32), `ffn_norm`, `ffn_gate`/`ffn_up`
[3072, 12288], `ffn_down` [12288, 3072]. `attn_v` and `ffn_down` are Q6_K in
layers 2 and 5, Q4_K elsewhere. The drafter reads the target's `token_embd`
(noise embeddings) and `output` (head), so hidden size and vocabulary must
match the target; `DFlashConfig::ValidateAgainstTarget` checks that.

## Tap convention (which hidden states, which indices)

`target_layers` uses **layer-input numbering, 1-based on the layer output**:
entry N is the residual stream entering target layer N (zero-based), which is
the output of zero-based layer N - 1. Entry 48 equals the layer count and means
the stream after the last layer, before the final `output_norm`. For
Laguna-S-2.1 the six taps are the outputs of zero-based layers
**1, 10, 19, 29, 38, 47**; the embedding output (input of layer 0) is never
tapped. All taps are the full residual stream, F32 in gufo (`h` after the
residual add of each layer, the same point as the executor's `layer_tap_`, and
the CPU oracle's `l_out-N` tap).

How this was confirmed (three independent sources, no numeric tap diff yet):

1. The fork's `src/models/laguna.cpp:176` stores `t_layer_inp[il] = inpL` at
   the top of loop iteration `il` (the input of layer `il`) and `:334` stores
   the post-loop pre-norm stream as `t_h_nextn`. `common/speculative.cpp:982`
   routes a tap equal to `llama_model_n_layer(target)` to the nextn path and
   every other tap to `layer_inp[tap]`. Laguna's comment: "input of layer
   n_layer in the training convention".
2. The HF config of the DFlash model carries both spellings:
   `dflash_config.target_layer_ids` [1, 10, 19, 29, 38, 47] with
   `num_target_layers` 48, and `eagle_aux_hidden_state_layer_ids`
   [2, 11, 20, 30, 39, 48]. The GGUF `target_layers` equals the second list
   (the converter writes id + 1).
3. gufo's Qwen path already normalizes the same GGUF key to zero-based layer
   outputs (`src/models/qwen/dflash_weights.cpp:257`: "layer-input indices ...
   normalize N to zero-based N - 1"). `DFlashConfig::tap_layer(i)` does the same
   for Laguna and `IDraftBackend::TargetHiddenLayerIds()` ("exact zero-based
   target layer outputs") can carry it unchanged.

The per-token encoder input is the six taps concatenated in list order:
`feat[t] = [tap0 | tap1 | ... | tap5]`, 6 x 3072 = 18432 floats.

## How the drafter consumes them

Encoder, per target token (no dependence between tokens; fork `dflash.cpp:125`
and the encoder graph):

```
f_k  = feat[t][k*3072 .. (k+1)*3072)                 k = 0..5
a_k  = RMSNorm(f_k, eps 1e-6) * aux_norm[k]          weight row k of [3072, 6]
u    = fc . concat(a_0 .. a_5)                       18432 -> 3072
fused[t] = RMSNorm(u) * enc.output_norm
```

Context K/V injection (embedding-input batch at the target positions, one row
per target token; fork `:196`): for each drafter layer `l`

```
x = RMSNorm(fused[t], blk.l.attn_norm)       (Laguna: the layer's input norm;
                                              the generic DFlash projects raw)
K = RMSNorm_head(x Wk, attn_k_norm); K = RoPE_NeoX(K, pos = target position)
V = x Wv                                      (no V norm, no RoPE)
```

K and V go into layer `l`'s cache at the target position. The injected rows are
only K/V: nothing else of the encoder pass is kept.

Block (token-input batch; fork `:236`): rows `r = 0..B-1` at positions
`n + r`, tokens `[id_last, MASK, ...]`, where `n` is the target's committed
length (`slot.prompt.n_tokens()`) and `id_last` is the sampled token not yet
fed to the target. `B = n_draft + 1`, so a short block is just fewer rows.

```
h = token_embd_target[token]                  (no scale)
per layer l:
  n  = RMSNorm(h, attn_norm)
  q  = RMSNorm_head(n Wq, q_norm), RoPE;  k = RMSNorm_head(n Wk, k_norm), RoPE;  v = n Wv
  o  = softmax(q . [ctx K ; block K]^T / sqrt(128)) . [ctx V ; block V]   (GQA 9, mask below)
  g  = softplus(n Wg)  per head, scales that head's 128 outputs;  o = (o*g) Wo
  h  = h + o ;  h = h + SwiGLU(RMSNorm(h, ffn_norm))
logits_r = RMSNorm(h, output_norm) . output_target        rows r = 1..B-1
draft[r-1] = argmax(logits_r)
```

Row 0's output is discarded; row `r` predicts the token **at its own position**
`n + r` (diffusion-style, every MASK row predicts in parallel, not the next
token). The fork keeps `top_k = 10` and takes the top candidate, with
`p_min = 0` it never truncates the block.

Attention mask (fork sets `llama_set_causal_attn(ctx_dft, true)` for
`decoder_arch laguna`; HF `dflash_config.causal` true; generic DFlash and
gufo's Qwen DFlash2 are non-causal inside the block and so differ):
row `r` at position `p = n + r` sees context keys `j < n` with `p - j < 512`
(that is `511 - r` context rows, `LLAMA_SWA_TYPE_STANDARD`, the same rule as
the target's sliding layers) and block keys `0..r` (causal). HF's
`sliding_window_base` defaults to `moving_query` in the card's `config.py`,
the same as this (each row's lower bound follows its own position; the legacy
`fixed_anchor` is not used). Consequence: row `r` does not depend on later
rows, so any shorter block gives the same drafts as the prefix of a longer
one.

KV handling in the fork (the contract to reproduce):

- The drafter cache holds only injected rows. The block's own K/V is written
  by the draft pass and removed again before the next step
  (`server-context.cpp:2980`, `seq_rm(ckpt.pos_max + 1, -1)` right after the
  draft pass, where `pos_max` is the target's last position, `n - 1`).
- Features are injected for **every verified row**, including rows later
  rejected (`speculative.cpp:1133`, one row per verify-batch token); after
  acceptance the cache is trimmed to the committed length
  (`server-context.cpp:3892`). So after a step with `a` accepted drafts the
  cache covers positions `< n + a + 1`, and the new anchor (the correction or
  bonus token) sits at `n + a + 1` with no feature of its own.
- Prompt prefill injects every prompt row (`process()` runs on each prefill
  ubatch).
- The cache is a sliding-window ring: window 512 plus the largest batch
  written at once.

Because the drafter is all sliding-window and the encoder has no cross-token
state, a position's K/V is a function of that position's six taps alone, and
nothing older than the 511 rows before the anchor is ever read. gufo needs to
inject only the **last 511 prompt rows**, not all of them (the Qwen path
captures its prompt features into bounded pinned host memory, b0eb5d47; the
Laguna case needs far less). Ring capacity `512 + max(16, inject chunk)` rows x 6
layers x 2 x 1024 x 2 B (F16) = 24 KiB per row, 13 to 25 MiB per session.

## Reuse versus Laguna-specific

Existing code (searched: `src/core/speculative`, `src/models/qwen`,
`src/cli`, `git log` for dflash). DFlash exists only for Qwen 27B (DFlash2, the
z-lab `Qwen3.8-27B-DFlash2-GGUF`; commits e4a219ef, b0eb5d47, da5d5a2b,
e4c661e6) plus a DSpark path in DeepSeek V4. `QwenDFlashWeights::LoadFromGguf`
is **not** generic: it requires the DFlash2 conv and selector keys,
`attention.causal` false, `selector_*` tensors, and non-causal topology, and
the Laguna file has none of them (and has `enc.aux_norm` and `attn_gate`,
which Qwen's parser never binds), so Laguna got its own parser.

| Piece | Verdict |
| --- | --- |
| `IDraftBackend`, `DraftProposal`, `DraftTargetContext`, `DraftProposalRequest`, `ProposeBatch`, snapshot interfaces (`src/core/speculative/draft_backend.hpp`) | reuse as is. They carry the tap list (`TargetHiddenLayerIds`), anchor and position, and sampled proposals with per-token probabilities |
| `SpeculativeVerifier` (`speculative_verifier.{hpp,cpp}`): step loop, acceptance, sampled rejection sampling, rolling acceptance, snapshots | reuse the algorithm; it is written against `ISpeculativeTargetExecutor` (HIP only, Qwen includes) and its commit path re-forwards accepted tokens, which Laguna does not need. A `LagunaSpeculativeTarget` adapter must implement the interface (see below) |
| `DFlashLengthController` (`dflash_policy.hpp`) | reuse the controller shape; its cost tables are measured on the dense 27B and must be re-measured, because verify cost on the MoE depends on width (below) |
| `TextSpeculativeConfig`, `--speculative`, `--dflash-model` plumbing | extend; today the options are described and validated as Qwen DFlash2 |
| Qwen DFlash GPU kernels (non-causal block attention, convs, selector) | not reusable; Laguna needs a causal SWA block attention, no convs, no selector |
| `DFlashConfig`, `DFlashWeights` (`src/models/laguna/dflash.*`) | new, done (CPU) |
| `detail::TensorBinder` (`tensor_binder.hpp`) | factored out of `weights.cpp` so both indexes share it |
| Target feature taps | new: production device-side capture (the executor's `layer_tap_` synchronizes the stream and is a debug facility) |

Laguna-specific rules the target must satisfy (all verified against the
current tree, none run):

1. **Rewind.** `rocm::Executor::Forward` only runs at `position()`; there is no
   rewind. `AttnOps::Forward` already accepts `start_pos < committed()`. Add
   `Executor::RewindTo(position)` (host bookkeeping only). Full layers need
   nothing; rings need the 2b9acd74 retained-history rule: a call is accepted
   only when every key its rows read before `start_pos` lies in
   `[retained_begin(), committed())`. Ring capacity is `512 + max_chunk`, and
   verifying `w <= max_chunk` rows from `n` leaves
   `retained_begin = n + w - capacity`, so rewinding to any `c` in `[n, n + w]`
   passes when `max_chunk >= 16` (`c - 511 >= n + w - 512 - max_chunk`). The
   rewind must still go through that validation, not around it. Serve runs
   chunk 512, so this holds with margin; a test must hold it at the smallest
   configured chunk too.
2. **Commit without replay.** The default `CommitVerificationChunk` of the
   Qwen verifier re-forwards committed tokens after a restore (the Qwen target
   has SSM state to replay). Laguna is attention only: after a
   verify pass the accepted rows' K/V are already correct, so the commit is
   `RewindTo(n + a + 1)` and nothing else (override `CommitVerificationChunk`
   to a rewind).
3. **Verify path and bit identity.** `Forward` picks the vector (MMVQ) path for
   <= 8 rows and the batched (MMQ) path above, which round activations
   differently; a 16-row verify would therefore not equal solo decode, and
   exact-greedy equivalence with plain decode would fail on near ties (PLAN.md's
   ubatch 8 vs 256 floor, 94.78% top-1, is a comparable rounding effect). Pin
   the vector path for verify chunks (<= 32 rows): `Path` is today an
   `ExecutorOptions` field fixed at creation, and `Path::kVector` rejects
   `max_chunk > 32`, which conflicts with 512-token prefill. Needs a per-call
   path argument on `Forward` (small, GPU-tested).
4. **Invalidation (2b9acd74).** A failure after a call's K/V append marks that
   cache invalid and poisons the executor until `Reset()`. A failed verify
   poisons the target session as today and the drafter state of that request
   is dropped with it. A drafter-side device failure must never poison the
   target: drop the drafter for the request and continue plain decode. The
   drafter ring needs the same retained-history bookkeeping (it is rewound by
   trimming rejected rows) and its own validity flag.
5. **Snapshots.** `SessionSnapshot` stores target KV only. The drafter cache
   depends on taps that are not stored, so a restored session has no drafter
   context. Snapshot the drafter rings with the session (13 to 25 MiB, host
   copy; plus the pending anchor and its controller state), or draft nothing
   until 511 new rows exist. Prefer the first; the second silently loses the
   cache-reuse speedup. `--cache-disk` is already refused.
6. **Batched decode.** `Executor::DecodeBatch` appends one token per session;
   a speculating session appends `w` rows. First version: speculating sessions
   verify alone (serial `Forward`), non-speculating sessions keep
   `DecodeBatch`, and the runner must not put a speculating session in a
   `DecodeBatch` group. A ragged multi-row batched pass (the vector path takes
   32 rows in total, so two 16-row sessions) is a later, separately measured
   item. Drafter blocks of several sessions can share projections through the
   existing `ProposeBatch`. Each session keeps its own drafter ring, scratch
   and stream, as the target does.
7. **Sampling.** The `kLaguna` preset is temperature 1.0, top_k 20, so default
   chat requests are sampled. Block rows are independent given the context,
   so the proposal distribution of row `r` is exactly the one sampled and the
   verifier's lossless rejection sampling applies unchanged; but acceptance at
   temperature 1 will be far lower than the greedy numbers. v1 can be
   greedy-only (temperature 0 and constrained/grammar decoding must be checked
   against the verifier: JSON tool-call constraints advance per token, so
   verification must run the constraint through the draft prefix).
8. **Residual magnitude.** The fork documents rows with `|x| ~ 1e6` in the
   pre-final-norm stream (attention-sink tokens) and clamps non-finite values.
   gufo keeps `h` in F32, so the per-tap RMSNorm sees finite values; never
   store taps in F16/BF16. Treat a non-finite tap row as a drafter failure for
   that request (drop the drafter), not as something to clamp.

## Cost note that gates the GPU work

On this MoE a verify chunk of `w` rows touches about `256 (1 - (1 - 10/256)^w)`
distinct experts per layer if routing were independent: 10 at w = 1, 38 at 4,
70 at 8, 121 at 16, so expert weight traffic per step grows roughly 4x, 7x,
12x over a decode token before the drafter's own 0.61 GiB per pass is added. A
one-sample measurement from the CPU oracle taps
(`/server/staging/laguna/oracle-20261025-capital-cpuq8`, 9 prompt tokens,
top-10 of `ffn_moe_probs_biased`, layers 1 to 47): 50.8 distinct experts per
layer on average (min 40, max 69) against 77 expected if independent, so real
routing overlaps, but 9 prompt tokens of one short prompt say little about
code generation. Myric's measurement shows the effect: 27.8% acceptance of 15
drafted tokens gave only 1.20x on a DGX Spark; Poolside's 2.5 to 3.7x are on
TP=2 GPUs at coding prompts (acceptance length 4.2 to 6.8). Whether DFlash
pays on gfx1151 is therefore not known and decides the project before any
kernel is written (step 6b below).

## Test plan (oracle first, then operators, then end to end)

Status column: done, CPU (can run now), GPU (needs the paused GPU window).

| Step | Check | Needs |
| --- | --- | --- |
| 6a | Config and weight index: `laguna.dflash_contract` (synthetic header and 76-tensor list, positive and negative: missing, extra, duplicate, wrong shape and type, own head or embeddings, per-element gate, sliding pattern, tap list, target pairing) and `laguna.dflash` (real header: config values written independently, 76 of 76 tensors bound, type census 27/45/4, pairing with the real target header) | done, CPU |
| 6b-0 | Decision data before any GPU kernel: (a) routing overlap for verify windows of 2, 4, 8, 16 consecutive tokens on real code/chat generations from the CPU oracle (`ffn_moe_probs_biased` top-10 per layer, distinct experts per window); (b) with Poolside's acceptance lengths (4.2 to 6.8 at n 15) and Myric's 27.8%, compute the break-even step-cost ratio; (c) target verify cost at w = 2, 4, 8, 16, vector path, per context depth. (a) and (b) are CPU. (c) is the first GPU item and is a plain `Forward` timing, no drafter | (a) (b) CPU; (c) GPU |
| 6b | CPU oracle `DFlashReference` in `reference.*` style: F32 dequantized weights and a Q8-activation-aware mode (kernel arithmetic versus expected quantization error, as phase 2). Taps named after the fork's callbacks: `enc_aux_norm`, `fc_out`, `enc_norm_out`, `kv_inp_normed-N`, `Kcur_injected-N`, `Vcur_injected-N`, `noise_norm-N`, `Qcur-N`, `Kcur-N`, `attn_gate_softplus-N`, `attn_gated-N`, `ffn_out-N`, `l_out-N`, `result_norm`, `result_output`. Tiny random fixtures with unequal dimensions (so a transposed projection or a swapped tap cannot pass), including a block at the window edge (anchor at 511/512/513, rows crossing the ring wrap) and a single-tap case. Real feature fixture: the oracle run above already holds `l_out-1, 10, 19, 29, 38, 47` for 9 tokens | CPU |
| 6b-ref | Independent reference for the semantics: a small harness over the fork's `libllama` (CPU) that loads the drafter plus the target (mmap touches only `token_embd` and `output`), feeds recorded features through `llama_encode`/`llama_decode` and dumps the callback taps. Diff 6b against it layer by layer. This is what turns "read the fork" into "confirmed"; it needs no GPU and no target forward. Pin its revision (06f8cebd7) and build it under `/server/staging/laguna/dflash-ref/` | CPU |
| 6c | GPU operators one at a time against 6b: encoder (per-tap RMSNorm, `fc` Q4_K 18432 to 3072, norm); K/V injection (norm, K/V, K-norm, RoPE, ring write; positions 511/512/513, wrap); block attention (GQA 9, window plus causal block, row `r` sees `511 - r` context rows, ring capacity, shorter blocks equal prefixes of longer ones); gate softplus; SwiGLU with Q4_K and Q6_K; final norm, head and argmax; every projection both vector and batched path. Reuse the `qfn_mmq` raw kernels with one expert and zero ids, as the target's head does | done, GPU |
| 6d | Target side: device-side tap capture of the six layers for prefill, decode and verify rows (untapped reruns too: debug taps hide stream races); `Forward(kAll)` with the pinned vector path equals the same tokens decoded alone, bit for bit, across 511/512/513 and ring wrap; `RewindTo` at every depth 0 to 16 at those boundaries; rewind past retained history is refused and leaves the cache untouched; an injected failure after the append poisons target and drops the drafter, and a drafter failure leaves the target usable | GPU |
| 6e | End to end, exact greedy: speculative output equals plain greedy output token for token (same binary, same pinned path) on a prompt set that straddles the window (lengths 509 to 515, 2k, 32k), thinking on and off, tool-call grammar, stop tokens (EOS 2, EOT 24) inside a draft, `max_tokens` inside a draft, cancel mid-step, snapshot and restore mid-request, retry, two and eight sessions with speculating and plain sessions mixed, load and unload. Any divergence is explained by logit gap before it is waved through | GPU |
| 6f | Acceptance and speed: acceptance per position and mean accepted per step against the fork on the same prompts (the fork is deterministic at temperature 0; run it on the same APEX target file; both sides report `draft_n_accepted / draft_n` the same way); set the tolerance from the fork's own run-to-run spread, not from a guess. Speed: unprofiled, counterbalanced A/B against plain decode on matched Release binaries, per concurrency, as PLAN.md phase 5 requires. Include the sampled path (temperature 1.0, top_k 20) once the lossless verifier is wired | GPU |

Each step lands as its own commit. No later step starts until 6b-0 (a)(b)(c)
say the drafter can pay.

## Open questions

1. Does the verify cost curve on gfx1151 leave room for a net gain at any
   acceptance Laguna actually reaches? (6b-0.)
2. Pinned vector path for verify (bit identity, needs per-call path) versus the
   faster batched path with a measured divergence budget. PLAN.md's own floor
   argues for pinning; the cost of 16 rows on the vector MoE path is unmeasured.
3. Which block length does the adaptive controller start at (Poolside serves
   7, benchmarks 15)? Causal rows make shorter blocks free to compute.
4. Drafter KV in snapshots (13 to 25 MiB per snapshot) versus a cold drafter
   after restore.
5. Greedy-only first, or the sampled verifier in v1, given the default preset
   is temperature 1.0.
6. Feature alignment (token p's taps at draft position p, anchor at `n_past`)
   is read from the fork, not from training code. 6b-ref and the acceptance
   comparison in 6f are the only things that can settle it.
7. Drafter quantization: Myric reports BF16, Q8_0 and Q4_K_M with the same
   acceptance; only the Q4_K file is on this box (0.61 GiB). Acceptable, but
   the oracle should also be run against it dequantized and not only against
   a BF16 reference nobody has here.
8. The fork is a branch of upstream with Laguna taps; its license and whether
   gufo may quote anything from it (nothing is copied here; this spec
   describes behavior).

## What phase 6a added

- `src/models/laguna/dflash.hpp`, `dflash.cpp`: `DFlashConfig::FromGguf` (strict
  `dflash.*` keys, vocabulary from `tokenizer.ggml.tokens`, profile check
  against the Laguna-S-2.1 drafter, `tap_layer(i)` zero-based outputs,
  `ValidateAgainstTarget`), `DFlashWeights::Bind` (name, shape and format of
  all 76 tensors, no leftovers).
- `src/models/laguna/tensor_binder.hpp`: `TensorBinder` and the format table
  moved out of `weights.cpp` unchanged so the two indexes share them.
- Tests `laguna.dflash_contract` (model-free) and `laguna.dflash` (real header,
  exit 77 without the file).

## What phase 6b added (2026-10-26, CPU only, no GPU used)

- `src/models/laguna/dflash_reference.{hpp,cpp}`: `reference::DFlashModel`,
  the drafter forward in F32 (or the target oracle's activation-quant and F16
  K/V modes), built on the target oracle's helpers. `reference::ForwardWithTaps`
  (in `reference.*`) gathers the `l_out-N` rows of chosen target layers
  (1, 10, 19, 29, 38, 47 = `DFlashConfig::tap_layer(i)`), and
  `GqaAttentionAt` is the oracle's attention with explicit positions.
- Tests: `laguna.dflash_reference` (tiny random models with unequal dimensions
  against an independent naive double implementation: window edge, context and
  block positions, in-block causality, tap names, tap-index convention with a
  tap one layer early failing), and the opt-in `laguna.dflash_real_smoke`
  (`GUFO_LAGUNA_DFLASH_REAL=1`, needs MemAvailable over 90 GiB).
- 6b-ref: `tests/models/laguna/fork_compare/` (CPU-only build of the fork at
  06f8cebd7 under `/server/staging/laguna/dflash-ref/build-cpu`). On the real
  drafter and an 8-token prompt the fork's encoder output matches the oracle
  to 1.2e-6 relative RMS when the oracle uses the CPU activation quantization,
  and the block Q, K, V and gate taps of layer 0 to 4e-7. The first
  difference is the attention output of layer 0 (3e-4, uniform over the 16
  rows, so a rounding effect, not a mask or position fault), which the
  quantized projections then amplify to 1 to 5 percent on the draft logits.
  Top-1 of the 15 drafts agrees at 14 of 15 positions (position 14: fork 350,
  oracle 81; not examined further). Not yet explained: the 3e-4
  attention difference.
- Cause of the 3e-4 (2026-10-26, CPU only): the fork's CPU flash-attention
  kernel (`ggml_compute_forward_flash_attn_ext_f16`, ops.cpp) keeps the V
  accumulator in F16 (`VKQ16`, rounded after every rescale and every key); the
  oracle accumulated in F32. The fork's GPU kernels accumulate in F32, so the
  target oracle's F32 attention is the right default for gufo and the
  F16 accumulator is only a comparison mode (`Options::fa_f16_accumulator`,
  `GUFO_LAGUNA_DFLASH_FA16ACC=1` in the smoke test). Evidence (8-token prompt,
  layer-0 `attn_gated`, oracle in CPU activation-quant mode with F16 K/V):
  fork flash-attn on 3.3e-4; fork `-fa off`, F16 cache 1.4e-4; F32 cache 2.3e-4
  (the oracle rounds Q/K/V through F16, so F16 storage is closer); with the
  F16 accumulator mode the same tap drops to 3.9e-5 when the oracle's encoder
  output is injected into the fork (`FORK_INJECT=oracle`). A second, separate
  amplifier: the encoder's 1.2e-6 difference becomes 9e-5 on the injected K
  and 1.2e-4 on V (the Q8_K activation quantization of `kv_inp_normed` flips
  rounding steps), so with the fork's own encoder the tap is 1.5e-4. The
  remaining 1 to 5 percent on the draft logits is that quantized-projection
  amplification, not attention. The first prompt's position 14 miss was a
  near tie (oracle logit gap 0.021).
- Top-1 of the 15 drafts against the fork (FA on, F16 accumulator mode):
  8 tokens 15/15 (was 14/15), 120 tokens (code) 15/15, 300 tokens (prose)
  15/15, 600 tokens (window 512 in play) 14/15 (row 11, logit gap 0.23 at 2 to
  3 percent logit noise). Four prompts, one drafter file: this confirms the
  graph and the numerics, not the acceptance rate.

## What phase 6c added (2026-10-26, GPU, no target integration)

Code (`src/models/laguna/kernels/dflash_ops.{hpp,hip.cpp}`, laguna only; no
shared code changed):

- `DeviceDrafter::Upload`: the drafter on the device (Q4_K/Q6_K projections
  with ggml's 512-column row padding, F32 norms), and `DrafterWeights`, a plain
  device view (tests copy it and change one field to prove the checks notice).
- `DrafterKv`: one F16 ring per layer, capacity `window + max(max_rows,
  block_size)` (544 at 32-row injections, 528 at 16), with the target cache's
  retained-history bookkeeping (`committed`, `retained_begin`, `valid`) plus
  `floor`, the first position the session injected: keys below it never
  existed and are not attended, as in the fork's cache, which holds only
  injected rows. A failure after a write invalidates the ring until `Reset`.
- `DFlashOps` (one per stream, all scratch per instance, allocate before free):
  `Encode` (per-tap RMSNorm with the `aux_norm` rows, `fc`, norm), `InjectKv`
  and `Inject` (attn_norm, K/V, K norm, NeoX RoPE theta 500000, F16 store at the
  target positions; a start below `committed` overwrites, as trimming rejected
  rows will), `Attention` (Q/K/V/gate projections, norms, RoPE, the block's K/V
  written into the ring at `n .. n + B - 1`, GQA attention where row `r` reads
  keys `max(p + 1 - window, floor) .. p`, the softplus gate, the output
  projection), `Ffn` (SwiGLU, Q6_K down in layers 2 and 5), `Head` (output
  norm, the target's head on rows 1..B-1, first-maximum argmax) and `Block`
  (the composed block, 2 <= B <= 16). Every projection is `qfn_mmq_moe_vec` or
  `qfn_mmq_q{4,6}_K_moe_raw` with one expert and zero ids, on either path. The
  attention kernel is the target's tiled online-softmax kernel with the
  drafter's key range, so a row's sum order depends on its position only.
- KV contract as the fork: a block pass leaves `committed()` at `n` (the next
  injection overwrites its rows) and records in `retained_begin()` what its
  rows evicted; the ring's spare rows (`max(16, chunk)`) are what keeps the
  block from evicting a key any block row reads.
- `DrafterSession`: the composed drafter forward without the target. It owns
  its `DFlashOps`, ring and device buffers on the caller's stream, injects host
  features (device-side capture is 6d), gathers `[anchor, MASK x (B - 1)]`
  from the target's mapped `token_embd` on the host, as the executor does, and
  returns the drafts and logits.

Tests (`ctest -R '^laguna\.dflash_(ops|gpu_real)$'`, run under the GPU lock):

- `laguna.dflash_ops`, 124 cases, nothing skipped: tiny random drafters with
  unequal dimensions (hidden 512/768, FFN 768/512, GQA 9 over 2 KV heads and GQA
  4 over 3, taps 3/2, window 512/37, theta 500000/10000, vocabulary 1000/777,
  Q6_K `attn_v`/`ffn_down` in the last layer). Every projection is checked
  against the oracle's exact product of its captured, independently validated
  Q8 input within the float accumulation bound (worst 6.8e-2 of the bound), so
  kernel arithmetic is judged against the expected quantization and nothing
  else; projections of one input must share one quantization. Norms, rope
  (2e-7 to 3e-7 of the head's largest element), stored K/V (half an F16 ulp),
  attention against a double oracle over the ring as stored (1.7e-6 to 5.5e-6
  of rms, within 3x plain float), gate, SwiGLU and residuals (bitwise) are checked
  from the GPU's own upstream stages. Window 511/513 and non-causal probes lie
  3e4 to 3e6 times further from the GPU than the correct mask. Covered: anchors
  at 9, 36/37/38, 511, 512, 513, 600, 700; injections crossing 511/512/513 and
  wrapping the ring; a session floor; blocks of 1 to 16 rows (shorter blocks
  equal the prefix of 16 bit for bit, on one path); overwrite equal to direct
  injection; injection over a block's rows equal to a ring that never held
  them; untapped and repeated runs bitwise; refused scratch growth; two
  sessions on two streams concurrently equal to their solo runs. 22
  mutations fail as they must (swapped taps, rotated aux_norm rows, swapped
  norm weight, halved theta, theta 10000, swapped K/V, rows one position late,
  window 511 and 513, non-causal block; each on both paths) and 11 bad
  arguments are refused without touching the ring (gap, start below floor,
  oversized chunk or block, 33 vector rows, other window, layer out of range,
  evicted keys after a rewind, block of 1 or 17, head of another hidden size,
  invalid ring).
- End to end the two sides quantize their own intermediates, which differ by
  float noise, so codes at rounding ties flip and the flips compound: on one
  stage with identical input the GPU is about 1e-7 of rms from the oracle in
  its mode (up to 2.4e-4 where a tie flips; Q8 noise 1e-2), after one layer
  0.4 to 0.7 of the noise, after three 0.6 to 1.1 of it, as the oracle's own
  Q8_1 and MMQ modes drift apart. The composed bound is therefore a noise-scale sanity bound (twice the
  larger of the noise and the Q8_1-vs-MMQ distance); the stage checks judge the
  arithmetic.
- `laguna.dflash_gpu_real` (skips 77 without the files): the real drafter
  (0.61 GiB on the device) and the target's head (0.24 GiB) with `token_embd`
  from the mapping, fed the CPU oracle's recorded `l_out-1, 10, 19, 29, 38,
  47` of the 9-token capital prompt (`oracle-20261025-capital-cpuq8`, anchor
  340 = its top-1) and a 600-row context built from those rows (cycled, 3 %
  noise; real long-context taps need 6d), both paths, all stage checks as
  above (projections within 2.3e-2 of their bound, attention 4.9e-6 to 7.0e-6
  of rms). Logits against the oracle
  in the same mode: 2.2e-2 relative rms (fixture vector 2.2e-2, batched
  2.7e-2; long 2.2e-2 both), max 0.33 to 0.44 logits, against 2.3e-2 to
  3.0e-2 of quantization noise and 2.2e-2 between the oracle's own two modes.
  Drafts: fixture 14/15 on both paths (row 11: GPU 81, oracle 395, oracle
  gap 0.080 vector and 0.40 batched, GPU gap 0.02; the 6b fork comparison
  also had its only miss at row 11 of the long prompt), long context 15/15 on
  both paths. For scale, GPU against the F32 oracle: 12 to 14 of 15. The
  session's untapped result equals the tapped block bit for bit; blocks of 4
  and 8 equal the prefix of 16.
- Regression: the whole `ctest -R '^laguna\.'` at a0158ead, 25 passed, 2
  skipped (`dflash_real_smoke` and `functional` are opt-in). The laguna test
  helper `tests/models/laguna/ffn_gpu.hpp` gained Q4_K in `RandomQuant`;
  `ffn_ops`, `ffn_real`, `attention_ops` and `attention_real` rerun and pass.
- GPU use: 09:34 to 10:05 EDT 2026-10-26 in short runs (seconds to five
  minutes each, one process at a time under the GPU admission lock); logs in
  `/server/staging/laguna/dflash-6c/`.

Open after 6c:

1. No timing was taken. The drafter's cost (6b-0 estimated 6.2 ms per pass
   from bandwidth) is unmeasured; the operators favour clarity (separate Q, K,
   V, gate and gate/up projections, each quantizing its input; host embedding
   gather and a synchronizing feature upload in `DrafterSession`).
2. A K/V row's rounding depends on the path that injected it: prompt rows
   injected in batched chunks and verify rows injected on the vector path
   round differently (each is a valid quantization; drafts, not outputs,
   depend on it). Within one path injection is chunking-invariant.
3. Under `Path::kAuto` blocks of 9 to 16 rows take the batched path and
   shorter ones the vector path, so a block of 16 is not the prefix-extension
   of a block of 8 across that switch. Pin the path per call in 6e.
4. Real-weight coverage has 9 recorded tap rows only; the 600-row case uses
   synthetic features. Real long-context taps come with 6d.
5. Row 11 of the fixture block is a near tie on every comparison so far (fork,
   oracle modes, GPU).

## What phase 6d added (2026-10-26, GPU, target side; no speculative loop)

Code (laguna only; no shared code changed):

- `Executor` (`kernels/executor.*`): `Forward(tokens, rows, logits,
  ForwardArgs)` with a per-call `path` (rule 3: a verify of up to 32 rows on
  the vector path while 512-row prefill stays batched; 33 vector rows are
  refused up front) and `features`, a device buffer that receives the outputs
  of the layers set by `SetFeatureTaps` (F32, `[T][taps x hidden]`) through
  one copy kernel per tapped layer on the executor's stream, with no
  synchronize. `RewindTo(position)` (rules 1 and 2) is host bookkeeping that
  checks every layer with `KvCache::RejectContinue(position, 0)`, the same
  function `AttnOps` now validates every call with (factored out of
  `AttnOps::Run`, messages unchanged), and refuses touching nothing.
  `DecodeBatch` refuses an executor with feature taps (rule 6).
- `DrafterKv::RejectContinue` and `RewindTo` (the drafter ring's own
  retained-history rule, also used by `DFlashOps` for every block);
  `DrafterSession::InjectDevice` (device features, no copy, no synchronize).
- `DFlashSession` (`kernels/dflash_session.*`): one session's executor
  (borrowed) and drafter (owned, on the executor's stream, so capture and
  injection are ordered by the stream alone). `Prefill` injects only the last
  511 rows a block can read; `Verify` runs on the vector path and injects every
  row; `Commit` rewinds both; `Propose` drafts on the batched path (pinned, so
  shorter blocks stay prefixes; see the timing below). Rule 4: a target call
  that fails after its append poisons the executor and drops the drafter; a
  drafter failure (device error in an injection or block, refused rewind, a
  non-finite tap row) drops only the drafter and the target continues as
  plain decode until `Reset`. Non-finite taps (rule 8) are flagged by a device
  kernel as rows are injected, because the drafter's activation quantization
  turns a NaN row into zeros and drafts would otherwise look normal (found by
  the test: an argmax check alone does not catch it).

Tests (`ctest -R '^laguna\.dflash_target'`, under the GPU lock):

- `laguna.dflash_target` (tiny random target with window 512 and a tiny
  drafter of the real head geometry; nothing skipped): captured taps equal the
  debug layer tap bit for bit for prefill (batched), decode and verify
  (vector) rows across 511/512/513; capture leaves logits and untapped reruns
  bit-identical; taps one layer early fail. A 16-row vector verify equals the
  same tokens decoded alone bit for bit and `RewindTo` at every depth 0..16
  gives a continuation equal to a twin that never saw the rejected rows
  (logits and retained K/V bytes), 425 cases at starts 480..528 and 1040..1056
  with chunk 16 (the smallest chunk that admits a 16-row verify; ring 528) and
  495..1024 with chunk 512; a batched verify is told apart. Refused without
  touching anything (bookkeeping and K/V bytes): rewind ahead, one past the
  ring's depth, past evicted keys, a chained rewind (10 then 8) that needs a
  lost key, a poisoned executor, 33 vector rows. Drafter ring: a rewound ring
  drafts exactly as one that never held the rejected rows (25 cases around
  511/512/513 and 600); positions outside `[floor, committed]`, evicted
  context and an invalid ring are refused. `DFlashSession`: drafts from
  device-captured taps equal drafts from the same taps uploaded from the host
  with another injection chunking (prompts 9, 600, 1030, four propose, verify,
  commit steps); the failure rules above; `DecodeBatch` exclusion; two
  sessions on two threads equal their solo runs.
- `laguna.dflash_target_real` (real target and drafter; skips 77 without the
  files). Fixture (9-token capital prompt, oracle in CPU activation mode):
  prefill taps 4e-3 to 1.5e-2 relative rms from the oracle's `l_out-1, 10, 19,
  29, 38, 47`, decode-row taps the same (9.4e-5 from the prefill taps); taps
  one layer early, and an executor tapping layers 0, 9, ..., are 7.8x to 2e5x
  farther. GPU top-1 340 = oracle. Drafts from the GPU taps against drafts
  from the oracle's taps (both through the GPU drafter): 10/15 (rows 6, 10,
  11, 12, 14; logits 0.19 relative rms; the drafter amplifies the 1e-2 tap
  noise: decode-row and prefill taps, 1e-4 apart, agree on 14/15). Long
  context (700 real tokens of the 6b-0 code text, prefill 512 + 188, rows
  189..699 injected; CPU oracle `--act-quant gpu --kv-f16 --taps l_out`,
  `/server/staging/laguna/dflash-6d/oracle-long700`): per-row relative rms,
  median over rows, taps 1..47 = 8e-4, 1.0e-2, 3.6e-2, 7.7e-2, 9.0e-2, 0.12
  (decode-path GPU vs oracle), against 9.6e-4 ... 0.12 between the GPU's own
  prefill and decode roundings, so the oracle is no farther from the GPU than
  the GPU's two roundings are from each other (asserted within 1.25x); one
  layer early 0.25 to 0.93. The large rows are runs of a repeated token (id
  328, a separator line), not position 512. Drafts: GPU prefill taps vs
  oracle taps 13/15, decode-path taps vs oracle 13/15, prefill vs decode-path
  13/15; device-injected drafts equal host-uploaded drafts bit for bit. The
  first 16-row verify at 700 equals the same tokens decoded alone bit for bit.
  Four greedy steps accepted 1, 10, 0, 0 of 15 (one text; not an acceptance
  measurement, that is 6f).
- Drafter cost (`laguna_dflash_cost`, real drafter and head, 511 real
  captured context rows, block at 700; unprofiled wall time of `Propose`
  including the host embedding gather and the drafts' download; 30 reps after
  5 warm-up, two process loads, medians load 1 / load 2):

  | B | vector path | batched path |
  | --- | --- | --- |
  | 4 | 8.03 / 8.26 ms | 9.14 / 9.23 ms |
  | 8 | 14.56 / 14.90 ms | 9.17 / 9.43 ms |
  | 16 | 28.75 / 29.39 ms | 9.76 / 9.79 ms |

  Injecting a verify's 16 rows (vector) 1.72 / 1.75 ms, one row 0.48 / 0.49
  ms. This replaces 6b-0's 6.2 ms bandwidth estimate: about 9.5 ms per pass
  plus 1.7 ms of injection per step, so the 6b-0 tables' 6.2 ms column is
  optimistic by about 5 ms per step (still far below the 18.7 ms margin
  column). Verify on the real model took 267 to 370 ms per 16-row step at
  700 (capture and injection included, first step with scratch growth).
- Regression: the full `ctest -R '^laguna\.'` at the 6d code commits (built
  at 9f47cf20, same code as this commit): 27 passed, 2 skipped
  (`dflash_real_smoke` and `functional` are opt-in); log
  `final-ctest-9f47cf20.log`.
- GPU use: 10:35 to 10:54 EDT 2026-10-26 in short runs (15 s to 70 s, one
  process at a time under the lock; the CPU oracle 10:26 to 10:35 under the
  lock); a Scriberr job held the lock 10:38 to 10:40 and was waited out. Logs
  in `/server/staging/laguna/dflash-6d/` (`gpu-windows.log`).

Decisions recorded for 6e:

1. Snapshots (rule 5) are not implemented: a target `RestoreSnapshot`
   desynchronizes the drafter, and `DFlashSession` drops it on the next call
   that finds the ring out of step (`Prefill` re-arms it from the rows it can
   see). Snapshotting the drafter ring (13 MiB at the 544-row ring) is 6e.
2. Batched decode (rule 6): implemented as a refusal in `DecodeBatch`;
   the runner must route speculating sessions to `Forward`.
3. Drafter blocks on the batched path (measured above); injections on the
   vector path (1.7 ms per 16 rows, chunking-invariant).
4. A speculating session needs `max_chunk >= B`; rewinds deeper than
   `max_chunk` are refused even when the ring still holds the keys (the
   existing conservative check), so commit depth is bounded by the chunk.

Open after 6d:

1. The drafter is sensitive to tap rounding: 1e-2 tap differences change 2 to
   5 of 15 drafts (the later block rows). Whether prefill-path versus
   decode-path rounding of the injected rows costs acceptance is a 6f
   measurement.
2. The vector path is not warmed up on a batched-chunk executor; the first
   verify grows scratch (370 ms against 270 ms). Warm both paths in 6e.
3. `LagunaSpeculativeTarget` (`ISpeculativeTargetExecutor` adapter with a
   rewind commit), sampling, serve wiring: 6e.

## What phase 6e added (2026-10-26, GPU)

Code (commits cffc548b, 495b5c15, d9d3b9d1, 43a44330):

- `src/models/laguna/speculative.{hpp,cpp}`: `LagunaSpeculativeTarget`, the
  `ISpeculativeTargetExecutor` adapter over one `DFlashSession` (verify on the
  pinned vector path; target predictions are the greedy sampler's own argmax
  of each row, so ties break as plain decode breaks them; the commit of a
  partly rejected chunk is `DFlashSession::Commit`, a rewind of target and
  drafter, never a replay; a verify at another position than the session's or
  a commit that is not a prefix of the rows just verified is refused with
  nothing touched). `LagunaDraftBackend` (the drafter's block, batched path;
  a dropped or suspended drafter proposes nothing and the step is
  target-only). `SpeculativeDecoder` (the shared `SpeculativeVerifier`,
  batched verification, adaptive length off, block of 2..16 rows, default 4).
- A step selects the frontier token with a copy of the request's sampler,
  verifies it with up to B - 1 drafts and emits it plus the accepted drafts,
  all consumed by the target. The correction token is not emitted: its logit
  row becomes the session's frontier, so the next step (or a plain decode, a
  snapshot, a cancelled request, a reused prefix) selects it again exactly as
  plain decode would. The session state after a step is the state plain
  decode reaches after the same tokens; serve needed no pending-token or
  rollback machinery.
- Engine: `Model::AttachDrafter` (pairing, memory check against the device,
  upload), sessions with a `DFlashSession` and decoder, the vector path warmed
  for a verify of B rows at session creation (6d open item 2),
  `Session::DecodeSpeculative`, `SuspendDrafter`. `DFlashSession` gained
  Suspend/Resume (a sampled request clears the executor's taps so
  `EvaluateBatch`/`DecodeBatch` take it; resuming restarts the ring cold at
  the position), SaveDrafter/RestoreDrafter (`DrafterKv::Export/Import`, the
  whole 544-row ring, 13,369,344 bytes) and a prefill `following` hint (a
  prompt chunk injects only rows a block at the prompt's end can read).
- Snapshots carry the drafter ring (decision 1 of 6d: preferred over a cold
  drafter). A snapshot taken while the drafter is out of step or suspended
  has no ring; restoring it starts the ring empty at the restored position
  (cold, documented fallback), never with the rows the session held before.
- Batched decode (rule 6): a session whose executor has taps decodes alone;
  `EvaluateBatch` routes it to `Evaluate`. In serve, a request speculates
  only while it decodes alone (`DecodeStep`); when the scheduler has several
  requests ready (`DecodeBatch`) all of them, greedy included, are suspended
  and decode plainly in shared passes, and a greedy request resumes
  speculation with a cold ring afterwards. A first version that verified the
  speculating requests one after another next to a batched pass of the
  sampled ones failed the functional `batch` suite (a greedy and a sampled
  request in flight together must report physical width 2), which passes
  without speculation on the same binary.
- Sampling: greedy only. The lossless rejection-sampling verifier was not
  wired: it needs the drafter's sampled proposal distribution and a published
  correction token (the frontier design above re-selects it, which is exact
  only for deterministic requests). A request that samples at random
  (`uses_random_sampling()`: temperature > 0 and top_k != 1) decodes plainly
  on a suspended session; deterministic requests that are not plain argmax
  (grammars, penalties, reasoning budgets, top_k 1) speculate, with the
  sampler run through the draft prefix by the verifier. Shared change for
  that: `SpeculativeOptions::draft_deterministic_sampled` (opt-in, default
  off; Qwen backends unchanged).
- Serve: `gufo serve llm --speculative dflash2 --dflash-model
  <laguna-s-2.1-DFlash-Q4_K.gguf>`; `--draft-tokens N` sets B = N + 1 (1..15),
  else B = 4; `--draft-policy`/`--min-draft-tokens` do not apply. The runner
  advertises multi-token and batched multi-token decode only with a drafter;
  resource claims include the drafter and per-session drafter buffers. A
  drafter model served without `--speculative` is refused. `gufo prompt` got a
  Laguna path (with or without the drafter; `gufo chat` still refuses).
  Shared serve/prompt changes: `TextSpeculativeConfig::max_draft_tokens_supplied`
  (set from `--draft-tokens`), `PromptOptions::draft_tokens_supplied`, help text.

Tests (`ctest -R '^laguna\.(speculative|speculative_real|serving)$'`, under the
GPU lock):

- `laguna.speculative` (tiny models, nothing skipped): every run equals plain
  greedy decode on a second model of the same weights without a drafter
  (tokens, stop, position, frontier logits bit for bit). 105 runs over prompts
  of 9, 509..515, 600, 2048, 8192 and chat prompts with thinking on/off, with
  the random tiny drafter, an oracle (all accepted) and a corrupted oracle
  (partial), blocks 4/8/16; EOS 2 and EOT 24 made greedy winners by aliased
  head rows, drafted through, stop_at_eos on and off (16 runs); max_tokens 1,
  2, 3, 5, 9 inside a block of 8; a Laguna tool-call grammar (with and without
  reasoning), repeat penalty and top_k 1 through the draft prefix (oracle
  drafts accepted, so constrained requests really speculate); snapshot and
  restore with the ring (drafts after restore equal bit for bit) and without
  it (cold, not stale); negative checks (wrong anchor position, commit past /
  off / from elsewhere, random sampling refused, a target failure inside a
  step resets the session and re-arms the drafter, the drafter on a poisoned
  target is dropped and refuses to draft, a drafter failure mid-request
  continues target-only and exact); `EvaluateBatch` with drafting and
  suspended sessions; 2 and 8 concurrent sessions (speculating greedy and
  plain sampled mixed) equal their solo runs; chunk 16 with block 16.
- `laguna.serving` gained a speculative server: greedy chat equals the plain
  model (thinking on/off) and drafts; a required tool call equals the plain
  server and drafts; a sampled request does not speculate and equals the
  plain server; then the whole existing suite (chat, snapshots and prefix
  reuse, cancellation queued and mid-decode with retry, multi-session,
  batched greedy+sampled with a peer cancelled and a device error in a shared
  step, failure injection in prefill/decode/restored requests, tool prompts,
  context limit) on it.
- `laguna.speculative_real` (real target and drafter): exact on every run, no
  divergence anywhere. Per run (steps, accepted/drafted, smallest plain top-2
  logit gap decoded through):

  | Prompt | B | tokens | accepted/drafted | accepted/step | min gap |
  | --- | --- | --- | --- | --- | --- |
  | code 509..515 (7 prompts) | 4 | 40 each | 25..26 / 40..42 | 1.67..1.86 | 0.098 |
  | code 512 | 8 / 16 | 40 | 29/76, 29/142 | 2.64 | 0.216 |
  | code 2048 | 4 / 8 / 16 | 48 | 34/39, 40/45, 41/78 | 2.43, 5.00, 5.86 | 1.85 |
  | code 8192 | 4 | 48 | 33/40 | 2.20 | 0.084 |
  | code 32768 (`--long`) | 4 | 48 | 32/45 | 2.00 | 0.250 |
  | chat code, think | 4 / 8 | 96 | 67/87, 78/119 | 2.31, 4.33 | 0.640 |
  | chat chat, think | 4 / 8 | 96 | 47/141, 47/321 | 0.96 | 0.045 |
  | chat code, no think | 4 / 8 | 96 | 68/81, 80/106 | 2.43, 5.00 | 0.073 |
  | chat chat, no think | 4 / 8 | 96 | 53/126, 57/270 | 1.23, 1.46 | 0.011 |
  | tool grammar, no think / think | 4, 8 | 38 / 39, stop | 17/63..154 | 0.77..0.81 | 1.76 |
  | short answers ending on EOT (3 prompts) | 4/8/16 | 1..5, stop | | | 0.155 |
  | ignore_eos short answer | 8 | 24 | 5/112 | 0.26 | 0.155 |
  | max_tokens 1, 2, 3, 6, 11 | 8 | as asked | 0/0 .. 8/10 | | 2.0 |

  The smallest gap decoded through exactly was 0.0107 (chat, no think). The
  real drafter never proposed EOS or EOT in any watched block (tool calls,
  three short answers, ignore_eos): stop tokens inside a draft are covered by
  the tiny test only. Snapshot/restore at 700 + 20 tokens: drafts after
  restore bit-identical, 40 more tokens accepted 25/43 on both. 2 and 8
  sessions on threads (half speculating greedy, half sampled at temperature
  1.0 / top_k 20 on suspended sessions) each equal their solo runs.

Smoke A/B (not the 6f verdict): `laguna_speculative_real_test --ab --rounds 5
--tokens 160`, one process loads the model and runs plain, B = 4 and B = 8 on
reused sessions, the mode order reversed every other round and prompt, round
0 dropped, decode wall time only (prefill excluded), unprofiled; run twice
(two loads) back to back in one lock hold (13:14:51 to 13:37:01 EDT).
Greedy, 160 tokens, chat template; every speculative output equal to plain.
Median tok/s (load 1 / load 2), speedup and mean accepted drafts per step:

| prompt (prompt tokens) | plain | B = 4 | B = 8 |
| --- | --- | --- | --- |
| code, no think (71) | 27.17 / 27.18 | 35.48 / 35.51 (1.31x, 1.91) | 28.19 / 28.20 (1.04x, 2.90) |
| prose summary, no think (2237) | 22.24 / 22.22 | 24.27 / 24.29 (1.09x, 1.16) | 15.54 / 15.55 (0.70x, 1.25) |
| chat, think (67) | 27.23 / 27.23 | 29.25 / 29.30 (1.07x, 1.32) | 19.82 / 19.83 (0.73x, 1.67) |
| code explanation, no think (1580) | 22.90 / 22.88 | 24.04 / 24.07 (1.05x, 1.13) | 15.00 / 15.00 (0.66x, 1.19) |

B = 4 wins on all four (1.05x to 1.31x); B = 8 only breaks even on code and
loses 27 to 34 percent elsewhere, as the 6b-0 tables predict at short
context (a step costs about the 6b-0 verify time plus 9 to 11 ms of drafter
and injection: code B = 4 is 55 steps in 4.52 s, 82 ms per step against
77 + 9 estimated). The two loads agree within 0.2 percent. Data:
`/server/staging/laguna/dflash-6e/ab.log`, `ab-load{1,2}.csv`.

Functional suites with `--speculative dflash2 --dflash-model ...` (per-suite
impact.json configurations, `build/gpu-test-tools/gufo`, under the lock;
`/server/staging/laguna/dflash-6e/functional-spec-{core2,cache2,tools2}`):
core 15 of 16 (reasoning-budget fails), cache 7 of 7, tools 6 of 7 (`tools`
fails in auto-tools). The same binary without speculation fails the same two
the same way (identical text; `functional-plain-failing/`): the
reasoning-budget control request gets no reasoning from Laguna, and with
tool_choice auto it declines to call the tool; both also failed in the phase
5 runs. `batch` failed with the first serve policy (see Batched decode) and
passes now. Not run at the final commit: the full `ctest -R '^laguna\.'`
(GPU handed to Hindsight; see `dflash-6e/PAUSED.md`).

GPU use: 2026-10-26, one process at a time under the admission lock
(`dflash-6e/gpu-windows.log`): 12:27 to 12:39 (tiny tests, the real matrix
12:30 to 12:39), 12:59 to 13:14 (matrix with 32k), 13:14 to 13:37 (A/B),
13:38 to 13:41 (shared regressions), 13:41 to 14:31 (functional), 15:05 to
15:08 (plain rerun of the failing suites), 15:09 to 16:02 (functional on the
final binary). Waits for Flash-Next holding the lock: 12:41 to 12:59 and
14:31 to 15:05; builds waited 11:51 to 12:24 for host memory.

Open after 6e (6f):

1. Acceptance against the fork on the same prompts and the per-position
   acceptance curve; the drafter's acceptance on chat (about 1 accepted per
   step at B = 4) is far below code (2 to 2.4).
2. Block length: B = 4 is the right default at short context; an adaptive
   controller priced on the measured verify table (6b-0 c) and this step cost
   (verify + about 10 ms) could move to B = 8 for code and deep contexts and
   back to plain decode when acceptance stays near zero (chat think runs
   accepted 0.96 per step: a B = 4 step then costs more than it saves).
3. Sampled requests (the default preset, temperature 1.0 / top_k 20) do not
   speculate: the lossless sampled verifier needs the drafter's sampled
   proposal rows and a published correction token.
4. Drafter cost per step (9 to 11 ms with the injections) is a quarter of a
   decode token; a fused drafter block and a cheaper verify for 4 to 8 rows
   (6b-0: the vector MoE path is 30 to 100 ms over its bandwidth floor at 8
   to 16 rows) are the speed levers.
5. Concurrent requests do not speculate (serve suspends every session of a
   batched decode step). Speculating under concurrency needs a ragged
   multi-session verify pass with per-row tap capture (rule 6's later item);
   whether it beats plain batched decode on this MoE is unmeasured.
6. A suspended drafter resumes cold (an empty ring at the resume position),
   after a sampled request on the same session or after a period of
   concurrent decoding; acceptance right after a resume is unmeasured, and a
   short plain warm-up with capture may be worth it.
