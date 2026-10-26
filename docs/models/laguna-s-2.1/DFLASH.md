# Laguna-S-2.1 DFlash drafter in gufo (phase 6 spec)

Status 2026-10-26: spec plus the CPU-only config and weight index
(`src/models/laguna/dflash.{hpp,cpp}`, tests `laguna.dflash_contract` and
`laguna.dflash`). No GPU work was done or is specified as done: every GPU item
below is a proposal. Target phases 1 to 5 are complete (PLAN.md); this
document is the "own feature-layer and mask verification" that PLAN.md item 6
asks for.

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
| 6c | GPU operators one at a time against 6b: encoder (per-tap RMSNorm, `fc` Q4_K 18432 to 3072, norm); K/V injection (norm, K/V, K-norm, RoPE, ring write; positions 511/512/513, wrap); block attention (GQA 9, window plus causal block, row `r` sees `511 - r` context rows, ring capacity, shorter blocks equal prefixes of longer ones); gate softplus; SwiGLU with Q4_K and Q6_K; final norm, head and argmax; every projection both vector and batched path. Reuse the `qfn_mmq` raw kernels with one expert and zero ids, as the target's head does | GPU |
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
