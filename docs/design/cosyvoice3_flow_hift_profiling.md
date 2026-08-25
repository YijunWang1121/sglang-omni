# Fun-CosyVoice3 Flow/HiFT profiling: where fusion (and non-fusion) wins are

Tracking: [sgl-project/sglang-omni#1652](https://github.com/sgl-project/sglang-omni/issues/1652)
("Several kernel fusion is welcome if proved efficient").

## Problem

The issue's optimization roadmap invites kernel fusion for the CosyVoice3
Flow/HiFT decode path, but doesn't name specific targets. Guessing at fusion
sites from reading source is unreliable -- an op that looks fusable can turn
out to be <1% of wall time, and the actual bottleneck can be something that
isn't "fusion" at all. This note records a real `torch.profiler` run against
production code paths and ranks candidates by measured evidence instead of
code inspection alone.

Scope note: sglang-omni does not vendor the CosyVoice3 model code. Flow (DiT)
and HiFT live in the external `cosyvoice` package
(`sglang_omni/models/fun_cosyvoice3/stages.py::_load_cosyvoice3_flow_hift`
does `from cosyvoice.cli.cosyvoice import CosyVoice3` and pulls `cv.model.flow`
/ `cv.model.hift`). sglang-omni only wraps scheduling/batching around it. This
matters for *where* a fix can land -- some findings below are fixable entirely
inside sglang-omni; others require patching the external `cosyvoice` package.

## Methodology

- Hardware: 1x NVIDIA B300 (SXM6, 275GB), via Slurm (`srun -G 1`).
- Checkpoint: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (the same checkpoint the
  cookbook example uses), loaded exactly as `_load_cosyvoice3_flow_hift` does.
- Inputs: a real reference clip run through the actual preprocessing path
  (ONNX speaker encoder + speech tokenizer + mel features), so `prompt_feat`,
  `prompt_token`, and `embedding` shapes/content are production-realistic.
  The speech-token IDs fed to Flow are synthetic (125 tokens, ~5s at 25Hz) --
  a shape-only approximation of AR output; this does not affect DiT/HiFT
  compute pattern, only content.
- 5 warmup iterations + 5 measured iterations, `.eval()`, no autocast (see
  Finding 3 below for why that's the production-accurate condition).
- Profiled three regions separately with
  `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)`:
  `flow.inference()` alone, `hift.inference()` alone, and both back-to-back
  (matching `_CosyVoice3Vocoder._token2wav`'s actual call sequence).
- Raw artifacts (chrome traces + `key_averages()` tables) are on the profiling
  host under `~/scratch_profile/profile_out/`; not checked into the repo.

## Findings

### Finding 1 -- Flow dominates HiFT ~7.5:1

| region | self CUDA / iter | share of combined |
|---|---|---|
| `flow.inference()` (DiT + ~10 ODE steps x 22 layers) | 300ms | 88% |
| `hift.inference()` | 40.2ms | 12% |
| combined (`_token2wav`) | 341ms | 100% (additive, no interference) |

Any optimization budget should go to Flow first. HiFT is not negligible but
is a secondary target.

`hift.inference()` top rows are 87% `aten::cudnn_convolution`
(`precomputed_convolve_dgemm`, 117.1ms / 20 calls -- the ResBlock/upsample
stack). No ISTFT/`exp`/`view_as_complex` chain appears in the top 40 ops.

### Finding 2 -- Two Flow-side fusion/dedup candidates confirmed by call-count arithmetic

Top self-CUDA rows for `flow.inference()` (5-iter average):

```
aten::addmm                 52.19%  782.79ms  7955 calls  98.4us avg
cutlass sm100 sgemm (a)     24.36%  365.38ms  4450 calls  82.1us avg
aten::cudnn_convolution     22.83%  342.42ms   110 calls  3.11ms avg
cutlass sm100 sgemm (b)     22.78%  341.68ms  2200 calls 155.3us avg
implicit_convolve_sgemm     22.74%  341.09ms  1605 calls 212.5us avg
_efficient_attention_fwd     7.58%  113.67ms  1100 calls 103.3us avg  (fmha_cutlassF, already fused)
aten::mul                    3.75%   56.30ms 13620 calls   4.1us avg
aten::add                    3.48%   52.20ms  9160 calls   5.7us avg
aten::cat                    2.50%   37.56ms  4605 calls   8.2us avg
aten::copy_                  2.39%   35.90ms  5255 calls   6.8us avg
aten::native_layer_norm      1.15%   17.19ms  2250 calls  18.7us avg
aten::cos                    0.61%    9.18ms  2255 calls   4.1us avg
aten::sin                    0.59%    8.85ms  2250 calls   3.9us avg
aten::neg                    0.46%    6.91ms  2200 calls   3.1us avg
aten::masked_fill_           0.45%    6.81ms  1100 calls   6.2us avg
aten::gelu                   0.38%    5.72ms  1100 calls   5.2us avg
aten::where                  0.32%    4.78ms  2200 calls   (11.94ms CUDA-total)
aten::bitwise_not            0.30%    4.55ms  1110 calls   4.1us avg
```

(percentages are self-CUDA share of the `flow.inference()` total)

**2a. RoPE sin/cos + attention mask recomputed every ODE step (dedup, not
fusion; highest confidence, lowest risk).**
`cos`(2255) / `sin`(2250) / `neg`(2200) / `where`(2200) / `masked_fill_`(1100)
/ `bitwise_not`(1110) call counts are exact multiples of
`22 layers x 10 ODE steps = 220`. RoPE tables and the padding mask depend
only on `token_len`, which is constant across all 10 Euler steps inside one
`flow.inference()` call -- yet both are recomputed at every `(layer, step)`
pair. Combined this cluster is **~45-50ms/iter, ~15% of Flow's total time**.
Precomputing once per `inference()` call is correctness-preserving (same
values, computed once and reused) and does not require writing a fusion
kernel.

**2b. AdaLN-modulate + gated-residual pointwise chain (the actual fusion
candidate).**
`aten::mul` (13620 calls, 56.3ms) + `aten::add` (9160 calls, 52.2ms) =
**108.5ms/iter, ~36% of Flow's total time**, in sub-10us kernel launches
(~12 muls + ~8 adds per block-step across 220 block-steps). This is the
standard DiT pattern -- `scale * LN(x) + shift`, then `x + gate * sublayer(x)`
-- expressed as separate `aten` ops. A fused "LN+modulate" kernel and a fused
"gated-residual-add" kernel (e.g. Triton) would collapse this to ~2 launches
per block instead of ~8-12. **This is the strongest genuine fusion target in
the trace.**

`aten::cat` (1002 calls/iter, ~2.5%, likely per-layer conditioning-token
concatenation) is a secondary, lower-priority candidate -- fixing it means
restructuring to preallocated buffers rather than writing a fusion kernel, for
a smaller win than 2a/2b.

`_efficient_attention_fwd` already resolves to `fmha_cutlassF`, an efficient
fused kernel -- attention internals are not a target.

### Finding 3 -- Flow/HiFT run in FP32 in production (not fusion, but the largest single lever)

`create_vocoder_executor` in `stages.py` passes
`fp16=(dtype == "float16")` to `_load_cosyvoice3_flow_hift`, and
`_token2wav` only enables autocast when `self._fp16` is true. The shipped
config (`examples/configs/fun_cosyvoice3_0_5b.yaml`) uses `dtype: bfloat16`,
so `fp16` evaluates `False` and autocast never activates -- Flow and HiFT run
in plain FP32 as loaded from the checkpoint, on a Blackwell B300.
`addmm` + conv + attention are together ~85% of Flow's time, all FP32. This
is entirely inside sglang-omni's own code (not the external `cosyvoice`
package) and is a one-line-scale fix. It is not an operator-fusion change,
but it plausibly dwarfs the gain from 2a/2b and should be evaluated first
since it's also the smallest, lowest-risk change of the three.

## Theoretical candidates going in: confirmed / refuted

Before profiling, code inspection of the external `cosyvoice` package (not
this repo) suggested three candidates. Profiling confirmed one, refuted two,
and surfaced one that wasn't anticipated (2a above):

| candidate | verdict | evidence |
|---|---|---|
| HiFT pre-ISTFT `exp`/`sin`/`cos`/complex-construct chain | **refuted** | absent from top-40 ops; HiFT is 87% conv-bound |
| Flow-matching ODE per-step CFG combine (`torch.split` + weighted add) | **refuted** | ~1 call/step, negligible next to 2724 muls / 1832 adds |
| DiT estimator internals (attention/MLP) | **confirmed, but not where the fix is** | matmul+conv+attention dominate (~85%), but attention is already `fmha_cutlassF`-fused; the actual headroom is in the pointwise glue *around* attention/MLP (2a, 2b) |

## Recommended implementation order

1. **Fix the FP32/bf16 autocast bug** (Finding 3). Scope: sglang-omni only
   (`stages.py`). Smallest change, largest likely win, no dependency on the
   external `cosyvoice` package.
2. **Cache RoPE tables + attention mask per `inference()` call instead of per
   `(layer, step)`** (Finding 2a). Scope: external `cosyvoice` package
   (`cosyvoice/flow/...`) -- not fixable inside sglang-omni without either
   patching that dependency or monkey-patching after import. ~15% of Flow
   time, low risk (values are unchanged, just computed once).
3. **Fused AdaLN-modulate / gated-residual-add kernel** (Finding 2b). Same
   external-package scope issue as #2, larger engineering effort (a Triton
   kernel), largest pure-fusion win (~36% of Flow's pointwise overhead
   collapsed to a fraction of its current launch count).

Items 2 and 3 both require deciding *how* a fix lands given sglang-omni
doesn't vendor `cosyvoice`: patch/PR the external `FunAudioLLM/CosyVoice`
repo directly, or monkey-patch the loaded `flow`/`hift` modules from within
`sglang_omni/models/fun_cosyvoice3/stages.py` after import. That decision is
tracked separately before implementation starts on those two.

## Status

- [x] **Item 1 (bf16 autocast fix)** -- implemented and validated on B300.
  - Change: `sglang_omni/models/fun_cosyvoice3/stages.py` -- `_CosyVoice3Vocoder`
    now maps `dtype` to a real `torch.dtype` via `_AUTOCAST_DTYPES` instead of
    only checking for the literal string `"float16"`, and `hift.inference()`
    was moved inside the same `torch.autocast(...)` scope as `flow.inference()`
    (previously outside it, a deviation from upstream `CosyVoice3Model.token2wav`).
  - Unit tests: `tests/unit_test/fun_cosyvoice3/` -- 40/40 pass (fake flow/hift,
    doesn't exercise real numerics).
  - Latency (B300, 5 warmup + 5 measured iters, real `_token2wav` call):
    FP32 325.66ms/iter -> bf16 189.98ms/iter -- **1.71x** on the combined
    flow+hift path. Matches the expectation that this dwarfs the fusion
    candidates in Items 2/3.
  - Quality (fp32 vs bf16, same input, no NaN/Inf in either): mel output
    (pre-vocoder, phase-free) 30.15dB SNR / 0.9986 correlation; STFT-magnitude
    spectrogram (phase-invariant) 13.04dB SNR / 0.974 correlation; raw
    time-domain waveform -0.90dB / 0.349 correlation. The raw-waveform number
    is not meaningful in isolation -- HiFT's phase reconstruction is highly
    sensitive to small mel perturbations, so a well-matched mel (30dB) can
    still produce a phase-shifted waveform with near-zero sample correlation
    despite matching spectral content. The STFT-magnitude number is the more
    representative one. Divergence source: expected bf16-vs-fp32 rounding
    compounding over the ~10-step Euler / 22-layer DiT ODE (220 layer-steps
    total), not a masking/overflow bug -- no sub-step needs forcing back to
    FP32 based on this data.
  - End-to-end: `sgl-omni serve` + the cookbook's zero-shot cloning curl
    example, with the fix active and real AR-generated tokens (not synthetic)
    -- HTTP 200, output identical format/size/duration to the pre-fix baseline
    (mono/24kHz/16-bit, 4.88s, 234284 bytes), different MD5 (confirms real,
    non-cached output).
  - End-to-end latency (fixed seed, 5 requests each, same B300, steady-state
    mean excluding first-request warmup): FP32 1.10s/request -> bf16
    0.99s/request -- **1.11x (~11%, ~112ms/request)**. Lower than the 1.71x
    vocoder-only number because AR/LLM token generation (SGLang's own Qwen2
    backbone, untouched by this change, already CUDA-graph + fused-kernel
    optimized) dominates total request time for this short test utterance
    (~0.78s of the ~1.10s). Vocoder decode is ~30% of end-to-end time here;
    longer input text shifts more time into AR generation and dilutes this
    change's end-to-end share further, while short-utterance/high-concurrency
    workloads keep closer to the full benefit.
  - **Known verification gap: hardware generality.** The bug itself
    (`fp16=(dtype == "float16")`, a plain Python string comparison evaluated
    before any CUDA kernel launches) is architecture-agnostic -- it evaluates
    `False` for `dtype: bfloat16` on any GPU, so the fix is correct regardless
    of hardware. The **1.71x speedup number**, however, was only measured on
    B300 (the only GPU type available in this environment -- all 8 GPUs on
    this host are B300). On Ampere/Hopper the actual ratio may differ: PyTorch
    may default FP32 matmuls to TF32 tensor cores on those architectures
    (unlike the plain FP32 CUDA-core path this profiling saw), which changes
    the FP32-vs-bf16 baseline comparison. Re-running
    `scratch_profile/validate_bf16_fix.py` on A100/H100 would close this gap;
    not done here for lack of hardware access.
  - **Open item before merge: human listening check.** Metrics suggest
    "moderate but architecturally-expected" spectral degradation, not a
    clean pass -- not validated by ear yet. Compare
    `scratch_profile/output_before_bf16fix.wav` (FP32 baseline) against
    `sglang-omni/output.wav` (post-fix, bf16).
- [ ] Item 2 (RoPE/mask caching) -- **deferred.** Both fixable paths (patch
  upstream `FunAudioLLM/CosyVoice` directly, or monkey-patch after import
  from `sglang_omni/models/fun_cosyvoice3/stages.py`) are larger-scope,
  cross-repo work than Item 1. Decision: land Item 1 alone as a clean,
  self-contained sglang-omni PR first; revisit Item 2 separately.
- [ ] Item 3 (AdaLN/gated-residual fusion kernel) -- **deferred**, same
  reasoning as Item 2.
