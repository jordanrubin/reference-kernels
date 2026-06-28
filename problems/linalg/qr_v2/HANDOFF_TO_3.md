# qr_v2 → top-of-leaderboard Handoff: build spec for the work-queue megakernel

**Audience:** an engineer pushing batched compact-Householder QR from the current **0.00594**
(≈22× over `torch.geqrf` = 0.131) toward the legit qr_v2 leaders.

> **⚠️ TARGET CORRECTION (verified via web, 2026-06).** The famous **#1≈1097µs / sub-2000µs** times
> were on the *original* QR leaderboard and were **reward hacks** (early-stop QR for known benchmark
> shapes + timing exploits, "Volkswagen cheat"). **qr_v2 was built specifically to block them.** The
> **legit qr_v2 top is ~2000µs (0.002)** — handle `badelsteinlelbach` = **Bryce Adelstein Lelbach**
> (NVIDIA, leads CUDA C++ Core Libraries), **hand-written CUDA in a plain submission.py**; #2–3 are
> Triton at ~2600µs. So the real gap from our 0.00594 is **~3×**, against expert hand-tuned CUDA of
> the *same* blocked-Householder family (no public code/writeup; not a secret algorithm). Community
> notes: **hand-written CUDA (nvcc) beats CuteDSL/Triton here**; one floated *"cublas masks + skip
> zero tiles"* (geometry exploit for structured/mixed cases).
>
> **⚠️ PROFILER IS AVAILABLE.** `popcorn --profile-brev` runs the submission on a GPU Mode Brev B200
> and **saves an Nsight Compute trace locally** (`--benchmark-index N` to profile one shape). The
> "blind, no profiler" caveat below is therefore SOFTENED — the profile-first plan and the go/no-go
> gate are executable. Verify it runs in your environment first.

This doc says (1) what the task is, (2) the current best algorithm, (3) every lever already tried and
why it's dead/tapped — *so you don't re-tread it*, (4) the one remaining mechanism with 3.5× in it and
a concrete build spec, (5) the profile-first plan and a hard go/no-go gate.

---

## 1. The task & scoring (what you're optimizing)

- Batched square QR: input `A` is `b×n×n` FP32 CUDA; return `(H, τ)` in `torch.geqrf` compact form
  (R in upper triangle, Householder vectors below, τ the `b×n` coeffs).
- Checker: materializes `Q = householder_product(H,τ)`, `R = triu(H)`; validates in **FP64**.
  - Factor residual gate: `rtol = 20·n·eps32` on `‖R − Qᵀ A‖₁`.
  - Orthogonality gate: `rtol = 100·n·eps32` on `‖QᵀQ − I‖₁`.
  - **Hard FP32-accuracy gate.** Low-bit (FP16/FP8/NVFP4) allowed only internally; returned factors
    must pass at FP32 accuracy. Note the gate **loosens with n** (20·n·eps32: n=512→1.2e-3, n=1024→2.4e-3).
- **Ranking = geometric mean over 12 benchmark cases, equal weight.** Conditioning robustness is
  *ranked, not just gated* — the benchmark includes fully ill-conditioned homogeneous batches.

### The 12 benchmark shapes (geomean — every case counts equally)
| # | b | n | case | notes |
|---|---|---|------|-------|
| 1 | 20 | 32 | dense | small-n, batch-limited |
| 2 | 40 | 176 | dense | small-n |
| 3 | 40 | 352 | dense | medium, overhead-sensitive |
| 4 | 640 | 512 | dense | **highest-weight band (n=512 ×4)** |
| 5 | 60 | 1024 | dense | |
| 6 | 8 | 2048 | dense | **panel catastrophically starved (8 warps)** |
| 7 | 2 | 4096 | dense | **2 warps; torch.geqrf fallback today** |
| 8 | 640 | 512 | mixed | heterogeneous conditioning |
| 9 | 60 | 1024 | mixed | |
| 10 | 640 | 512 | rankdef | ill-conditioned |
| 11 | 640 | 512 | clustered | ill-conditioned |
| 12 | 60 | 1024 | nearrank | ill-conditioned |

Implication: n=512 (×4) and n=1024 (×3) dominate; small-n (×3) is overhead-bound and batch-limited;
the two large/small-batch cases (6,7) are occupancy-catastrophic on the panel.

---

## 2. Current best algorithm (the converged approach — start here)

Two agents converged on this independently; codex's `submission.py` (=0.00594) is the fastest
instance. Files: `claude/linalg/qr_v2/submission_small.py` (mine, 0.006486) and
`codex/linalg/qr_v2/submission.py` (0.00594; ahead only via micro-specialization).

- Host transposes to column-major (`Acm = data.transpose(-2,-1).contiguous()`), so kernels index
  `As[r + c*n] = A[r,c]` with coalesced columns. Output `H` is a stride-only transpose view (no copy).
- **n ≤ 200 → `qr_small`**: fused, whole matrix resident in shared, one block/matrix, **one launch**,
  no cuBLAS. Sequential geqr2 in shared. (This captured the small-n overhead; +5.4% on geomean.)
- **200 < n ≤ 2048 → `qr2`**, 2-level blocked Householder, **NB=64, IB=8**:
  - `panel_warp`: **1 warp per matrix** geqr2 of the inner IB-panel (warp-shuffle reductions, no
    shared, no atomics). Beat `cublasSgeqrfBatched`.
  - `inner_apply_vcache8`: applies the IB=8 reflectors to the rest of the outer block. **V and the
    trailing columns C both cached in shared**, `nW=16` columns/CTA, IB=8 `#pragma unroll`,
    `__launch_bounds__(512,1)`. (Reflector+column caching = the bandwidth lever, +~10% total.)
  - `wy_update`: cuBLAS WY for the big outer trailing (`G=VᵀC`, T-solve, `C −= V·W2`), **native FP32**.
- **n > 2048 → `torch.geqrf`** fallback (cuSOLVER wins at very large n / tiny batch).
- `--use_fast_math`. Closed-form block-T `T = inv(diag(1/τ) + striu(VᵀV))`.

Reflector convention (matches geqrf, passes the gate): for column with `α=A[i,i]`,
`xn2 = ‖A[i:,i]‖² − α²`; if `xn2≤0`: `τ=0, β=α`; else `β = −sign(α)·‖A[i:,i]‖`, `τ=(β−α)/β`,
`v = A[i+1:,i]/(α−β)`, `v[i]=1`, then `A[i,i]=β`.

---

## 3. Lever map — already tried, DO NOT re-tread

| Lever | Result | Why |
|---|---|---|
| Reflector cache (V in shared) | **+6%** ✓ | inner-apply was L2-bandwidth-bound on V reloads |
| Column cache (C in shared, "cacheC") | **+4%** ✓ | C re-read per reflector; cache → read/write once |
| Reuse width nW=16 cols/CTA | ✓ | nW=32 worse (occupancy: 1 block/SM); nW=8 worse (more V reloads) |
| Fused small-n (`qr_small`, n≤200) | **+5.4%** ✓ | small-n cases are launch/overhead-bound, not compute-bound |
| **Tensor cores via cuBLAS FP32 emulation (BF16x9, EAGER)** | **DEAD** | uniform NB64=0; NB128/NB256 worse; per-shape n≥2048=tied, n≥1024=worse. **Intensity roofline**: tensor cores need ~241 flops/byte (1926 TF ÷ 8 TB/s); QR supplies ~IB..NB = 8–128. Crossing 241 needs NB≈256, which explodes the BLAS-2 inner cost. |
| Low-precision TF32/BF16 trailing (single pass) | DEAD | fails the gate at all n (TF32 err/gate=1.8 @ n=1024; BF16 worse). Refinement = 3x-split = the emulation above = 0 net. |
| CholeskyQR / Jacobi / symmetric-dilation / Gram-tower / higher-order lift | DEAD | all route through `AᵀA` ⇒ κ² ⇒ fail the ill-conditioned benchmark cases (rankdef/clustered/nearrank are there *by design*). Reconstruction to (H,τ) is rankdef-fragile. The dilation gives SVD not QR. |
| Multi-warp / cooperative panel | DEAD | `wp2` register-accum = 0.0233 (5× worse). Panel is **occupancy-bound**; cross-warp reduction overhead > occupancy gain at the dominant n. Codex committed to `__launch_bounds__(32)` (1 warp). |
| Persistent full-fusion megakernel (`qr_persist`, custom trailing) | **0.00722 LOST** | one block/matrix ⇒ trailing under-parallelized (1 block vs cuBLAS's ~56 tiles/matrix). |
| Fused outer-panel in shared (`fused_panel`, 16→1 launch/outer block) | **0.00651 TIED** | launch savings exactly cancelled by occupancy loss (128KB shared ⇒ 1–2 blocks/SM). |
| Wide NB (no emu): 96/128/256 | worse | panel BLAS-2 ∝ NB |
| Adaptive nW; structural fast-lanes (upper/diag); n=352 persistent block | no gain / worse | upper/diag not in scored set; persistent loses to cuBLAS |

**Net:** the algorithm is at its ceiling. Caching is tapped, small-n is fused, tensor cores are
structurally unusable, and every fusion so far trades occupancy for launches at ~zero net.

---

## 4. The #3 mechanism + build spec

### Why 3.5× has exactly one home
We are **above** the FP32-CUDA-core roofline already (gap is overhead/occupancy/dispatch, not flops or
precision). The dominant remaining inefficiency: the **panel phase runs the chip at ~7% occupancy**
— `panel_warp` is 1 warp/matrix ⇒ `b` warps total (b=640 → 4.3 warps/SM out of 64; b=8/2 →
catastrophic) — and it is **barrier-separated** from the full-occupancy trailing. The 3.5× = filling
that idle 93% by **pipelining**: overlap matrix A's low-occupancy panel with matrix B's high-occupancy
trailing.

### The foreclosure that makes it hard (and why it needs *you*, not blind iteration)
- The submission scanner **bans the lowercase CUDA-queue word** ("str·eam") ⇒ **no CUDA graphs, no
  multi-stream overlap**. So pipelining must be **intra-kernel** — a single persistent megakernel.
- A persistent kernel **cannot call cuBLAS** ⇒ the trailing must be a **custom in-kernel GEMM**.
- Both fusion builds confirmed custom trailing < cuBLAS today.

So #3 reduces to: **a custom parallel FP32 GEMM competitive with cuBLAS, inside a deadlock-free
work-queue megakernel.** That needs a profiler and local iteration — not 5-min blind leaderboard runs.

### Profile FIRST (confirm the hypothesis before building)
1. Nsight Compute/Systems on the current best at **n=512, b=640** and **n=2048, b=8**:
   - Measure the **fraction of wall-time the SMs are <20% occupied** (expect: large, during panels).
   - Confirm `wy_update`'s GEMMs are *not* the bottleneck (emulation gave 0 — corroborate).
   - Confirm `panel_warp` + `inner_apply` dominate and are latency/occupancy-bound.
   - If the idle-chip hypothesis is wrong, **re-target** before writing a megakernel.

### HARD GO/NO-GO GATE (do this second, it's cheap and decisive)
Write a standalone custom **K=64** FP32 GEMM for the trailing update `C(m×ncols) −= V(m×64)·W(64×ncols)`
and benchmark vs cuBLAS on the real trailing shapes (m,ncols ∈ {512,1024,2048}).
- K=64 is *small* — cuBLAS's general GEMM may be suboptimal here; that's the opening.
- Standard recipe: 128×64 or 64×64 thread-block tiles, 8×8 register micro-tiles, double-buffered
  shared loads of V and W, FP32 accumulate.
- **Gate: custom GEMM must be ≥ cuBLAS on these shapes.** If you cannot beat (or match) cuBLAS for
  K=64, **the megakernel path is dead — stop and bank 0.0059.** Everything below depends on passing this.

### The megakernel (only if the gate passes)
Persistent work-queue kernel, launched once at a resident grid size
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor × #SMs`):
- **Task types:** `PANEL(m, ko)` (few-block, warp-level geqr2 of the IB-panel) and
  `TRAIL_TILE(m, ko, t)` (one output tile of the WY trailing, using the custom GEMM above).
- **Dependencies:** `PANEL(m,ko)` → all `TRAIL_TILE(m,ko,·)` → `PANEL(m,ko+1)`. Track with a
  per-(m,ko) `panel_done` flag and a `tiles_remaining` atomic counter in global memory.
- **Scheduling:** each block loops `idx = atomicAdd(&head,1)`; map idx→task; if its dependency isn't
  met, **re-enqueue / grab the next ready task** (never spin-wait on an unmet dep → no deadlock). The
  mix of low-occupancy panels and high-occupancy trailing tiles from *different matrices* keeps every
  SM full — this is the entire point.
- **Keep the working parts:** reuse `panel_warp` for PANEL, V/C-cached logic for the within-block
  apply, and the closed-form block-T. Only the *outer trailing* becomes the custom GEMM.

### Milestones & validation
1. Profile confirms idle-chip hypothesis. ✅/re-target.
2. Custom K=64 GEMM ≥ cuBLAS (standalone). ✅/STOP.
3. Megakernel skeleton (queue + deps + correctness on n=256,512; verify against `torch.geqrf`).
4. Integrate GEMM; measure occupancy-timeline gain (target: chip rarely <50% occupied).
5. Per-shape tuning. Biggest pipelining wins: **cases 6,7** (n=2048 b=8, n=4096 b=2 — most starved)
   and the **n=512 band** (highest weight). Small-n stays on `qr_small`.

### Risk register
- **GEMM-vs-cuBLAS (highest):** gated at milestone 2 on purpose.
- **Deadlock:** never block on an unmet dependency; always fall through to another ready task.
- **Megakernel occupancy:** the GEMM's register pressure + panel code may cap blocks/SM — watch with
  the profiler; consider separate panel/trailing block specializations.
- **Ill-conditioned correctness:** validate on cases 10–12 (rankdef/clustered/nearrank) every step.
- **Account is shared with another agent** (codex uses `submission.py`); use distinct filenames.
- **No benchmark gaming** (input-keyed result caching, well-conditioned-only fast paths) — the spec
  forbids it and the benchmark is built to catch it.

---

## 5. If the gate fails (realistic fallback)
The ceiling is **0.0059 (~22×)**. Partial, lower-risk wins that do *not* need a custom GEMM:
- Cooperative-groups `grid.sync()` fused inner-loop (launch reduction with full multi-CTA parallelism
  preserved) — `fused_panel` suggests occupancy cancels it, but `grid.sync` + multi-CTA has a
  different occupancy profile and is worth one profiled try (~1.2–1.3× *if* occupancy holds).
- Route n=2048 to `torch.geqrf` too (test: cuSOLVER may already beat our custom path there).
- Fuse the WY-setup kernels (`extract_V`+`build_M_I`+`set_ptrs`) — prior runs showed ~3% from launch
  reduction alone.

**Bottom line:** #3 is a real, well-scoped kernel project, but its make-or-break is one question —
*can a custom K=64 FP32 GEMM beat cuBLAS on a B200?* Answer that first; everything else follows.
