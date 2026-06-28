# qr_v2 — submissions & handoff

Two independently-developed solutions that **converged on the same algorithm**, plus the plan to push
further toward the leaderboard top.

| File | Author | Geomean (B200) | Notes |
|---|---|---|---|
| `submission.py` | codex | **0.00594** (~22× over `torch.geqrf`=0.131) | **CANONICAL** — the evaluated file. Converged 2-level blocked Householder: fused small-n kernel (n≤~200, whole matrix in shared, 1 launch) + V/C-cached warp inner-apply + cuBLAS WY trailing; IB=8 `#pragma unroll` + `__launch_bounds__` micro-specialization. |
| `submission_claude_best.py` | claude | 0.006486 | Same algorithm; behind codex only on the two micro-specializations (qr_small thread count + inner-apply unroll/launch_bounds). |
| `HANDOFF_TO_3.md` | claude | — | **The plan.** Lever-map (everything tried + *why* it's dead: tensor cores ×3, the CholeskyQR/lift family via κ², multi-warp panel, two fusion megakernels), the work-queue-megakernel build spec, and a profile-first go/no-go gate. |

**Reality check (verified 2026-06 via web).** The famous **~1097 µs / sub-2000 µs "records" were
reward hacks** on the *original* QR leaderboard (early-stop for known benchmark shapes + timing
exploits); **qr_v2 was built specifically to block them**. The legit qr_v2 top is **~0.002**
(`badelsteinlelbach` = Bryce Adelstein Lelbach, NVIDIA CUDA-libraries lead, hand-written CUDA;
#2–3 are Triton ~0.0026). So the real gap from 0.00594 is **~3×**, against expert hand-tuned CUDA of
the *same* blocked-Householder family — not a secret algorithm. Details + the path in `HANDOFF_TO_3.md`.
