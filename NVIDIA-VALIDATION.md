# NVIDIA PearlHash validation branch

Experimental source for RTX 3070 Ti (Ampere) and RTX 2070 (Turing). This branch combines PRs #3–#6 and the opt-in proof-construction experiments. NVIDIA behavior has NOT been validated on the author's A770. No private wallet or pool logs are included.

Build from this branch: both the native Node addon and SYCL library must be rebuilt together because the proof snapshot interface changed. Do not mix with published release binaries or use the Intel-only local build helper.

## Get the source

```sh
git clone --branch pearlhash-nvidia-validation --single-branch https://github.com/friskens/mo-miner.git
cd mo-miner
```

Follow DEVELOPMENT.md for prerequisites. Linux uses Docker buildx and the NVIDIA container runtime:

```sh
MOM_GPU_BACKEND=nvidia ./r.sh node mom.js algo_params
MOM_GPU_BACKEND=nvidia ./r.sh npm run test:github
MOM_GPU_BACKEND=nvidia MOM_GPU_TEST_VENDORS=nvidia ./r.sh npm run test:gpu-discrete
```

On Windows, use the repository's NVIDIA multicompiler build:

```powershell
powershell -File .github/workflows/scripts/build-windows-multicompiler.ps1 -Backend nvidia
powershell -File scripts/test-windows-current-multicompiler.ps1 -Backend nvidia -SkipBuild
```

These toolchain builds can be substantial. Review DEVELOPMENT.md first. No prebuilt NVIDIA binaries are supplied with this branch.

## Compare on each GPU separately

Use the same existing PRL pool configuration, unique worker names, dimensions, driver and power settings for both modes. Select the physical card using MOM_GPU_INDEX after checking algo_params; index0 may be exposed as gpu1. Prefer backend=native for NVIDIA PearlHash and verify startup output actually selects CUDA native. Do not force level_zero:gpu on NVIDIA.

Mode A: clear MOM_PEARLHASH_ASYNC_PROOF, MOM_PEARLHASH_REUSE_ROOT and MOM_PEARLHASH_GPU_LEAVES; set MOM_PEARLHASH_PROOF_TIMING=1.

Mode B: set MOM_PEARLHASH_ASYNC_PROOF=1 and MOM_PEARLHASH_REUSE_ROOT=1; keep MOM_PEARLHASH_GPU_LEAVES unset; keep proof timing enabled. Set environment variables BEFORE starting Node. Repeat A afterward if possible, with equal run durations (e.g.30min each) and no concurrent compilation.

GPU-leaf reuse is currently gated to Intel SYCL GPU execution. Setting MOM_PEARLHASH_GPU_LEAVES=1 on NVIDIA should retain CPU proof regeneration; it does NOT enable a CUDA leaf-copy optimization. Test this separately as a fallback check; do not claim CUDA leaf-reuse performance.

Both cards typically have 8GiB VRAM. Check allocation success and keep matrix settings identical between A/B; do not assume the A770 16GiB configuration fits. Use documented automatic NVIDIA tuning first. If reducing dimensions is necessary, repeat both phases at the reduced dimensions. For LuckyPool, enable pearlhash_rank_penalty=true, use base target format, and retain V3 handling. Never substitute our historical wallet.

## Return results

Record exact commit, GPU, OS, driver, compiler/runtime, selected backend, M/N/K/rank, power limit, phase start/end, proof build_ms, local reported rate, share difficulty, submitted/accepted/rejected counts/reasons and pool worker rate. Keep logs private: verbose plain_proof records and wallet addresses must be removed before posting to GitHub. Stop on invalid-proof rejections. Distinguish reported MAC/s from accepted work.

Exercise job rotation and graceful shutdown while proofs are pending. Current experimental behavior waits for a pending proof at shutdown and explicitly discards that result. Construction-failure injection and broader CUDA lifecycle coverage are outstanding. This is a validation branch, not a production recommendation.
