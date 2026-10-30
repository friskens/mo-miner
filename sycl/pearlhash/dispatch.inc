enum class PearlHashSearchBackend {
  sycl,
  hip_jit,
  cuda_jit
};

// Prepare this seed's commitments/matrices, select a compatible search, and report the backend used.
static PearlHashSearchBackend
attempt(sycl::queue& q, Buffers& b, uint32_t seed, int m, int n, int k, int rank,
        const std::string& backend = "auto", PearlHashHipSearch* hip = nullptr,
        PearlHashCudaSearch* cuda = nullptr, sycl::queue* hip_prep_queue = nullptr,
        bool validate_native = false, bool cuda_tensor = false, bool cuda_f16 = false,
        bool pair_dist8 = false, bool fixed_b_reuse = false,
        const sycl::event* roots_override = nullptr) {
  // Only the already-admitted native CUDA pair route may skip B preparation.  Every other
  // backend keeps the original single-seed producer contract, including all AMD/Intel paths.
  b.fixed_b_reuse = pair_dist8 && fixed_b_reuse;
  // A second producer queue belongs only to HIP; CUDA/portable preparation stays on q.
  if (!hip || (hip_prep_queue && !mom_is_hip(q.get_device())))
    hip_prep_queue = nullptr;
  const bool split_hip_noise = hip_prep_queue != nullptr;
  static thread_local unsigned stats_attempts = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && stats_attempts++ < 3;
  auto stats_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* stage) {
    if (!stats)
      return;
    q.wait_and_throw();
    if (hip_prep_queue)
      hip_prep_queue->wait_and_throw();
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - stats_last).count();
    std::fprintf(stderr, "PEARLHASH_SYCL_STATS %s=%.3fms\n", stage, ms);
    stats_last = now;
  };
  sycl::event roots_ready;
#if defined(_WIN32)
  const bool windows_cpu = q.get_device().is_cpu();
  if (windows_cpu) {
    q.wait_and_throw();
    prepare_roots_cpu(b, seed, m, n, k);
  } else {
#endif
    roots_ready = roots_override ? *roots_override : k_roots(q, b, seed, m, n, k);
#if defined(_WIN32)
  }
#endif
  mark("roots");
  PearlHashEventPair noise_ready;
#if defined(_WIN32)
  if (windows_cpu)
    prepare_noise_cpu(b, m, n, k, rank);
  else {
#endif
    sycl::queue& noise_b_queue = hip_prep_queue ? *hip_prep_queue : q;
    noise_ready = k_noise(q, noise_b_queue, b, m, n, k, rank,
                          roots_ready); // sparse low-rank A/B noise
#if defined(_WIN32)
  }
#endif
  mark("noise");

  // Pick the search kernel by the running device's backend BEFORE laying out A'/B': compute_ab needs to
  // know whether to write the portable row-major A' / column-major B' (search()) or the ESIMD tile-major
  // layout. CUDA -> search_cuda (mma.sync, nvptx). Any OpenCL device (AMD GPU, the CPU device, and
  // Intel OpenCL) -> the portable dp4a search() -- it is the single matrix-hardware-free path, so
  // the same image runs on AMD and the CPU and Intel OpenCL behaves like another OpenCL GPU. (Set
  // MOM_PEARLHASH_ESIMD to opt an Intel-OpenCL *GPU* back into the faster ESIMD path.) Level-Zero / default
  // Intel -> ESIMD. search_cuda/search_esimd/search() are each compiled only in the builds that can run
  // them, so the references below are guarded to match.
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::cuda;
#else
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
#endif
#else
  const bool is_cuda = false;
#endif
#if !defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  (void)cuda_f16;
#endif
  const bool dbg = validate_native || std::getenv("MOM_PEARLHASH_CHK") != nullptr;
  const bool tuned_sycl = backend == "auto" || backend == "sycl-native" || backend == "native";
#if defined(MOM_SYCL_HAS_CUDA)
  const bool cuda_sycl =
      is_cuda && tuned_sycl && cuda_tensor && cuda_sycl_search_supports_shape(m, n);
#else
  (void)cuda_tensor;
  [[maybe_unused]] const bool cuda_sycl = false;
#endif
  // Decide before compute_ab chooses its layout. Optional ESIMD queries fail closed; unsupported
  // devices and explicit overrides retain the ordinary matrix layout and search.
  bool use_portable = !cuda_sycl;
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  const unsigned esimd_width = [&] {
    if (!tuned_sycl || is_cuda)
      return 0u;
    try {
      const sycl::device device = q.get_device();
      const bool supported_backend = sycl_is_level_zero_gpu(device) ||
          (mom_is_opencl(device) && std::getenv("MOM_PEARLHASH_ESIMD") != nullptr);
      // Integrated Intel GPUs can lack the LSC prefetch used by this search even with matrix support.
      const bool discrete_gpu = device.is_gpu() && !is_integrated_gpu(device);
      const bool intel_matrix = discrete_gpu && device.has(sycl::aspect::ext_intel_matrix);
      // Query the DPAS layout before preparation; unknown widths retain the portable reference.
      const unsigned width =
          supported_backend && intel_matrix ? esimd_search_width(device) : 0u;
      return pearlhash_esimd_route(tuned_sycl, supported_backend, discrete_gpu, intel_matrix, width)
          ? width : 0u;
    } catch (...) {
      return 0u;
    }
  }();
  use_portable = !cuda_sycl && esimd_width == 0u;
#endif
  // The SYCL tensor kernel has the same SM80 instruction floor as CUTE; failed capability
  // detection must retain the ordinary matrix layout/search even for an explicit native request.
  if (is_cuda && !cuda_sycl)
    use_portable = true;

  // Only JIT unavailability permits fallback. Preparation, launch and completion faults must
  // escape: retrying another kernel on the same potentially invalid queue is not recovery.
#if defined(MOM_SYCL_HAS_HIP)
  if ((backend == "auto" || backend == "native") && hip && hip->ensure(q, b, m, n, k, rank)) {
    sycl::queue& prep = hip_prep_queue ? *hip_prep_queue : q;
    PearlHashEventPair ready = compute_ab_amd_wmma(q, prep, b, seed, m, n, k, rank,
                                                    noise_ready.a, noise_ready.b);
    // Reuse the shared low-CPU wait, including its query-fault retirement and OpenCL policy.
    sycl_wait_and_throw(ready.a, q.get_device());
    sycl_wait_and_throw(ready.b, prep.get_device());
    mark("ab");
    hip->launch(b, seed, m, n, k, rank, dbg);
    return PearlHashSearchBackend::hip_jit;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  const bool cuda_ready = (backend == "auto" || backend == "native") && is_cuda && cuda &&
      cuda->ensure(q, b, m, n, k, rank);
  if (cuda_ready) {
    // Rank-128 native CUTE consumes the producer-packed A/B representation. Rank-256
    // deliberately retains the ordinary row/column layout used by the staged payload.
    if (cuda->compact) {
      // Ap/Bp are unused by the raw solver; reuse their first K words for
      // canonical packed input and sorted output instead of another allocation.
      q.parallel_for(sycl::range<1>(static_cast<size_t>(k)),[=](sycl::id<1> id) {
        const size_t i=id[0];
        reinterpret_cast<uint32_t*>(b.Ap)[i]=static_cast<uint32_t>(b.EARp1[i]) |
            (static_cast<uint32_t>(b.EARp2[i])<<8) |
            (static_cast<uint32_t>(b.EBLq1[i])<<16) |
            (static_cast<uint32_t>(b.EBLq2[i])<<24);
      });
    } else if (rank == 128)
      compute_ab<false, true>(q, b, seed, m, n, k, rank, /*portable=*/true);
    else
      compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
    q.wait_and_throw();
    mark("ab");
    cuda->launch(b, seed, m, n, k, dbg);
    return PearlHashSearchBackend::cuda_jit;
  }
  if (pair_dist8)
    throw std::string("PearlHash pair-dist8 CUDA route became unavailable after admission");
#endif
  // If the optional native route declined after split submission, rejoin the prep queue before
  // the shared path reads the noise tables. Successful native dispatches already returned above.
  if (split_hip_noise) {
    sycl_wait_and_throw(noise_ready.a, q.get_device());
    sycl_wait_and_throw(noise_ready.b, hip_prep_queue->get_device());
  }
#if defined(PEARLHASH_ESIMD)
  // Paired DG2 proofs/rates are qualified on oneAPI Level Zero; combined AOT/OpenCL stay ordinary.
  const bool use_dpasw = pearlhash_dpasw_route(
      !use_portable && q.get_backend() == sycl::backend::ext_oneapi_level_zero, esimd_width,
      esimd_width == 8u && dpasw_esimd_device(q.get_device()), rank);
#endif
  if (q.get_device().is_cpu()) {
    q.wait_and_throw();
    compute_ab_cpu(b, seed, m, n, k, rank);
  } else {
    // CUDA uses the same column-major B' layout in both generators. Tile an oversized B' to
    // avoid DPC++'s signed 1D range limit without changing the selected CUDA search.
    const bool tiled_cuda_b =
        cuda_sycl && checked_elements(n, k, "N*K") >
                         static_cast<size_t>(std::numeric_limits<int>::max());
#if defined(PEARLHASH_ESIMD)
    if (use_dpasw)
      compute_ab<true>(q, b, seed, m, n, k, rank, /*portable=*/false);
    else
#endif
      compute_ab(q, b, seed, m, n, k, rank, /*portable=*/use_portable || tiled_cuda_b);
  }
  mark("ab");
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  if (is_cuda && tuned_sycl && cuda_f16 && rank > 0 && rank <= 1024 && rank % 16 == 0) {
    search_t<2, 4, true, true>(q, b, seed, m, n, k, rank, dbg);
    mark("search_f16_matrix");
    return PearlHashSearchBackend::sycl;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  if (cuda_sycl) {
    search_cuda(q, b, seed, m, n, k, rank, dbg);
    mark("search_cuda");
    return PearlHashSearchBackend::sycl;
  }
#endif
#ifndef MOM_PEARLHASH_ESIMD_TU // search() is not compiled into the ESIMD-only TU (which only exports search_esimd)
  if (use_portable) {
    if (q.get_device().is_cpu())
      search_cpu(q, b, seed, m, n, k, rank, dbg);
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    else if (is_cuda && tuned_sycl)
      search_t<pearlhash_portable_er, pearlhash_portable_ec, true>(q, b, seed, m, n, k, rank, dbg);
#endif
    else
      search(q, b, seed, m, n, k, rank, dbg);
    mark("search_portable");
    return PearlHashSearchBackend::sycl;
  } // portable dp4a int8 GEMM (AMD / CPU / Intel-OpenCL)
#endif
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  // B580: about 45.4 TH/s with the canonical M=N=131072, K=2048, rank=128 profile.
#if defined(PEARLHASH_ESIMD)
  if (use_dpasw) {
    search_esimd_dpasw(q, b, seed, m, n, k, rank, dbg);
    mark("search_esimd_dpasw");
  } else
#endif
  {
    search_esimd(q, b, seed, m, n, k, rank, dbg, esimd_width);
    mark("search_esimd");
  }
#else
  search(q, b, seed, m, n, k, rank, dbg); // Defensive fallback if future backend selection skips use_portable.
  mark("search_portable");
#endif
  return PearlHashSearchBackend::sycl;
}
