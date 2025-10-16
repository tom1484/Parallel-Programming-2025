Here’s a complete design doc you can drop into your repo as docs/parallel-sift-mpi-omp.md.

Parallel SIFT: Hybrid MPI + OpenMP Design

Target platform
	•	1–3 nodes, up to 36 MPI ranks total
	•	6 OpenMP threads per rank (≈ 6 cores/rank)
	•	Shared-nothing across ranks, shared memory within a rank

Algorithmic baseline
We follow Lowe’s SIFT detector–descriptor (DoG extrema, sub-pixel refinement, 36-bin orientation, 4×4×8 descriptor with trilinear interpolation, clamp + renorm). For theory/geometry, see Lowe and the VLFeat/OpenCV descriptions.  ￼

⸻

1) High-level parallelization model

1.1 Domain decomposition (MPI)

Use a fixed 2-D Cartesian process grid (Px × Py), created once via MPI_Cart_create. Each rank owns a rectangular tile of the image for every octave/scale; per-octave tile bounds are scaled from the base image size. This spatial decomposition suits stencil-like stages (Gaussian, gradients, DoG, extrema) and minimizes cross-rank dependencies.  ￼
	•	For octave o: image dims (W_o,H_o) \approx (W_0/2^o, H_0/2^o).
Rank (p_x,p_y) owns:
x\in[\lfloor p_xW_o/P_x\rfloor,\;\lfloor (p_x{+}1)W_o/P_x\rfloor),\quad
y\in[\lfloor p_yH_o/P_y\rfloor,\;\lfloor (p_y{+}1)H_o/P_y\rfloor).
	•	Same neighbors across all octaves; only the per-octave buffers change. Use the Cartesian communicator for neighbor relations and halo exchanges.  ￼

Refinement for small octaves. When tiles shrink below a practical compute size (e.g., interior < ~64×64 after halos), create a smaller sub-grid for that octave with MPI_Cart_sub, or simply mark some ranks as idle for that octave.  ￼

1.2 Hybrid threading (OpenMP inside each rank)
	•	Threading model: MPI_THREAD_FUNNELED; only the OpenMP master thread calls MPI. This avoids the overheads and complexity of MPI_THREAD_MULTIPLE.  ￼
	•	Affinity: set OMP_PROC_BIND=close (or spread for wide cores) and configure OMP_PLACES=cores (or platform equivalent) to prevent thread migration and NUMA penalties.  ￼
	•	Inner parallelism: use #pragma omp parallel for over pixels for image-wide loops and keypoint-parallel loops with schedule(dynamic, chunk) for irregular work (orientation/descriptor). Avoid false sharing via per-thread temporaries and padding if needed.  ￼

1.3 Halo exchanges & overlap

Stencil phases require ghost/halo regions on tile borders. Use either nonblocking p2p (MPI_Isend/Irecv) to overlap interior computation with communication, or neighborhood collectives on the Cartesian communicator (e.g., MPI_Neighbor_alltoallw) for convenience.  ￼

⸻

2) Stage-by-stage parallel plan

You’ve already implemented separable Gaussian (row + col passes). The design below focuses on distribution & synchronization and on our chosen on-the-fly gradients path.

2.1 I/O & distribution
	•	Option A (simple): rank 0 reads the image and scatters tiles with MPI_Scatterv.
	•	Option B (scalable): use MPI-IO: each rank sets a subarray file view (MPI_Type_create_subarray, MPI_File_set_view) and reads its tile directly. This is ideal if you store pyramids/tiles in contiguous row-major files.  ￼

2.2 Gaussian pyramid (your Phase 1)
	•	MPI: identical 2-D tiling for each octave/scale.
	•	Halos: for level with blur σ, exchange halo width r=\lceil 3\sigma\rceil (practical Gaussian truncation).
	•	Overlap: post nonblocking exchanges; compute interior rows/cols first, then process border once halos arrive.  ￼

2.3 DoG pyramid (Phase 2)
	•	Local compute only. DoG = difference of adjacent Gaussian levels, done in-place over each tile; no comm besides what Gaussian already needed. (DoG approximates scale-normalized LoG and is standard for SIFT.)  ￼

2.4 Keypoint detection & refinement (Phase 3)
	•	Extrema scan: each rank scans its local DoG volume (x, y, scale). The 3×3×3 test uses already-present scale neighbors and spatial halos; no extra cross-rank traffic.  ￼
	•	Interior ownership rule: keep candidates whose integer (i, j) lie one pixel inside the tile and one scale away from octave ends; neighbors drop duplicates.
	•	Refinement (Taylor fit, contrast, edge filters): pure per-candidate compute within the rank. (Hessian ratio edge test is local.)  ￼
	•	OpenMP: parallelize pixel scans with collapse(2) and use straightforward loops for refinement; schedule(static) or guided both work here.

2.5 On-the-fly gradients for orientation & descriptor (Phases 5–6)

Rather than precomputing a full gradient pyramid, compute gradients at sampling time from the Gaussian image at the keypoint’s scale. This preserves SIFT’s definition (gradients on the scale-appropriate blur) and typically slashes time/memory if many pixels never contribute to descriptors.  ￼

2.5.1 Orientation assignment

For each keypoint (x,y,\sigma), on its owning rank:
	1.	Select blurred image L(\cdot,\cdot;\sigma) (nearest level).
	2.	Collect gradients via central differences near the keypoint inside a circular window (σ-scaled), weight magnitudes by a Gaussian, and build a 36-bin orientation histogram. Smooth histogram and select peaks ≥ 0.8×max; assign one orientation per peak (with sub-bin refinement).  ￼

2.5.2 Descriptor (4×4×8 = 128-D)

Work on the same L(\cdot,\cdot;\sigma), in a rotated frame aligned with the reference orientation:
	1.	For each sample, compute (L_x,L_y) by central difference; get magnitude m and relative angle \theta’.
	2.	Trilinear interpolation: distribute m to the two nearest orientation bins (out of 8) and four neighboring spatial cells (2×2) of the 4×4 grid.
	3.	Concatenate 16 histograms, L2-normalize → clamp (≈0.2) → renormalize (then quantize to uint8 if you keep that interface).  ￼

Ownership near borders. Before computing a descriptor, confirm that the rotated patch fits within the tile. If not, send the keypoint record (position, scale, orientation) to the neighboring rank that contains the full support and compute there (tiny messages; no image shipping). VLFeat/OpenCV document this geometry; your runtime already has \lambda_{\text{ori}}, \lambda_{\text{desc}}, N_{\text{HIST}}, etc., to determine the support window.  ￼

OpenMP pattern (irregular, best speedups):
	•	#pragma omp parallel for schedule(dynamic,64) over keypoints.
	•	Keep per-thread private histograms (36-bin ori, 128-D desc), then write out one descriptor vector per keypoint (prevents false sharing).  ￼

2.6 Gather results
	•	Use MPI_Gatherv (or Allgatherv) to collect (x,y,octave,scale,descriptor[128]) records to rank 0 for output. Collectives generally outperform ad-hoc p2p for this pattern.
	•	If you output tiled visualizations, use MPI-IO + subarray file views to write a single image per level directly from distributed tiles.  ￼

⸻

3) Communication mechanics

3.1 Cartesian topologies & neighbor collectives
	•	Create a 2-D topology once: MPI_Cart_create(comm_world, 2, dims, periods, reorder, &cart).
	•	For stencil halos, either:
	•	Nonblocking p2p: MPI_Isend/Irecv for the 4 (or 8) neighbors; overlap compute with comm.
	•	Neighborhood collectives: MPI_Neighbor_alltoall[w] on cart—less boilerplate and maps neatly to fixed neighbor sets.  ￼

3.2 Overlap pattern (pseudo)
	•	Post all Irecvs for halos
	•	Pack & post all Isends
	•	Compute interior region
	•	MPI_Waitall
	•	Compute border region
This classic stencil pipeline is well-studied for image/PDE codes.  ￼

⸻

4) Load balancing
	•	Convolution/DoG/extrema: uniform work per pixel ⇒ fixed tiles are fine.
	•	Keypoints: highly clustered. Use two levers:
	1.	Over-decomposition: split the image into more tiles (e.g., 12×12 logical) and assign 2–4 tiles/rank; spreads hotspots statistically.
	2.	Within a rank: OpenMP schedule(dynamic) for keypoint loops; chunks of 32–128 are effective.
	•	If you opted for on-the-fly gradients, migrating only keypoint records (few bytes) to a neighbor is cheap when a tile becomes overloaded. (Avoid shipping image patches.)

⸻

5) Threading details & pitfalls
	•	Thread level: initialize with MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided) and assert provided >= MPI_THREAD_FUNNELED. Query with MPI_Query_thread if needed.  ￼
	•	Affinity: OMP_PROC_BIND=close or spread; set OMP_PLACES=cores (or platform-specific). Pinning prevents cache/NUMA thrash in hybrid runs.  ￼
	•	False sharing: never let multiple threads update adjacent elements of a shared histogram/array; keep private accumulators per thread and pad if you must share.  ￼

⸻

6) Memory layout & vectorization
	•	Store Gaussian levels row-major; keep tiles contiguous per level.
	•	For on-the-fly gradients, access directly from the Gaussian image; central differences touch only immediate neighbors, which stream well.
	•	Favor AoS → SoA where it helps inner loops (e.g., keep gx, gy transient in registers during sampling rather than writing a gradient image).
	•	Compile with -O3 -ffast-math (as safe for your pipeline) and enable auto-vectorization; test with and without -fno-math-errno depending on your environment.

⸻

7) Pseudocode skeletons

7.1 Program layout

// MPI init (FUNNELED), build Px×Py Cartesian 'cart'
omp_set_num_threads(6);

for (octave = 0; octave < N_OCT; ++octave) {
  // compute local tile bounds (xo: x1, yo: y1) for this octave
  // GAUSSIAN levels (k = 0..N_SPO+2):
  for (k = 0; k < N_LEVELS; ++k) {
    post_halo_exchange(cart, buf[k], halo_r[k]);          // Irecv/Isend
    convolve_rows_interior(buf[k]);                       // OpenMP parallel for
    finish_halo_and_borders(cart, buf[k], halo_r[k]);     // Waitall + borders
    convolve_cols_all(buf[k]);                            // OpenMP parallel for
  }

  // DoG levels
  for (k = 0; k < N_LEVELS-1; ++k) dog[k] = buf[k+1] - buf[k]; // OpenMP

  // Extrema + refinement (local)
  keypoints = find_and_refine_candidates(dog, interior_rule);

  // Keypoint routing near borders (descriptor support check)
  route_border_keypoints(cart, keypoints); // send records only

  // Orientation + descriptor (on-the-fly gradients)
  #pragma omp parallel for schedule(dynamic,64)
  for each kp in owned_keypoints {
    orientations = assign_orientation_on_the_fly(buf[kp.level], kp);
    for θ in orientations:
      desc = compute_descriptor_on_the_fly(buf[kp.level], kp, θ);
      write_out(kp, θ, desc);
    }
  }
}

// gather results (Gatherv) and finalize

7.2 On-the-fly descriptor kernel (sketch)

Descriptor compute_descriptor_on_the_fly(const Image& L, const Keypoint& kp, float theta0) {
  Hist8x4x4 H; // thread-local 4×4×8, zeroed
  // iterate samples inside rotated square support
  for (each sample s in neighborhood_of(kp, lambda_desc)) {
    // rotate (x,y) -> (x',y') around kp by -theta0, normalize by sigma
    // skip if (x',y') outside 4×4 grid bounds
    float Lx = L(x+1,y) - L(x-1,y);
    float Ly = L(x,y+1) - L(x,y-1);
    float mag = hypot(Lx, Ly);
    float ang = atan2f(Ly, Lx) - theta0; // wrap to [0,2π)
    float w = gaussian_weight(x,y,kp, lambda_desc);
    trilinear_accumulate(H, x',y', ang, w*mag);
  }
  return normalize_clamp_normalize(H); // L2 -> clamp(0.2) -> L2
}

Design matches SIFT’s canonical descriptor build with trilinear interpolation in (x,y,\theta).  ￼

⸻

8) Testing & validation
	1.	Functional parity: compare single-rank, single-thread outputs vs hybrid runs:

	•	#keypoints, descriptor norms, match counts on standard pairs.

	2.	Border correctness: synthesize keypoints near tile edges; verify ownership/reroute logic yields identical descriptors to serial baseline.
	3.	Determinism: with dynamic schedules, descriptor order may differ—store with stable IDs if needed.

⸻

9) Performance plan
	•	Primary targets:
	•	Gaussian + (previously) gradient computation dominated your time; with on-the-fly gradients, expect a large drop in the former 36.6% “gradient pyramid” bucket.
	•	Descriptor step (24%) should scale well with keypoint-parallel OpenMP.
	•	Measure & tune:
	•	Vary Px×Py near your node topology; keep ranks per socket balanced.
	•	Try schedule(dynamic, 32..128) for keypoints.
	•	Compare nonblocking p2p vs neighbor collectives for halos.
	•	Confirm binding with environment/launcher options (OMP_PROC_BIND, MPI launcher --bind-to, etc.).  ￼

⸻

10) Why this design
	•	MPI tiling aligns with image stencils and SIFT’s local computations; no scale-direction communication is required.  ￼
	•	On-the-fly gradients compute exactly what SIFT uses (gradients of the scale-blurred image) while avoiding a heavyweight gradient pyramid.  ￼
	•	Hybrid FUNNELED + pinned OpenMP is the most portable, low-overhead model for your scale.  ￼
	•	Overlap hides halo latency; neighbor collectives simplify halo code on a Cartesian grid.  ￼

⸻

Appendix A — Key references
	•	Lowe, SIFT original paper (IJCV 2004). (algorithmic details, orientation & descriptor)  ￼
	•	VLFeat SIFT overview & API (descriptor trilinear interpolation, geometry).  ￼
	•	OpenCV SIFT tutorial (36-bin orientation; σ window ≈ 1.5×scale).  ￼
	•	MPI neighbor collectives and Cartesian topologies (halo exchange on grids).  ￼
	•	Nonblocking communication to overlap compute and comm.  ￼
	•	Hybrid MPI+OpenMP best practices, thread levels & binding.  ￼

If you’d like, I can adapt this into a minimal C++ skeleton (MPI + OpenMP) wired to your sift.cpp entry points so you can slot it into your build.