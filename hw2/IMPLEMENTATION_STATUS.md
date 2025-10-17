# SIFT Parallelization Implementation Status

## Overview
This document tracks the implementation of hybrid MPI+OpenMP parallelization for the SIFT algorithm, following the design outlined in `PARALLEL_SIFT_GUIDE.md`.

**Note:** OpenMP is currently disabled in the implementation (MPI-only mode) to simplify correctness debugging of halo exchanges and boundary handling. The profiler remains enabled and will report MPI rank information; any OpenMP metrics will be absent while OMP is disabled. OpenMP pragmas will be re-enabled after MPI correctness is fully validated.

**Date:** October 18, 2025  
**Status:** Phase 3 (Keypoint Detection Parallelization) - Implementation Complete and Tested

---

## ✅ Completed Components

### 1. MPI Infrastructure (`mpi_utils.hpp/cpp`)

**Implemented structures:**
- `CartesianGrid`: 2D Cartesian MPI process grid
  - Automatic topology creation with `MPI_Cart_create`
  - Neighbor rank discovery (TOP, BOTTOM, LEFT, RIGHT)
  - Optimal grid dimension calculation (e.g., 36 ranks → 6×6 grid)

- `TileInfo`: Tile bounds and metadata
  - Computes per-rank tile boundaries for each octave
  - Handles octave-specific scaling (images shrink by 2^octave)
  - Detects too-small tiles for adaptive parallelism

- `HaloBuffers`: Storage for halo exchange
  - Separate buffers for each direction (top/bottom/left/right)
  - Dynamic allocation based on kernel size

**Implemented functions:**
- `pack_boundaries()`: Extract boundary rows/columns for sending
- `unpack_boundaries()`: Integrate received halos
- `exchange_halos()`: Nonblocking halo exchange with `MPI_Isend/Irecv`
- `wait_halos()`: Synchronization barrier
- `scatter_image_tiles()`: Distribute image from rank 0 to all ranks
- `gather_image_tiles()`: Collect tiles at rank 0

### 2. Parallel Gaussian Blur (`image.cpp`)

**Function:** `gaussian_blur_parallel()`

**Key features:**
- **Separable 2D convolution** (vertical → horizontal passes)
- **Halo exchange with computation overlap:**
  1. Post nonblocking receives for vertical pass (top/bottom halos)
  2. Pack and send boundary rows
  3. Compute interior rows (overlap with communication)
  4. Wait for halos to arrive
  5. Compute border rows using received halos
  6. Repeat process for horizontal pass (left/right halos)
- **OpenMP:** Currently disabled for debugging. Code runs single-threaded per MPI rank for deterministic behavior. Will re-enable after validation.
- **Boundary handling:**
  - Interior processes use received halos from neighbors
  - Boundary processes clamp to edge (no neighbor data available)
  - Special handling for small tiles (< halo_width) with clamping
- **Odd-sized image support:**
  - Correctly handles cases where image dimensions don't divide evenly among ranks
  - Computes neighbor tile dimensions deterministically to ensure MPI message size matching

**Complexity:**
- Communication: O(halo_width × perimeter) per pass
- Computation: O(tile_width × tile_height × kernel_size)
- Overlap efficiency: ~70-90% depending on tile size and kernel radius

### 3. Parallel Gaussian Pyramid (`sift.cpp`)

**Function:** `generate_gaussian_pyramid_parallel()`

**Implementation:**
- **Distributed scale-space construction:**
  1. **Initial upscaling:** Use `resize_parallel()` to distribute 2x bilinear upscaling across ranks
     - Broadcasts source image to all ranks
     - Each rank computes its tile of the upscaled image
  2. Apply initial blur with `gaussian_blur_parallel()` (distributed)
  3. For each octave:
     - Update tile bounds for current octave (images shrink by 2^octave)
     - Check tile sizes for adaptive processing mode decision
     - Apply successive blurs with increasing sigma (k = 2^(1/scales_per_octave))
     - **Downsample between octaves** via gather-downsample-scatter pattern

- **Adaptive octave processing:**
  - **Normal mode (tiles ≥ 20 pixels):** Distributed processing across all ranks
  - **Small-tile mode (tiles < 20 pixels):** Sequential processing on rank 0 only
  - Uses `MPI_Allreduce(MIN)` to collectively decide which mode to use
  - Ensures all octaves are processed (no skipping), preserving pyramid structure
  - Non-rank-0 processes create empty placeholder images for small octaves

- **Downsampling strategy:**
  - **Current implementation (gather-downsample-scatter):**
    1. Gather full image from all ranks to rank 0
    2. Rank 0 uses `resize_parallel()` with nearest-neighbor interpolation
    3. All ranks receive their tiles of the downsampled image
  - **Rationale:** Ensures consistent pixel selection at tile boundaries
  - **Cost:** 2 collective operations (gather + broadcast) per octave transition
  - **Benefit:** Eliminates accumulated boundary errors across octaves

**Key parameters:**
- `min_tile_size = 20`: Threshold for switching to rank-0-only mode
- Upscaling factor: 2x (SIFT standard)
- Downsampling factor: 2x per octave
- Sigma progression: k = 2^(1/scales_per_octave)

### 4. Main Program Integration (`hw2.cpp`)

**MPI initialization:**
- `MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)`
  - Requests `MPI_THREAD_FUNNELED` support (only master thread calls MPI)
  - Validates that required thread level is provided
  - Aborts if `MPI_THREAD_FUNNELED` is not available
- Gets rank and size from `MPI_COMM_WORLD`
- Rank 0 prints configuration: number of ranks, grid dimensions

**Cartesian grid setup:**
- Computes optimal Px×Py dimensions via `CartesianGrid::get_optimal_dims()`
  - Creates most square-like grid possible
  - Prefers more rows than columns (images typically wider)
- Initializes Cartesian communicator with `grid.init(px, py)`
- All ranks participate in topology creation

**Profiler integration:**
- Initializes with `Profiler::getInstance().initializeMPI(rank, size)`
- Tracks MPI communication separately (via `PROFILE_MPI()` macro)
- Uses `gatherAndReport()` to collect timing data from all ranks
- Rank 0 prints aggregated profiling results

**Current execution flow:**
1. Rank 0 loads and converts image to grayscale
2. Broadcast image dimensions to all ranks
3. Create `TileInfo` for base octave on all ranks
4. Call `find_keypoints_and_descriptors_parallel()`:
   - Generates Gaussian pyramid in parallel (distributed)
   - Gathers full pyramid to rank 0
   - Continues with serial DoG, keypoint, and descriptor on rank 0
5. Rank 0 writes results (text output + visualized keypoints)
6. Gather and print profiling data
7. Clean up: `grid.finalize()` before `MPI_Finalize()`

**OpenMP status:**
- Currently disabled in computation loops (MPI-only debugging mode)
- OpenMP library still linked for future re-enablement
- No `omp_set_num_threads()` calls active

### 5. Parallel SIFT Pipeline (`sift.cpp`)

**Function:** `find_keypoints_and_descriptors_parallel()`

**Implementation phases:**

1. **Preprocessing (rank 0 only):**
   - Convert RGB to grayscale if needed
   - Non-rank-0 processes create empty placeholder

2. **Parallel Gaussian pyramid generation (all ranks):**
   - Calls `generate_gaussian_pyramid_parallel()`
   - Each rank computes its local tiles for distributed octaves
   - Rank 0 handles small octaves sequentially

3. **Pyramid gathering (collective operation):**
   - For each octave and scale:
     - Check if octave was processed in distributed mode (tile size check)
     - **Distributed octaves:** Use `gather_image_tiles()` to collect full image at rank 0
     - **Rank-0-only octaves:** Rank 0 already has full image, no gathering needed
   - All ranks must participate in collectives to avoid deadlock
   - Uses `MPI_Allreduce` to synchronize mode detection

4. **Serial SIFT pipeline (rank 0 only):**
   - Generate DoG pyramid from full Gaussian pyramid
   - Find and refine keypoints (extrema detection, Taylor fit, filtering)
   - Generate gradient pyramid (central differences)
   - Assign orientations (36-bin histogram, peak detection)
   - Compute descriptors (4×4×8 histogram, trilinear interpolation)

5. **Output:**
   - Returns keypoint vector (empty on non-rank-0 processes)
   - Rank 0 writes to text file and visualization image

**Design rationale:**
- Validates parallel Gaussian blur correctness in complete SIFT context
- Establishes infrastructure for future parallelization of remaining stages
- Maintains compatibility with serial baseline for validation

**Debugging features (commented out):**
- Can save pyramid images to disk for octave-by-octave comparison
- Useful for diagnosing boundary artifacts and numerical issues

### 6. Parallel Image Resize (`image.cpp`)

**Function:** `resize_parallel()`

**Purpose:** Distribute image resizing (upscaling/downsampling) across MPI ranks

**Implementation:**
- **Input:** Source image (valid on rank 0), target dimensions, tile info for output
- **Strategy:** Broadcast-compute pattern
  1. Rank 0 has source image, broadcasts dimensions and channels
  2. Broadcast full source image to all ranks
  3. Each rank computes its tile of the output image independently
  4. Each rank maps its output coordinates to source coordinates
  5. Performs interpolation (bilinear or nearest-neighbor)
- **Output:** Each rank holds its tile of the resized image

**Usage in SIFT:**
- Initial 2x upscaling at pyramid start
- Downsampling between octaves (via rank 0, then distributed)

**Trade-offs:**
- **Pros:** Simple, correct, deterministic results
- **Cons:** Full source image broadcast (O(source_size) communication)
- **Justification:** Resize is not the bottleneck; simplicity > optimization here

### 7. Build System (`CMakeLists.txt`)

**Configured:**
- **Compiler:** `mpicxx` (MPI C++ wrapper)
- **Standard:** C++17 with no extensions
- **MPI:** `find_package(MPI REQUIRED)`, linked with `MPI::MPI_CXX`
- **OpenMP:** Configured for macOS Clang (Xpreprocessor flags), linked with `OpenMP::OpenMP_CXX`
- **Source files:**
  - Main: `hw2.cpp`, `sift.cpp`, `image.cpp`, `profiler.cpp`, `mpi_utils.cpp`
  - Sequential: `sequential/image.cpp`, `sequential/sift.cpp`
  - Validation: `validate.cpp` (Release only, `EXCLUDE_FROM_ALL`)
- **Optimization:**
  - Release: `-O3` flag
  - Debug: `-g -Wall` flags, `DEBUG` preprocessor definition
- **Output:** `compile_commands.json` for LSP/IDE support

### 8. Parallel DoG Pyramid (`sift.cpp`) - Phase 2

**Function:** `generate_dog_pyramid_parallel()`

**Implementation:**
- **Embarrassingly parallel computation** - no MPI communication needed
- Each rank processes its local tiles from the Gaussian pyramid
- Pixel-wise subtraction: DoG[j] = Gaussian[j+1] - Gaussian[j]
- Operates on distributed data (tiles), no gathering required until later

**Algorithm:**
```cpp
For each octave:
    For each scale j (1 to imgs_per_octave):
        Copy Gaussian[j] to diff
        Subtract Gaussian[j-1] pixel-by-pixel
        Store diff as DoG[j-1]
```

**Key characteristics:**
- **Zero communication overhead:** Pure local computation
- **Memory efficient:** Operates on already-distributed tiles
- **Scales linearly:** Near-perfect parallel efficiency
- **OpenMP ready:** Comments indicate future parallelization points

**Performance (Testcase 00, 4 ranks):**
- Total time: ~71.79 ms across 4 ranks
- Per-rank average: ~17.95 ms
- Overhead: 0% (no communication)
- Expected speedup: Linear with rank count

**Integration:**
- Called immediately after `generate_gaussian_pyramid_parallel()`
- Both Gaussian and DoG pyramids gathered together to rank 0
- Eliminates serial DoG computation on rank 0

### 9. Parallel Keypoint Detection (`sift.cpp`) ⭐ NEW - Phase 3

**Function:** `find_keypoints_parallel()`

**Implementation:**
- **Distributed extrema detection** across MPI ranks
- Each rank scans its local DoG tiles independently
- **Interior ownership rule:** keeps keypoints 1 pixel inside tile boundary
- Eliminates duplicates at borders and ensures all neighbors available
- Uses existing helper functions: `point_is_extremum()`, `refine_or_discard_keypoint()`

**Algorithm:**
```cpp
For each octave:
    Compute tile boundaries for this rank
    Define interior region (1 pixel inside boundary)
    For each scale (skip first/last):
        For each pixel in interior region:
            Quick contrast check (0.8 × threshold)
            If promising:
                3×3×3 extremum test
                If extremum: refine and validate
                If valid: add to local keypoints
```

**Key characteristics:**
- **Zero communication during detection:** Pure local computation
- **Interior ownership rule:** prevents duplicate keypoints at tile borders
- **Handles rank-0-only octaves:** gracefully skips empty tiles
- **Coordinate transformation:** local tile coords → global image coords

**Keypoint Gathering:**
- `MPI_Gather` collects keypoint counts from all ranks
- `MPI_Gatherv` transfers all keypoint data to rank 0
- Transfer using `MPI_BYTE` (Keypoint struct is POD-compatible)
- Efficient variable-length gather

**Performance (Testcase 00, 4 ranks):**
- Detection time: ~181.39 ms total (~45.35 ms per rank average)
- Gathering time: ~227.03 ms
  - MPI_Gather (counts): 144.86 ms
  - MPI_Gatherv (data): 82.04 ms
- Load imbalance: 11.83 ms (min) to 101.98 ms (max) per rank
- Found 2542 keypoints (expected ~2536, within 0.2%)

**Integration:**
- Called after `generate_dog_pyramid_parallel()`
- Replaces serial `find_keypoints()` on rank 0
- Gathered keypoints used for orientation/descriptor phases

**Exposed Helper Functions** (in `include/sequential/sift.hpp`):
- `bool point_is_extremum(const vector<Image>& octave, int scale, int x, int y)`
- `bool refine_or_discard_keypoint(Keypoint& kp, const vector<Image>& octave, ...)`

---

## 🚧 In Progress / To Do

### Phase 1: Testing & Validation

**Implementation Status:** ✅ Complete

**Fixed Issues:**

**✅ FIXED: Octave 2+ Boundary Artifacts**
- **Issue:** Numerical differences exploded at octave 2+ when using multiple ranks
- **Root cause:** Independent local downsampling caused tile boundary misalignment
  - Each rank's nearest-neighbor interpolation rounded coordinates differently at boundaries
  - Errors accumulated across octaves (octave 0→1→2...)
- **Solution:** Gather-downsample-scatter pattern
  1. Gather full image to rank 0 between octaves
  2. Use `resize_parallel()` for consistent downsampling
  3. Each rank gets its tile of the downsampled image
- **Result:** Eliminates boundary discontinuities, ensures octave-to-octave consistency

**✅ FIXED: Halo Exchange for Odd-sized Images**
- **Issue:** MPI message size mismatches at high octaves (e.g., 15×19 image at octave 6)
- **Root cause:** Odd-sized images split unevenly across ranks
  - Example: 15 pixels / 2 ranks → rank 0: 7 pixels, rank 1: 8 pixels
  - Halo exchange code assumed uniform tile sizes
  - Send size (based on sender's tile) ≠ Recv size (based on receiver's tile)
- **Solution:** Compute neighbor dimensions deterministically
  - Each rank calculates neighbor tile sizes using same formula
  - Use neighbor's dimensions for recv buffer sizing
  - Conservative buffer allocation with safety margins
- **Result:** Correct halo exchanges for all image sizes and rank counts

**✅ FIXED: Small Octave Handling (Adaptive Processing)**
- **Issue:** At high octaves (7+), tiles become smaller than halo width
  - Example: Octave 7, tile=4×3, halo_width=6 → impossible to exchange halos
- **Root cause:** Fixed halo width grows relative to shrinking tile dimensions
- **Solution:** Adaptive processing with automatic mode switching
  1. Before each octave, check if tiles ≥ 20 pixels (collective decision)
  2. If too small, switch to rank-0-only sequential mode
  3. Rank 0 processes octave with sequential `gaussian_blur()`
  4. Non-rank-0 processes create empty placeholder images
  5. All octaves still processed (preserves pyramid structure)
- **Benefits:**
  - Automatic adaptation (no manual tuning)
  - Maintains correctness for all image sizes
  - Main speedup from large octaves (where it matters)
- **Result:** Handles any image size and process count gracefully

**Testing Instructions:**

Use the provided scripts in `scripts/` directory for easy testing:

1. **Debug single testcase:**
   ```bash
   # Usage: ./scripts/run_debug <testcase_id> [PE] [NP] [THREADS]
   # Example: Run testcase 01 with 4 MPI ranks, 6 threads each
   ./scripts/run_debug 01 2 4 6
   ```
   - Builds Debug configuration
   - Runs with specified MPI/OpenMP configuration
   - Automatically validates against golden reference
   - Reports timing and validation result

2. **Release single testcase:**
   ```bash
   # Usage: ./scripts/run_release <testcase_id> [PE] [NP] [THREADS]
   # Example: Run testcase 04 (large image) for performance testing
   ./scripts/run_release 04 2 4 6
   ```
   - Builds Release configuration (-O3 optimization)
   - Reports timing and validation result

3. **Benchmark all testcases:**
   ```bash
   # Usage: ./scripts/run_benchmark [N] [PE] [NP] [THREADS]
   # Example: Benchmark first 8 testcases with 4 ranks
   ./scripts/run_benchmark 8 2 4 6
   ```
   - Runs N testcases (default: 10)
   - Reports individual and average timing
   - Validates each output (PASSED/FAILED)
   - Logs saved to `results/*.log` and `results/*.val.log`

**Script Parameters:**
- `testcase_id`: Test number (01-08)
- `PE`: Processing elements per socket (default: 2)
- `NP`: Number of MPI ranks (default: 4)
- `THREADS`: OpenMP threads per rank (default: 6, currently unused)

**Testing Strategy:**

1. **Correctness first (Debug):**
   ```bash
   # Test various rank counts with small testcase
   ./scripts/run_debug 01 2 1 6  # 1 rank (should match serial)
   ./scripts/run_debug 01 2 2 6  # 2 ranks
   ./scripts/run_debug 01 2 4 6  # 4 ranks
   ./scripts/run_debug 01 2 9 6  # 9 ranks
   ```

2. **Edge cases:**
   ```bash
   ./scripts/run_debug 01 2 4 6  # Small image (152×185)
   ./scripts/run_debug 04 2 4 6  # Large image (964×1248)
   ```

3. **Performance testing (Release):**
   ```bash
   # Benchmark with various rank counts
   ./scripts/run_benchmark 8 2 1 6   # Baseline: 1 rank
   ./scripts/run_benchmark 8 2 4 6   # 4 ranks
   ./scripts/run_benchmark 8 2 16 6  # 16 ranks
   ```

**Validation Metrics:**
- **Descriptor match:** ≥ 98% (keypoint descriptor similarity)
- **SSIM:** ≥ 98% (structural similarity of output images)
- **Result:** Script prints "Pass" or "Wrong"

**Expected Outcomes:**
- ✅ Single-rank: Should match serial baseline (identical output)
- ✅ Multi-rank: Pass validation (≥98% match, ≥98% SSIM)
- ✅ All octaves processed correctly (check profiler output in logs)
- ✅ Graceful handling of small tiles (look for "rank-0-only mode" messages)
- ⚠️ Speedup: Limited by Amdahl's law (only Gaussian pyramid parallelized in Phase 1)

**Profiler Analysis:**
Check logs for:
- Gaussian blur time (should decrease with more ranks)
- MPI communication overhead (should be < 20% of total)
- Gather/scatter overhead (visible in MPI_ sections)
- Rank-0-only mode activation for high octaves

### Phase 2: Complete SIFT Parallelization

**Completed stages:**

1. **✅ DoG Pyramid** (Phase 2 - October 18, 2025)
   - Implemented `generate_dog_pyramid_parallel()`
   - Embarrassingly parallel - zero communication overhead
   - Each rank processes local tiles independently
   - Performance: ~17.95 ms per rank (testcase 00, 4 ranks)
   - Details: See `PHASE2_DOG_IMPLEMENTATION.md`

2. **✅ Keypoint Detection** (Phase 3 - October 18, 2025)
   - Implemented `find_keypoints_parallel()`
   - Interior ownership rule (1 pixel inside boundary)
   - Zero communication during detection
   - Performance: ~45.35 ms per rank (testcase 00, 4 ranks)
   - Gathering: ~227 ms for MPI_Gather + MPI_Gatherv
   - Details: See `PHASE3_KEYPOINT_DETECTION.md`

**Remaining stages:**

3. **On-the-fly Gradients** ⭐ (Next: Phase 4 - major optimization)
   - Parallelize extrema scanning with OpenMP
   - Implement ownership rule for boundary keypoints
   - Gather keypoints from all ranks

3. **On-the-fly Gradients** ⭐ (major optimization)
   - Remove `generate_gradient_pyramid()` entirely
   - Compute gradients during orientation/descriptor phases
   - Expected: eliminate 36.6% of runtime

4. **Orientation Assignment**
   - Check keypoint support window vs tile bounds
   - Migrate keypoints to neighbors if needed (tiny messages)
   - `#pragma omp parallel for schedule(dynamic)` over keypoints

5. **Descriptor Computation**
   - Similar to orientation (dynamic scheduling)
   - Per-thread private histograms
   - Trilinear interpolation with on-the-fly gradients

6. **Result Gathering**
   - `MPI_Gatherv` for keypoint records
   - Serialize (x, y, octave, scale, descriptor[128])

### Phase 3: Performance Optimization (After Correctness)

**TODO: Optimize Downsampling Strategy**
- **Current:** Solution 1 (Gather-Downsample-Scatter)
  - ✅ Correct and simple
  - ⚠️ Communication overhead: 2 collectives per octave
  - Overhead: O(W×H) data movement per octave
  
- **Future:** Solution 2 (Aligned Tile-Based Downsampling)
  - Force tile boundaries to align on even coordinates
  - Each rank downsamples only interior pixels
  - After downsampling, exchange 1-pixel borders
  - Benefits:
    - ✅ Eliminates gather/scatter overhead
    - ✅ Better scalability for large images
    - ⚠️ More complex boundary handling
  - Implementation steps:
    1. Modify `TileInfo::compute_for_octave()` to ensure even-aligned tiles
    2. Add boundary pixel exchange after local downsampling
    3. Test and validate against Solution 1 output

---

## 📊 Performance Expectations

**Current Implementation (Phase 1 - Gaussian Pyramid Only):**

**Serial baseline breakdown (approximate):**
- Gaussian pyramid generation: ~40-50% of total time
- Gradient pyramid generation: ~36.6%
- Keypoint detection & refinement: ~10%
- Orientation & descriptor: ~24%

**Phase 1 parallelization (current):**
- ✅ **Parallelized:** Gaussian pyramid generation only
- ⚠️ **Sequential:** All other stages (DoG, gradient, keypoint, descriptor)
- **Best-case speedup:** 1.67× to 2× for Gaussian-dominated workloads
  - Limited by Amdahl's law: S = 1 / ((1-P) + P/N)
  - If Gaussian is 50% and we get 8× speedup on that part: S ≈ 1.77×
- **Communication overhead:**
  - Halo exchanges: 2 per blur (vertical + horizontal)
  - Gather-scatter: 2 per octave transition (7 transitions for 8 octaves)
  - Pyramid gather to rank 0: 8 octaves × 8 scales = 64 gathers

**Performance bottlenecks:**
- ✅ Halo exchange overlapped with computation
- ⚠️ Gather-scatter downsampling (2 collectives per octave)
- ⚠️ Full pyramid gather to rank 0 (for serial processing)
- ⚠️ Serial processing of DoG, gradient, keypoint stages (not parallelized yet)

**Future optimization targets (Phase 2+):**
- **On-the-fly gradients:** Eliminate gradient pyramid (36.6% time savings)
- **Parallel keypoint processing:** Near-linear scaling (embarrassingly parallel)
- **Parallel descriptor computation:** Near-linear scaling
- **Target with full parallelization (36 ranks):** 10-15× overall speedup

---

## 🔧 Debugging Guide

**Common Issues & Solutions:**

**1. MPI Errors:**
- **Deadlock:** Ensure all ranks call collective operations (Bcast, Allreduce, gather, scatter)
- **Invalid communicator:** Call `grid.finalize()` before `MPI_Finalize()`, not after
- **Thread level:** Verify `provided >= MPI_THREAD_FUNNELED` after `MPI_Init_thread()`
- **Rank mismatch:** Check that operations use correct communicator (cart_comm vs COMM_WORLD)

**2. Halo Exchange Issues:**
- **Buffer overrun:** Verify halo width doesn't exceed tile dimensions
- **Message size mismatch:** For odd-sized images, check neighbor dimension computation
- **Wrong data:** Verify pack/unpack indexing matches send/recv sizes
- **Debugging aid:** Add debug prints for tile dimensions and halo widths
  ```cpp
  if (rank == 0 || tile.width < 50) {
      printf("[Rank %d] Octave %d: tile=%dx%d, halo=%d\n", 
             rank, octave, tile.width, tile.height, halo_width);
  }
  ```

**3. OpenMP Issues (when re-enabled):**
- **Race conditions:** Check shared variable access in parallel regions
- **False sharing:** Pad per-thread data structures or use thread-private variables
- **Reproducibility:** Set `OMP_NUM_THREADS=1` for deterministic debugging
- **Affinity:** Export `OMP_PROC_BIND=close` and `OMP_PLACES=cores`

**4. Correctness Validation:**
- **Single-rank test:** Must match serial baseline exactly (same RNG, same algorithm)
- **Multi-rank test:** Use validation script (descriptor match ≥98%, SSIM ≥98%)
- **Octave-by-octave:** Save pyramid images and compare with numpy/matplotlib
- **Numerical precision:** Expect minor differences (<1e-5) due to floating-point order

**5. Resolved Issues (for reference):**

**Octave boundary artifacts (RESOLVED):**
   - **Symptom:** Differences explode at higher octaves (2+) with multiple ranks
   - **Diagnosis steps:**
     1. Export pyramid images per octave: `pyramid[oct][scale].save_text()`
     2. Compare with serial baseline octave by octave
     3. Check if differences appear at tile boundaries
     4. Verify if errors accumulate across octaves
   - **Root cause:** Independent downsampling on each rank
     - Nearest-neighbor interpolation uses `round()` for coordinate mapping
     - Adjacent tiles round differently at boundaries → discontinuities
     - Errors compound: octave N uses downsampled output from octave N-1
   - **Solution:** Gather-downsample-scatter (ensures consistent sampling)
   - **Prevention:** Always test multi-rank with np>1 AND validate each octave

6. **Halo exchange mismatch for odd-sized images (RESOLVED):**
   - **Symptom:** Differences explode at high octaves (e.g., octave 6, 15×19 image)
   - **Diagnosis steps:**
     1. Check if image dimensions at the problematic octave are odd numbers
     2. Verify tile dimensions differ across ranks
     3. Check MPI message sizes in halo exchange
   - **Root cause:** Odd-sized image partitioning creates unequal tile sizes
     - 15 pixels / 2 ranks → rank 0: 7 pixels, rank 1: 8 pixels
     - Halo exchange assumed uniform tile sizes
     - Send size (sender's width) ≠ Recv size (receiver's width) → mismatch
   - **Solution:** Compute neighbor dimensions deterministically and use for message sizes
   - **Prevention:** Test with odd-sized images at high octaves (dimensions not divisible by 2^octave × num_ranks)

**Profiling:**
```bash
# Run with profiling output
mpirun -np 4 ./hw2 input.jpg output.jpg output.txt

# Check profiler output for:
# - gaussian_blur_parallel timing
# - Halo exchange overhead
# - Load balance across ranks
# - gather_downsample_scatter overhead per octave
```

**Debugging octave consistency:**
```bash
# Export pyramids and compare with Python
cd results/tmp
python3 << EOF
import numpy as np
import matplotlib.pyplot as plt

# Load octave images
parallel = np.loadtxt('2_0.txt')  # Octave 2, scale 0 (parallel)
serial = np.loadtxt('../../golden_pyramid/2_0.txt')  # Serial baseline

# Compute differences
diff = np.abs(parallel - serial)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
print(f"Differences > 0.01: {(diff > 0.01).sum()}")

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(parallel); plt.title('Parallel'); plt.colorbar()
plt.subplot(132); plt.imshow(serial); plt.title('Serial'); plt.colorbar()
plt.subplot(133); plt.imshow(diff); plt.title('Difference'); plt.colorbar()
plt.savefig('octave2_comparison.png')
EOF
```

---

## 📝 Code Structure

```
hw2/
├── include/
│   ├── mpi_utils.hpp           ← MPI infrastructure
│   │                             - CartesianGrid: 2D process topology
│   │                             - TileInfo: Tile bounds computation
│   │                             - HaloBuffers: Halo exchange storage
│   │                             - Functions: pack/unpack, exchange, gather/scatter
│   ├── image.hpp               ← Parallel image operations
│   │                             - gaussian_blur_parallel: Distributed blur with halos
│   │                             - resize_parallel: Distributed interpolation
│   ├── sift.hpp                ← Parallel SIFT pipeline
│   │                             - generate_gaussian_pyramid_parallel
│   │                             - find_keypoints_and_descriptors_parallel
│   ├── profiler.hpp            ← Performance profiling
│   │                             - Hierarchical timing, MPI-aware
│   │                             - PROFILE_FUNCTION(), PROFILE_SCOPE(), PROFILE_MPI()
│   ├── sequential/
│   │   ├── image.hpp           ← Pure sequential image API (no MPI)
│   │   │                         - Image struct, resize, blur, interpolation
│   │   └── sift.hpp            ← Pure sequential SIFT API (no MPI)
│   │                             - Full SIFT pipeline, used by parallel version
│   └── stb/
│       ├── image.h             ← STB image loading (header-only)
│       └── image_write.h       ← STB image saving (header-only)
├── src/
│   ├── hw2.cpp                 ← Main program
│   │                             - MPI initialization, grid setup
│   │                             - Image I/O, SIFT invocation
│   │                             - Profiler reporting
│   ├── sift.cpp                ← Parallel SIFT implementation
│   │                             - Gaussian pyramid: distributed + adaptive
│   │                             - Pyramid gathering logic
│   │                             - Serial continuation (DoG, keypoints, descriptors)
│   ├── image.cpp               ← Parallel image implementation
│   │                             - Gaussian blur: separable convolution + halos
│   │                             - Resize: broadcast + distributed computation
│   ├── mpi_utils.cpp           ← MPI utilities implementation
│   │                             - Cartesian grid creation
│   │                             - Tile computation (with octave scaling)
│   │                             - Nonblocking halo exchange (Isend/Irecv)
│   │                             - Tile-based scatter/gather
│   ├── profiler.cpp            ← Profiling implementation
│   │                             - Timing data collection
│   │                             - MPI aggregation (gather to rank 0)
│   │                             - Hierarchical report printing
│   ├── sequential/
│   │   ├── image.cpp           ← Sequential image operations
│   │   └── sift.cpp            ← Sequential SIFT implementation
│   └── validate.cpp            ← Descriptor validation tool
│                                 - Compares output against golden reference
│                                 - Used by validation script
├── CMakeLists.txt              ← Build configuration
│                                 - MPI + OpenMP linkage
│                                 - Debug/Release configs
│                                 - macOS Clang compatibility
├── scripts/
│   ├── validate.py             ← Validation script (descriptor + SSIM)
│   ├── run_debug               ← Helper: run Debug build
│   ├── run_release             ← Helper: run Release build
│   └── ...
└── assets/
    ├── testcases/              ← Input images
    ├── goldens/                ← Reference outputs
    └── *.md                    ← Design docs
```

---

## 🎯 Next Session Plan

1. Build and test current implementation
2. Debug any compilation or runtime errors
3. Validate correctness against golden outputs
4. Profile and optimize halo exchange
5. Implement remaining SIFT stages (DoG, keypoints, descriptors)
6. Integrate on-the-fly gradients
7. Full system testing and optimization

---

## Notes

- **Thread safety:** MPI_THREAD_FUNNELED ensures only master thread calls MPI
- **Affinity:** Set `OMP_PROC_BIND=close` for better cache locality
- **Scalability:** Current design targets 1-3 nodes, up to 36 ranks
- **Future optimization:** MPI-IO for image loading, overlap more stages

---

## 🎨 Code Organization Philosophy

The codebase follows a **clean separation strategy** between sequential and parallel implementations:

### Sequential (`sequential/` subdirectory)
- **Pure algorithms** with no MPI/distributed computing dependencies
- **Reference implementations** for correctness validation
- **Easy to understand** - focuses solely on SIFT algorithm logic
- **Portable** - can be compiled without MPI if needed

### Parallel (main directory)
- **Distributed implementations** using MPI + OpenMP
- **Includes sequential headers** to leverage base implementations
- **Wraps or extends** sequential functions with parallel logic
- **Communication-aware** - handles tiles, halos, and data distribution

### Benefits
1. **Maintainability**: Clear separation reduces cognitive load
2. **Testing**: Easy to validate parallel against sequential baseline
3. **Development**: Can work on either version independently
4. **Scalability**: Pattern extends cleanly to new parallel functions

### Example Usage Pattern
```cpp
// Parallel code includes both:
#include "sequential/image.hpp"  // For Image struct and utilities
#include "image.hpp"              // For gaussian_blur_parallel()

// Sequential code only needs:
#include "sequential/image.hpp"  // Self-contained
```

---

## 📋 Summary

**Implementation Status: Phase 1 Complete ✅**

**What's Done:**
- ✅ MPI infrastructure (Cartesian grid, tile management, halo exchange)
- ✅ Parallel Gaussian blur (separable convolution, overlap communication)
- ✅ Parallel Gaussian pyramid (adaptive octave processing)
- ✅ Parallel image resize (distributed upscaling/downsampling)
- ✅ Profiler with MPI support (hierarchical timing, rank aggregation)
- ✅ Bug fixes: boundary artifacts, odd-sized images, small tiles
- ✅ Build system (MPI + OpenMP configured, macOS compatible)
- ✅ Validation infrastructure (descriptor matching, SSIM comparison)

**What's Next:**
- 🔄 Testing & validation (single-rank, multi-rank, edge cases)
- 🔄 Performance benchmarking (measure speedup, identify bottlenecks)
- 🔜 Phase 2: Parallel DoG, keypoint detection
- 🔜 Phase 3: On-the-fly gradients, parallel orientation/descriptor
- 🔜 OpenMP re-enablement (after MPI correctness validated)

**Key Design Decisions:**
- **Clean separation:** Sequential code in `sequential/` subdirectory, no MPI dependencies
- **Adaptive processing:** Automatic switch to rank-0-only for small tiles
- **Gather-scatter downsampling:** Ensures correctness, acceptable overhead
- **MPI-only debugging:** OpenMP disabled to isolate MPI correctness issues
- **Profiler integration:** Track MPI communication separately from computation

**Validation Criteria:**
- Single-rank output must match serial baseline exactly
- Multi-rank descriptor match ≥ 98%
- Multi-rank SSIM ≥ 98%
- All octaves processed correctly (check console output)

**Known Limitations:**
- Sequential processing after Gaussian pyramid (limits speedup)
- Full pyramid gather to rank 0 (communication overhead)
- Gather-scatter downsampling (could be optimized with tile-aligned approach)
- OpenMP disabled (will restore after validation)

**Code Quality:**
- Well-documented with inline comments
- Hierarchical profiling for performance analysis
- Debug output for troubleshooting tile/halo issues
- Validation script for automated testing

---

**Last Updated:** October 17, 2025  
**Status:** Ready for comprehensive testing and validation
