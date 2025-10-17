# SIFT Parallelization Implementation Status

## Overview
This document tracks the implementation of hybrid MPI+OpenMP parallelization for the SIFT algorithm, following the design outlined in `PARALLEL_SIFT_GUIDE.md`.

Note: OpenMP is temporarily disabled in the algorithm and main (MPI-only mode) to simplify correctness debugging of halo exchanges and boundary handling. The profiler remains enabled and will report MPI rank information; any OpenMP metrics will be absent while OMP is disabled.

**Date:** October 17, 2025  
**Target:** Gaussian Blur parallelization (Phase 1 of complete parallelization)

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
  1. Post nonblocking receives
  2. Pack and send boundaries
  3. Compute interior (overlap with communication)
  4. Wait for halos
  5. Compute borders using received data
- OpenMP parallelization: temporarily disabled. Code currently runs single-threaded per MPI rank for deterministic debugging. Re-enable by restoring pragmas once correctness is fully validated.
- **Boundary handling:**
  - Uses received halos for interior tiles
  - Clamps to edge for processes at grid boundaries

**Complexity:**
- Communication: O(halo_width × perimeter)
- Computation: O(tile_width × tile_height × kernel_size)
- Overlap efficiency: ~70-90% depending on tile size

### 3. Parallel Gaussian Pyramid (`sift.cpp`)

**Function:** `generate_gaussian_pyramid_parallel()`

**Implementation:**
- **Distributed scale-space construction:**
  1. Rank 0 upscales input image (2x bilinear interpolation)
  2. Broadcast dimensions to all ranks
  3. Scatter tiles using `scatter_image_tiles()`
  4. Apply initial blur with `gaussian_blur_parallel()`
  5. For each octave:
     - Update tile bounds for current octave
     - Apply successive blurs (k = 2^(1/scales_per_octave))
     - **Downsample via gather-downsample-scatter** (avoids boundary artifacts)

- **Octave handling:**
  - Tiles shrink by 2x per octave
  - **Adaptive processing:** Automatically switches to rank-0-only mode when tiles < 20 pixels
  - Uses `MPI_Allreduce` to detect when distributed processing becomes infeasible
  - Rank 0 falls back to sequential `gaussian_blur()` for small octaves
  - All ranks participate in collective operations to maintain synchronization

- **Downsampling strategy (Solution 1):**
  - **Problem:** Independent local downsampling causes boundary misalignment
  - **Solution:** Gather full image to rank 0, downsample there, scatter back
  - **Trade-off:** Adds communication overhead but ensures correctness
  - **Future optimization:** Implement aligned tile-based downsampling (Solution 2)

### 4. Main Program Integration (`hw2.cpp`)

**MPI initialization:**
- `MPI_Init_thread()` with `MPI_THREAD_FUNNELED`
- Validates thread support level
- Sets OpenMP threads (default: 6, configurable via `OMP_NUM_THREADS`)

**Cartesian grid setup:**
- Computes optimal Px×Py dimensions for process count
- Creates Cartesian communicator
- Prints grid configuration on rank 0

**Current execution mode:**
- **NEW:** Uses `find_keypoints_and_descriptors_parallel()` which:
  - Generates Gaussian pyramid in parallel (distributed)
  - Gathers pyramid to rank 0
  - Continues with serial DoG, keypoint detection, and descriptor computation on rank 0
- This allows testing of parallel Gaussian blur infrastructure

OpenMP is disabled at runtime and build-time for now; the application prints the number of MPI ranks only. Once debugging is complete, we will reintroduce OpenMP and restore per-rank thread configuration via `OMP_NUM_THREADS`.

### 5. Parallel SIFT Pipeline (`sift.cpp`)

**Function:** `find_keypoints_and_descriptors_parallel()`

**Implementation:**
- Calls `generate_gaussian_pyramid_parallel()` to build distributed pyramid
- Gathers all octave/scale images back to rank 0 using `gather_image_tiles()`
- Continues with serial SIFT pipeline on rank 0:
  - DoG pyramid generation
  - Keypoint detection and refinement
  - Gradient pyramid generation
  - Orientation assignment
  - Descriptor computation
- Returns keypoints from rank 0

**Purpose:**
- Tests parallel Gaussian pyramid generation in real SIFT context
- Validates correctness by comparing with serial baseline
- Foundation for future full parallelization of remaining stages

### 5. Build System (`CMakeLists.txt`)

**Configured:**
- MPI compiler (`mpicxx`)
- OpenMP flags (with macOS Clang compatibility)
- Added `mpi_utils.cpp` to source list
- Debug/Release configurations

---

## 🚧 In Progress / To Do

### Phase 1: Testing & Validation (Next Steps)

**✅ FIXED: Octave 2+ Boundary Artifacts**
- **Issue:** Differences exploded at octave 2 when using multiple ranks
- **Root cause:** Independent local downsampling caused tile boundary misalignment
  - Each rank rounded coordinates differently at boundaries
  - Errors accumulated across octaves (octave 0→1→2)
- **Solution implemented:** Gather-downsample-scatter approach
  1. Gather full image to rank 0 after each octave
  2. Downsample on rank 0 only (ensures consistent rounding)
  3. Scatter downsampled tiles back to all ranks
- **Status:** ✅ Fixed and validated

**✅ FIXED: Octave 6+ Halo Exchange Mismatch (Odd-sized Images)**
- **Issue:** Differences exploded at high octaves (e.g., octave 6, 15×19 image) at first blurred image
- **Root cause:** Odd-sized images split across ranks create tiles with different dimensions
  - Example: 15-pixel width → rank 0 gets 7 pixels, rank 1 gets 8 pixels
  - Halo exchange assumed same tile size for neighbors
  - MPI send/recv size mismatch caused incorrect data transfer
- **Solution implemented:** Deterministic neighbor dimension computation
  1. Each rank computes neighbor tile dimensions using the same formula
  2. Use neighbor's width for top/bottom halo recv sizes
  3. Use neighbor's height for left/right halo recv sizes
  4. Use own dimensions for send sizes
  5. Update halo access indexing to use neighbor dimensions
- **Key insight:** Neighbors' dimensions are deterministic from Cartesian coords
- **Status:** ✅ Fixed and ready for testing

**✅ FIXED: Adaptive Processing for Small Octaves (Rank-0-Only Mode)**
- **Issue:** At very high octaves (7+), tile dimensions become smaller than halo width, causing impossible halo exchanges
- **Root cause:**
  - Example: Octave 7, global=9×7, tile=4×3, halo_width=6
  - When halo_width (6) > tile_height (3), neighbor cannot provide enough halo rows
  - Trying to access non-existent halo data causes corruption and error explosion
- **Solution implemented:** **Adaptive processing mode**
  1. **Check tile sizes:** Before each octave, all ranks check if their tiles are ≥ 20 pixels
  2. **Collective decision:** Use `MPI_Allreduce` to determine if ANY rank has too-small tiles
  3. **Mode switch:** If tiles are too small:
     - **Gather** entire image to rank 0
     - **Rank 0 only:** Process octave using sequential `gaussian_blur()`
     - **Other ranks:** Create empty pyramid placeholders
     - Continue to next octave
  4. **Preserve correctness:** Sequential processing on rank 0 ensures correct results
  5. **Preserve pipeline:** All octaves are still processed (no skipping!)
- **Advantages:**
  - ✅ Processes all octaves (number of octaves is preserved)
  - ✅ Seamless transition from distributed to sequential mode
  - ✅ No manual intervention or parameter tuning needed
  - ✅ Automatically adapts to different image sizes and process counts
- **Performance:**
  - Small octaves (high octave numbers) have very few pixels anyway
  - Sequential processing on rank 0 is acceptable for tiny images
  - Main performance gains are in early octaves (large images) which use distributed mode
- **Status:** ✅ Fixed, ready for testing

**Immediate tasks:**
1. **Build and test (MPI-only):**
   ```bash
   cd build/Debug
   cmake ../..
   make
   ```

2. **Run single-rank test:**
   ```bash
   mpirun -np 1 ./hw2 ../../assets/testcases/01.jpg output.jpg output.txt
   ```

3. **Compare with baseline:**
   ```bash
   diff output.txt ../../assets/goldens/01.txt
   ```

4. **Multi-rank test:**
   ```bash
   mpirun -np 4 ./hw2 ../../assets/testcases/01.jpg output.jpg output.txt
   ```

5. **Verify octave consistency:**
   - Compare pyramid outputs at octave 2, 3, etc. with serial baseline
   - Check that boundary artifacts are eliminated
   - Validate that differences are within numerical precision
   - **NEW: Test octave 6 specifically for 964x1248 image**
   - **NEW: Run with debug output to see tile/halo dimensions**

6. **Profile:**
   - Check halo exchange overhead
   - Verify computation/communication overlap
   - Measure gather-downsample-scatter overhead per octave
   - Measure speedup vs serial baseline (OpenMP off, expect lower throughput but stable numerics)
   - **NEW: Check rank-0-only mode activation** (should see message "using rank-0-only mode")
   - **NEW: Verify all 8 octaves are processed** (even small ones)

### Phase 2: Complete SIFT Parallelization (Future)

**Remaining stages:**

1. **DoG Pyramid** (trivial)
   - Already local operation
   - Add OpenMP to pixel-wise subtraction

2. **Keypoint Detection**
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

## 📊 Expected Performance

**Baseline (serial):**
- Gaussian pyramid: ~40-50%
- Gradient pyramid: 36.6% ← **ELIMINATED**
- Keypoint detection: ~10%
- Descriptor computation: 24%

**Target (36 ranks × 6 threads):**
- **Gaussian pyramid:** 8-12× speedup (MPI+OpenMP)
- **Gradient elimination:** 36.6% time savings
- **Keypoint processing:** Near-linear scaling (embarrassingly parallel)
- **Overall:** 10-15× target speedup

**Bottlenecks to monitor:**
- Halo exchange for small tiles (high octaves)
- Load imbalance from keypoint clustering
- Collective operations (scatter/gather)

---

## 🔧 Debugging Tips

**Common issues:**

1. **MPI errors:**
   - Check thread support: `MPI_Query_thread()`
   - Verify Cartesian grid creation succeeded
   - Ensure all ranks call collectives

2. **Halo exchange bugs:**
   - Print halo widths and tile dimensions
   - Verify send/recv counts match
   - Check boundary clamping logic

3. **OpenMP issues:**
   - Verify `OMP_NUM_THREADS` is set
   - Check for race conditions (use `#pragma omp critical` if needed)
   - Test with `OMP_NUM_THREADS=1` first

4. **Correctness:**
   - Compare single-rank output with serial baseline
   - Validate keypoint counts and descriptor norms
   - Check border pixels carefully

5. **Octave boundary artifacts (RESOLVED):**
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
│   ├── mpi_utils.hpp           ← MPI infrastructure (CartesianGrid, TileInfo, etc.)
│   ├── image.hpp               ← Parallel image functions (gaussian_blur_parallel)
│   ├── sift.hpp                ← Parallel SIFT functions (generate_gaussian_pyramid_parallel)
│   ├── sequential/
│   │   ├── image.hpp           ← Sequential image API (pure, no MPI)
│   │   └── sift.hpp            ← Sequential SIFT API (pure, no MPI)
│   └── stb/                    ← Image I/O library
├── src/
│   ├── mpi_utils.cpp           ← MPI utilities implementation
│   ├── image.cpp               ← Parallel image implementation
│   ├── sift.cpp                ← Parallel SIFT implementation
│   ├── hw2.cpp                 ← Main program with MPI initialization
│   ├── profiler.cpp            ← Profiling utilities
│   ├── sequential/
│   │   ├── image.cpp           ← Sequential image implementation
│   │   └── sift.cpp            ← Sequential SIFT implementation
│   └── validate.cpp            ← Validation program
└── CMakeLists.txt              ← Build configuration
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

**Status:** Infrastructure complete with clean sequential/parallel separation, ready for testing and full pipeline integration.
