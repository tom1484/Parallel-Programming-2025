# Quick Reference: Parallel SIFT Implementation

## Using Test Scripts

The `scripts/` directory provides convenient wrappers for building, running, and validating:

### run_debug - Debug Single Testcase
```bash
# Usage: ./scripts/run_debug <testcase_id> [PE] [NP] [THREADS]
./scripts/run_debug 01              # Default: PE=2, NP=4, THREADS=6
./scripts/run_debug 04 2 9 6        # Custom: 9 MPI ranks, 6 threads each
```
**Features:**
- Auto-builds Debug configuration
- Reports MPI process bindings
- Shows execution time
- Auto-validates against golden reference
- Outputs "Pass" or "Wrong"

### run_release - Release Single Testcase
```bash
# Usage: ./scripts/run_release <testcase_id> [PE] [NP] [THREADS]
./scripts/run_release 04 2 16 6     # Performance test with 16 ranks
```
**Features:**
- Auto-builds Release configuration (-O3)
- Same validation as run_debug
- Use for performance measurements

### run_benchmark - Benchmark Multiple Testcases
```bash
# Usage: ./scripts/run_benchmark [N] [PE] [NP] [THREADS]
./scripts/run_benchmark 8 2 4 6     # Run first 8 testcases with 4 ranks
./scripts/run_benchmark             # Default: 10 testcases, 4 ranks
```
**Features:**
- Runs N testcases (default: 10)
- Reports individual timing for each testcase
- Shows PASSED/FAILED for each
- Reports total and average time
- Saves logs to `results/*.log` and `results/*.val.log`

**Example output:**
```
Testcase 01: 0.123456 seconds - PASSED
Testcase 02: 0.234567 seconds - PASSED
...
Total time: 2.345678 seconds
Average time per testcase: 0.234568 seconds
```

### Remote Server Testing (srun_* scripts)

For HPC clusters using SLURM, equivalent `srun_debug`, `srun_release`, and `srun_benchmark` scripts are provided. These use `srun` instead of `mpirun` with appropriate SLURM parameters.

### Script Parameters
- **testcase_id**: Test number (01-08, or 01-04 for common tests)
- **PE**: Processing elements per socket (default: 2)
- **NP**: Number of MPI ranks (default: 4)
  - 1 rank: Serial baseline (should match exactly)
  - 4 ranks: 2×2 grid (typical)
  - 9 ranks: 3×3 grid (good for testing)
  - 16 ranks: 4×4 grid (scalability test)
- **THREADS**: OpenMP threads per rank (default: 6, currently unused)

## Quick Testing Examples

```bash
# Verify correctness with single rank (should match serial baseline)
./scripts/run_debug 01 2 1 6

# Test with typical configuration
./scripts/run_debug 01 2 4 6

# Test scalability with large image
./scripts/run_release 04 2 16 6

# Benchmark all testcases
./scripts/run_benchmark 8 2 4 6

# Compare different rank counts (use run_benchmark for each)
./scripts/run_benchmark 8 2 1 6   # Baseline
./scripts/run_benchmark 8 2 4 6   # 4 ranks
./scripts/run_benchmark 8 2 16 6  # 16 ranks
```

## Manual Build (if needed)

```bash
# Debug build
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug -j

# Release build
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -j
```

## Manual Validation (if needed)

```bash
python3 scripts/validate.py \
    results/01.txt assets/goldens/01.txt \
    results/01.jpg assets/goldens/01.jpg
```

## Profiler Output Interpretation

```
Function Name              Total(ms)    Min(ms)    Max(ms)  %Total  Calls   Avg(ms)
├─ SIFT_TOTAL             12345.67     ...        ...      100.0%    1     12345.67
│  ├─ gaussian_pyramid     5678.90     ...        ...       46.0%    1      5678.90
│  │  ├─ gaussian_blur      456.78     5.23       89.45     8.0%   48        9.52
│  │  │  └─ MPI_exchange     34.56     0.12        2.34     0.6%   96        0.36
│  │  └─ MPI_Bcast           12.34     0.05        0.15     0.2%   16        0.77
│  └─ gather_pyramid        123.45     ...        ...       1.0%    1       123.45
```

**Key metrics:**
- **Total time:** Overall execution time
- **MPI_ sections:** Communication overhead (should be < 20%)
- **gaussian_blur:** Main parallel computation
- **Calls:** Number of invocations (48 = 8 octaves × 6 blurs)

## Expected Console Output

```
Running with 4 MPI ranks (OpenMP disabled)
Using 2 x 2 process grid
Octave 7 and beyond: tiles too small (min=4x3), switching to rank-0-only mode
Execution time: 1234.56 ms
Found 1234 keypoints.

Profiling Report (Rank 0):
[... profiler output ...]
```

## Testcase Characteristics

| Testcase | Dimensions | Pixels | Keypoints | Notes |
|----------|-----------|--------|-----------|-------|
| 01.jpg   | 152×185   | 28K    | ~200      | Small, tests edge cases |
| 02.jpg   | 256×256   | 65K    | ~400      | Square, balanced |
| 03.jpg   | 512×384   | 196K   | ~800      | Medium |
| 04.jpg   | 964×1248  | 1.2M   | ~2000     | Large, tests scalability |

## File Locations

```
hw2/
├── build/
│   ├── Debug/hw2         ← Debug executable
│   └── Release/hw2       ← Release executable
├── assets/
│   ├── testcases/        ← Input images (01-08.jpg)
│   └── goldens/          ← Reference outputs (01-08.txt)
├── results/              ← Your outputs (*.jpg, *.txt)
└── scripts/
    └── validate.py       ← Validation script
```

## Validation Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Descriptor Match | ≥ 98% | Percentage of keypoints with matching descriptors |
| SSIM | ≥ 98% | Structural similarity of output images |

**Pass:** Both thresholds met  
**Wrong:** Either threshold not met

## Common Errors & Solutions

### Error: MPI_ERR_TRUNCATE
**Cause:** Message size mismatch (send size ≠ recv size)  
**Solution:** Check neighbor dimension computation in halo exchange

### Error: Segmentation fault in gather
**Cause:** Wrong tile bounds or unallocated buffer  
**Solution:** Verify tile.compute_for_octave() called correctly

### Error: Deadlock
**Cause:** Not all ranks calling collective operation  
**Solution:** Ensure MPI_Bcast, MPI_Allreduce called by ALL ranks

### Error: Wrong output (Wrong from validation)
**Cause:** Numerical differences exceed threshold  
**Solution:** Check single-rank first, then compare pyramid images octave by octave

## Performance Tips

1. **For testing:** Use Debug build (faster compile, easier debugging)
2. **For benchmarking:** Use Release build (-O3 optimization)
3. **For profiling:** Watch MPI communication time (should be < 20% of total)
4. **For scaling:** Test with powers of 2 or 3 (better grid factorization)
5. **For validation:** Always test single-rank first (eliminates MPI from debugging)

## Environment Variables

The test scripts automatically set optimal MPI and OpenMP variables:
- `OMP_NUM_THREADS`: Set via script parameter (default: 6)
- `OMP_PROC_BIND=close`: Thread affinity
- `OMP_PLACES=cores`: Bind threads to cores
- `--map-by socket:PE=$PE`: MPI rank placement

**Additional environment variables (if needed):**
```bash
# Disable KNEM if causing issues (platform-specific)
export OMPI_MCA_btl_sm_use_knem=0

# Set MPI timeout (seconds)
export MPIEXEC_TIMEOUT=300
```

## Testing Workflow

**Step 1: Verify Correctness**
```bash
# Single rank should match serial baseline exactly
./scripts/run_debug 01 2 1 6

# Multi-rank should pass validation (≥98%)
./scripts/run_debug 01 2 4 6
./scripts/run_debug 01 2 9 6
```

**Step 2: Test Edge Cases**
```bash
# Small image
./scripts/run_debug 01 2 4 6

# Large image (triggers rank-0-only mode at high octaves)
./scripts/run_debug 04 2 4 6
```

**Step 3: Performance Testing**
```bash
# Benchmark with different rank counts
./scripts/run_benchmark 8 2 1 6   # Baseline (serial)
./scripts/run_benchmark 8 2 4 6   # 2×2 grid
./scripts/run_benchmark 8 2 9 6   # 3×3 grid
./scripts/run_benchmark 8 2 16 6  # 4×4 grid
```

**Step 4: Analyze Results**
- Check `results/*.log` for profiler output and console messages
- Check `results/*.val.log` for validation results
- Look for "rank-0-only mode" messages for high octaves
- Compare timing across different rank counts

## Key Code Locations

| Component | Header | Implementation |
|-----------|--------|----------------|
| MPI Utils | `include/mpi_utils.hpp` | `src/mpi_utils.cpp` |
| Parallel Blur | `include/image.hpp` | `src/image.cpp` |
| Parallel Pyramid | `include/sift.hpp` | `src/sift.cpp` |
| Main Program | - | `src/hw2.cpp` |
| Profiler | `include/profiler.hpp` | `src/profiler.cpp` |
| Sequential (reference) | `include/sequential/*.hpp` | `src/sequential/*.cpp` |

## Documentation

- **IMPLEMENTATION_STATUS.md** - Current status, testing checklist
- **IMPLEMENTATION_SUMMARY.md** - What was implemented and how
- **PARALLEL_SIFT_GUIDE.md** - Overall design and strategy
- **SIFT_ALGORITHM_SUMMARY.md** - Algorithm reference
- **QUICK_REFERENCE.md** - This file

---

**Last Updated:** October 17, 2025  
**Status:** Phase 1 (Gaussian Pyramid) Complete
