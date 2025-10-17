# SIFT Algorithm Implementation Summary

This document provides a comprehensive breakdown of the SIFT (Scale-Invariant Feature Transform) algorithm implementation in this codebase.

## Overall Flow

The main execution flow (from `hw2.cpp`):

1. Load input image from file
2. Convert RGB to grayscale (if needed)
3. Execute SIFT algorithm to find keypoints and compute descriptors
4. Save results (keypoint coordinates, octave, scale, and 128-dimensional descriptors) to output text file
5. Visualize keypoints on image and save as output image

---

## Core SIFT Algorithm Pipeline

The main algorithm is implemented in `find_keypoints_and_descriptors()` in `sift.cpp`, which orchestrates six major phases:

### Phase 1: Scale-Space Construction

**Function:** `generate_gaussian_pyramid()`

**Purpose:** Build a multi-scale representation of the image to achieve scale invariance.

**Process:**
- Upscale input image by 2x using bilinear interpolation
- Apply initial Gaussian blur to reach base sigma (σ_min / MIN_PIX_DIST)
- Create pyramid with N octaves (default: 8)
  - Each octave contains `scales_per_octave + 3` images (default: 5 + 3 = 8 images)
  - Within each octave, progressively blur images with increasing sigma values
  - Sigma values follow: σ_k = σ_base × k^i where k = 2^(1/scales_per_octave)
- Between octaves, downsample by 2x using nearest-neighbor interpolation
- Each octave's base image is derived from a specific scale in the previous octave

**Output:** Multi-octave pyramid where each level contains multiple Gaussian-blurred images at different scales

**Key Parameters:**
- `SIGMA_MIN = 0.8`: Minimum sigma in original image
- `N_OCT = 8`: Number of octaves
- `N_SPO = 5`: Number of scales per octave

---

### Phase 2: Difference-of-Gaussian (DoG) Pyramid

**Function:** `generate_dog_pyramid()`

**Purpose:** Approximate Laplacian of Gaussian for efficient blob detection.

**Process:**
- For each octave, compute pixel-wise difference between consecutive Gaussian images
- Creates `scales_per_octave + 2` DoG images per octave (one less than Gaussian images)
- DoG approximates scale-normalized Laplacian: DoG(x,y,σ) = G(x,y,kσ) - G(x,y,σ)

**Output:** DoG pyramid used for extrema detection in the next phase

---

### Phase 3: Keypoint Detection

**Function:** `find_keypoints()`

**Purpose:** Locate scale-space extrema that are potential interest points.

**Sub-steps:**

#### 3.1 Extrema Detection
**Function:** `point_is_extremum()`
- Check each pixel in DoG images (excluding borders and first/last scale within each octave)
- Compare pixel value with 26 neighbors:
  - 9 neighbors in the same scale
  - 9 neighbors in the scale above
  - 9 neighbors in the scale below
- Mark as candidate keypoint if it's either a local maximum or minimum

#### 3.2 Keypoint Refinement
**Function:** `refine_or_discard_keypoint()`

**Sub-process:**

**a) Quadratic Fitting** (`fit_quadratic()`):
- Fit 3D quadratic function (Taylor expansion) to localize extremum with sub-pixel accuracy
- Compute gradient vector (g) and Hessian matrix (H) at the discrete extremum
- Solve for offset: x̂ = -H^(-1) × g
- Calculate interpolated position offsets in scale dimension and spatial dimensions (x, y)
- Update keypoint with interpolated extremum value

**b) Iterative Refinement:**
- Adjust keypoint to nearest discrete coordinates based on offsets
- Repeat up to `MAX_REFINEMENT_ITERS = 5` iterations
- Stop when converged (all offsets < 0.6) or moved out of valid scale range

**c) Filtering Criteria:**

1. **Contrast Threshold:**
   - Reject low-contrast keypoints where |extremum_val| < `C_DOG = 0.015`
   - Ensures detected features are significant

2. **Edge Response** (`point_is_on_edge()`):
   - Compute eigenvalue ratio of 2D Hessian matrix
   - Edge response metric: (tr(H))² / det(H)
   - Reject keypoints with ratio > (C_EDGE + 1)² / C_EDGE where `C_EDGE = 10`
   - Eliminates poorly localized points along edges

#### 3.3 Coordinate Transformation
**Function:** `find_input_img_coords()`
- Convert keypoint coordinates from pyramid space to original image space
- Calculate position: (x, y) = MIN_PIX_DIST × 2^octave × (offset + discrete_coord)
- Calculate corresponding sigma: σ = 2^octave × σ_min × 2^(scale/scales_per_octave)

**Output:** List of refined keypoints with both discrete (i, j, octave, scale) and continuous (x, y, sigma) coordinates

---

### Phase 4: Gradient Pyramid

**Function:** `generate_gradient_pyramid()`

**Purpose:** Compute image gradients needed for orientation assignment and descriptor computation.

**Process:**
- For each Gaussian image in the pyramid, compute derivatives using central differences:
  - gx(x,y) = [I(x+1,y) - I(x-1,y)] / 2
  - gy(x,y) = [I(x,y+1) - I(x,y-1)] / 2
- Store gradients as 2-channel images (channel 0: gx, channel 1: gy)
- Maintains same pyramid structure as Gaussian pyramid

**Output:** Gradient pyramid parallel to Gaussian pyramid

---

### Phase 5: Orientation Assignment

**Function:** `find_keypoint_orientations()`

**Purpose:** Assign one or more dominant orientations to each keypoint for rotation invariance.

**Process:**

1. **Boundary Check:**
   - Discard keypoints too close to image borders
   - Minimum distance required: √2 × λ_desc × σ

2. **Gradient Accumulation:**
   - Define circular patch around keypoint with radius = 3 × λ_ori × σ
   - For each pixel in patch:
     - Compute gradient magnitude: m = √(gx² + gy²)
     - Compute gradient orientation: θ = atan2(gy, gx)
     - Apply Gaussian weighting: w = exp(-dist² / (2σ_patch²))
     - Accumulate weighted gradient magnitude into 36-bin orientation histogram

3. **Histogram Smoothing:**
   - Apply 3-element box filter (averaging with neighbors) 6 times
   - Smooths histogram for better peak detection

4. **Peak Detection:**
   - Find histogram peaks with value ≥ 80% of maximum peak
   - Use parabolic interpolation for sub-bin accuracy:
     - θ = θ_peak + (Δθ/2) × (h_prev - h_next) / (h_prev - 2h_peak + h_next)
   - Each peak represents a dominant orientation

5. **Multiple Orientations:**
   - Create separate keypoint instance for each dominant orientation
   - Enables matching under different rotations

**Output:** Vector of reference orientations (angles in radians) for the keypoint

**Key Parameters:**
- `LAMBDA_ORI = 1.5`: Controls patch size for orientation
- `N_BINS = 36`: Number of orientation bins (10° per bin)

---

### Phase 6: Descriptor Computation

**Function:** `compute_keypoint_descriptor()`

**Purpose:** Create a distinctive 128-dimensional feature vector for robust matching.

**Process:**

1. **Patch Extraction:**
   - Extract square patch around keypoint
   - Patch size: √2 × λ_desc × σ × (N_HIST + 1) / N_HIST
   - Covers region needed for 4×4 histogram grid with smooth boundaries

2. **Coordinate Rotation:**
   - Rotate each sample point relative to keypoint's reference orientation
   - Normalized coordinates relative to keypoint position and scale:
     - x' = [(x - kp.x)cos(θ) + (y - kp.y)sin(θ)] / σ
     - y' = [-(x - kp.x)sin(θ) + (y - kp.y)cos(θ)] / σ

3. **Histogram Construction** (`update_histograms()`):
   - Divide patch into 4×4 spatial regions (N_HIST = 4)
   - For each sample point:
     - Compute gradient magnitude and orientation (relative to reference)
     - Apply Gaussian weighting: w = exp(-dist² / (2σ_patch²))
     - **Trilinear interpolation:** Distribute weighted gradient to:
       - Neighboring spatial bins (2×2 regions)
       - Neighboring orientation bins (2 bins in 8-bin histogram)
     - Creates smooth descriptor that's less sensitive to small shifts
   - Results in 4×4 grid of 8-bin orientation histograms

4. **Feature Vector Generation** (`hists_to_vec()`):
   - Concatenate all histogram values: 4×4×8 = 128 dimensions
   - **First normalization:** Normalize to unit length (L2 norm)
   - **Illumination invariance:** Threshold all values at 0.2 (clips large gradients)
   - **Second normalization:** Normalize again to unit length
   - **Quantization:** Scale by 512 and clip to [0, 255] range
   - Convert to uint8_t array

**Output:** 128-element descriptor vector stored as uint8_t array in keypoint structure

**Key Parameters:**
- `LAMBDA_DESC = 6`: Controls patch size for descriptor
- `N_HIST = 4`: 4×4 spatial grid
- `N_ORI = 8`: 8 orientation bins per histogram

---

## Key Supporting Functions

From `image.cpp`:

### Image Processing Functions

- **`gaussian_blur()`**
  - Implements separable 2D Gaussian convolution
  - First convolves vertically, then horizontally
  - Kernel size: ceil(6σ), forced to be odd
  - More efficient than 2D convolution: O(n·m·k) vs O(n·m·k²)

- **`resize()`**
  - Bilinear interpolation: Weighted average of 4 nearest pixels
  - Nearest-neighbor interpolation: Rounds to closest pixel
  - Coordinate mapping accounts for pixel center alignment

- **`rgb_to_grayscale()`**
  - Converts RGB to luminance using standard weights
  - Formula: Y = 0.299R + 0.587G + 0.114B

- **`bilinear_interpolate()`** and **`nn_interpolate()`**
  - Handle sub-pixel image sampling
  - Bilinear: smooth, used for upsampling
  - Nearest-neighbor: fast, preserves sharp edges, used for downsampling

### Image Data Structure

- **Storage format:** CHW (Channel-Height-Width)
  - data[c × height × width + y × width + x]
  - Allows efficient channel-wise operations

- **Pixel access:**
  - `get_pixel()`: Handles boundary by clamping coordinates
  - `set_pixel()`: Direct access with bounds checking

---

## Algorithm Parameters

All default parameters are defined in `sift.hpp`:

### Scale Space
- `SIGMA_MIN = 0.8`: Minimum blur in original image
- `SIGMA_IN = 0.5`: Assumed blur in input image
- `MIN_PIX_DIST = 0.5`: Pixel distance in upsampled image
- `N_OCT = 8`: Number of octaves
- `N_SPO = 5`: Scales per octave (3 for DoG extrema detection)

### Keypoint Detection
- `MAX_REFINEMENT_ITERS = 5`: Maximum refinement iterations
- `C_DOG = 0.015`: Contrast threshold
- `C_EDGE = 10`: Edge response threshold

### Orientation & Descriptor
- `N_BINS = 36`: Orientation histogram bins
- `LAMBDA_ORI = 1.5`: Orientation patch size factor
- `N_HIST = 4`: Descriptor spatial bins (4×4)
- `N_ORI = 8`: Descriptor orientation bins
- `LAMBDA_DESC = 6`: Descriptor patch size factor

### Matching (not used in this implementation)
- `THRESH_ABSOLUTE = 350`: Absolute distance threshold
- `THRESH_RELATIVE = 0.7`: Ratio test threshold

---

## Profiling Support

The code includes a built-in profiler (`profiler.hpp/cpp`) that:
- Tracks execution time for each function/scope
- Provides hierarchical timing breakdown
- Shows percentage of parent time and total time
- Counts function calls and computes average time per call

Usage:
- `PROFILE_FUNCTION()`: Profile entire function
- `PROFILE_SCOPE(name)`: Profile specific code block

---

## Parallelization Opportunities for OpenMPI

Based on the algorithm structure, here are the most promising targets for distributed-memory parallelization:

### High Priority (Coarse-grained parallelism)

1. **Keypoint Processing (Phase 5 & 6)**
   - **Target:** Orientation assignment and descriptor computation
   - **Strategy:** Distribute keypoints across MPI processes
   - **Advantages:** 
     - Embarrassingly parallel
     - No dependencies between keypoints
     - Good load balancing if keypoints evenly distributed
   - **Communication:** Gather final keypoints at root process

2. **Octave-level Parallelism (Phases 1-4)**
   - **Target:** Gaussian pyramid, DoG pyramid, gradient pyramid generation
   - **Strategy:** Assign different octaves to different processes
   - **Challenges:** 
     - First octave depends on input image processing
     - Octaves have different computational costs (higher octaves are smaller)
   - **Communication:** Scatter octave assignments, gather results

3. **Keypoint Detection per Octave (Phase 3)**
   - **Target:** Finding extrema and refinement within each octave
   - **Strategy:** Process different octaves in parallel
   - **Advantages:** Natural data decomposition
   - **Communication:** Gather keypoint lists from all processes

### Medium Priority (Medium-grained parallelism)

4. **Image-level Parallelism within Octaves**
   - **Target:** Gaussian blur operations within an octave
   - **Strategy:** Distribute images within octave across processes
   - **Trade-off:** More communication overhead, smaller work units

5. **Gaussian Blur (Supporting Function)**
   - **Target:** Row-wise or tile-based decomposition
   - **Strategy:** Distribute image rows/tiles across processes
   - **Challenges:** 
     - Requires halo exchange for border regions
     - Separable convolution needs two parallel passes
   - **Best for:** Large images where blur is dominant cost

### Lower Priority (Fine-grained parallelism)

6. **Pixel-level Operations**
   - **Target:** DoG computation, gradient computation
   - **Strategy:** Domain decomposition with halo regions
   - **Trade-off:** High communication-to-computation ratio

### Recommended Strategy

**Hybrid approach:**
1. Distribute octaves across MPI processes (coarse level)
2. Within each octave, process scales in parallel when possible
3. Distribute keypoint processing across all available processes after detection
4. Use collective operations (MPI_Gather, MPI_Scatterv) efficiently

**Load Balancing Considerations:**
- Higher octaves (smaller images) complete faster
- Consider dynamic load balancing or weighted distribution
- Keypoint count varies significantly between images

**Communication Patterns:**
- Minimize data movement by keeping pyramid data local
- Use asynchronous communication where possible
- Gather final results at root for validation

---

## Validation

The `validate.cpp` program compares output against golden reference:
- Reads keypoints from text files (both test output and golden reference)
- Sorts keypoints by (i, j, octave, scale) for comparison
- For each golden keypoint:
  - Finds matching keypoints at same discrete position (i, j)
  - Computes Euclidean distance between 128D descriptors
  - Matches if minimum distance < 1.5
- Reports match percentage

This allows verification that parallelized implementation produces equivalent results.
