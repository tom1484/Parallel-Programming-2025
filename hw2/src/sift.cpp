#include "sift.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "image.hpp"
#include "mpi_utils.hpp"
#include "profiler.hpp"
#include "sequential/image.hpp"
#include "sequential/sift.hpp"

using namespace std;

// Parallel Gaussian pyramid generation with MPI+OpenMP
ScaleSpacePyramid generate_gaussian_pyramid_parallel(const Image& img, const TileInfo& base_tile,
                                                     const CartesianGrid& grid, float sigma_min, int num_octaves,
                                                     int scales_per_octave) {
    PROFILE_FUNCTION();

    // Compute base sigma and initial blur (same as serial version)
    float base_sigma = sigma_min / MIN_PIX_DIST;

    // Step 1: Rank 0 pre-generates all base images for each octave
    // Other ranks will wait at the broadcast, which is more efficient than having
    // all ranks do redundant resizing work
    vector<Image> full_octave_bases;      // Full base images on rank 0
    int base_width = 0, base_height = 0;  // Dimensions of octave 0 base image

    if (grid.rank == 0) {
        PROFILE_SCOPE("prepare_octave_bases");

        // Resize input to 2x and apply initial blur
        base_width = img.width * 2;
        base_height = img.height * 2;
        Image upscaled = img.resize(base_width, base_height, Interpolation::BILINEAR);

        float sigma_diff = sqrt(base_sigma * base_sigma - 1.0f);
        Image base_img = gaussian_blur(upscaled, sigma_diff);

        full_octave_bases.push_back(base_img);

        // Generate base images for remaining octaves by downsampling
        // These will be refined during pyramid generation using the proper scale images
        for (int oct = 1; oct < num_octaves; oct++) {
            const Image& prev_base = full_octave_bases.back();
            Image next_base = prev_base.resize(prev_base.width / 2, prev_base.height / 2, Interpolation::NEAREST);
            full_octave_bases.push_back(next_base);
        }
    }

    // Broadcast base dimensions to all ranks
    // Note: The time shown here includes wait time for non-rank-0 processes
    {
        PROFILE_MPI("Bcast_base_dimensions");
        MPI_Bcast(&base_width, 1, MPI_INT, 0, grid.cart_comm);
        MPI_Bcast(&base_height, 1, MPI_INT, 0, grid.cart_comm);
    }

    int imgs_per_octave = scales_per_octave + 3;

    // Determine sigma values for blurring (same as serial)
    float k = pow(2, 1.0 / scales_per_octave);
    vector<float> sigma_vals{base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * pow(k, i - 1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev));
    }

    // Create scale space pyramid
    ScaleSpacePyramid pyramid = {num_octaves, imgs_per_octave, vector<vector<Image>>(num_octaves)};

    // Step 2: Process each octave - distribute tiles and generate scale space
    for (int oct = 0; oct < num_octaves; oct++) {
        // Compute tile for this octave
        TileInfo tile;
        tile.compute_for_octave(oct, base_width, base_height, grid);

        // Check if we're in rank-0-only mode
        bool is_rank0_only = tile.is_rank0_only_mode(grid);

        // Step 2a: Distribute base image tiles to all ranks
        Image local_base;
        if (is_rank0_only) {
            // Rank-0-only mode: only rank 0 has the image
            if (grid.rank == 0) {
                local_base = std::move(full_octave_bases[oct]);
            } else {
                local_base = Image(0, 0, 1);
            }
        } else {
            // Distributed mode: scatter tiles from rank 0 to all ranks
            if (grid.rank == 0) {
                local_base = Image(tile.width, tile.height, 1);
                scatter_image_tiles(full_octave_bases[oct].data, local_base.data, tile.global_width, tile.global_height,
                                    tile, grid);
            } else {
                local_base = Image(tile.width, tile.height, 1);
                scatter_image_tiles(nullptr, local_base.data, tile.global_width, tile.global_height, tile, grid);
            }
        }

        // Step 2b: Generate scale space for this octave
        if (is_rank0_only) {
            // Rank-0-only mode: sequential processing on rank 0
            if (grid.rank == 0) {
                pyramid.octaves[oct].reserve(imgs_per_octave);
                pyramid.octaves[oct].push_back(std::move(local_base));

                // Apply successive blurs sequentially
                for (int j = 1; j < sigma_vals.size(); j++) {
                    const Image& prev_img = pyramid.octaves[oct].back();
                    Image blurred = gaussian_blur(prev_img, sigma_vals[j]);
                    pyramid.octaves[oct].push_back(std::move(blurred));
                }

                // Update base for next octave (if needed)
                if (oct < num_octaves - 1) {
                    // Use the appropriate scale image and downsample it
                    const Image& next_base_src = pyramid.octaves[oct][imgs_per_octave - 3];
                    full_octave_bases[oct + 1] =
                        next_base_src.resize(next_base_src.width / 2, next_base_src.height / 2, Interpolation::NEAREST);
                }
            } else {
                // Other ranks create empty pyramids for this octave
                pyramid.octaves[oct].resize(imgs_per_octave);
                for (int j = 0; j < imgs_per_octave; j++) {
                    pyramid.octaves[oct][j] = Image(0, 0, 1);
                }
            }
        } else {
            // Normal distributed processing
            pyramid.octaves[oct].reserve(imgs_per_octave);
            pyramid.octaves[oct].push_back(std::move(local_base));

            // Apply successive blurs in parallel
            for (int j = 1; j < sigma_vals.size(); j++) {
                const Image& prev_img = pyramid.octaves[oct].back();
                Image blurred = gaussian_blur_parallel(prev_img, sigma_vals[j], tile, grid);
                pyramid.octaves[oct].push_back(std::move(blurred));
            }

            // Update base for next octave (if needed)
            if (oct < num_octaves - 1) {
                // Check if next octave will be rank-0-only
                TileInfo next_tile;
                next_tile.compute_for_octave(oct + 1, base_width, base_height, grid);
                bool next_is_rank0_only = next_tile.is_rank0_only_mode(grid);

                if (next_is_rank0_only) {
                    // Gather to rank 0 and update base for rank-0-only processing
                    const Image& next_base_src = pyramid.octaves[oct][imgs_per_octave - 3];
                    if (grid.rank == 0) {
                        Image full_img = Image(tile.global_width, tile.global_height, 1);
                        gather_image_tiles(next_base_src.data, full_img.data, tile.global_width, tile.global_height,
                                           tile, grid);
                        full_octave_bases[oct + 1] =
                            full_img.resize(full_img.width / 2, full_img.height / 2, Interpolation::NEAREST);
                    } else {
                        gather_image_tiles(next_base_src.data, nullptr, tile.global_width, tile.global_height, tile,
                                           grid);
                    }
                } else {
                    // Gather and prepare for redistribution in next iteration
                    const Image& next_base_src = pyramid.octaves[oct][imgs_per_octave - 3];
                    if (grid.rank == 0) {
                        Image full_img = Image(tile.global_width, tile.global_height, 1);
                        gather_image_tiles(next_base_src.data, full_img.data, tile.global_width, tile.global_height,
                                           tile, grid);
                        full_octave_bases[oct + 1] =
                            full_img.resize(full_img.width / 2, full_img.height / 2, Interpolation::NEAREST);
                    } else {
                        gather_image_tiles(next_base_src.data, nullptr, tile.global_width, tile.global_height, tile,
                                           grid);
                    }
                }
            }
        }
    }

    return pyramid;
}

// Parallel DoG pyramid generation
// Each rank processes its local tiles - no communication needed (embarrassingly parallel)
ScaleSpacePyramid generate_dog_pyramid_parallel(const ScaleSpacePyramid& img_pyramid) {
    PROFILE_FUNCTION();

    // Initialize DoG pyramid structure
    // DoG has one fewer image per octave than Gaussian (difference between consecutive scales)
    ScaleSpacePyramid dog_pyramid = {img_pyramid.num_octaves, img_pyramid.imgs_per_octave - 1,
                                     vector<vector<Image>>(img_pyramid.num_octaves)};

    // Process each octave
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);

        // Compute DoG for each scale in this octave
        // DoG[j] = Gaussian[j+1] - Gaussian[j]
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            // Copy the higher-scale image
            Image diff = img_pyramid.octaves[i][j];

            // Subtract the lower-scale image pixel-by-pixel
            // Parallelized with OpenMP for better performance
            #pragma omp parallel for schedule(static)
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j - 1].data[pix_idx];
            }

            dog_pyramid.octaves[i].push_back(std::move(diff));
        }
    }

    return dog_pyramid;
}

// Parallel gradient pyramid generation with halo exchange
// Each rank processes its local tiles with boundary data from neighbors
ScaleSpacePyramid generate_gradient_pyramid_parallel(const ScaleSpacePyramid& img_pyramid, const TileInfo& base_tile,
                                                     const CartesianGrid& grid) {
    PROFILE_FUNCTION();

    // Initialize gradient pyramid structure
    // Gradient pyramid has same dimensions as Gaussian pyramid
    // Each image has 2 channels: gx (gradient in x) and gy (gradient in y)
    ScaleSpacePyramid grad_pyramid = {img_pyramid.num_octaves, img_pyramid.imgs_per_octave,
                                      vector<vector<Image>>(img_pyramid.num_octaves)};

    // Process each octave
    for (int octave_idx = 0; octave_idx < grad_pyramid.num_octaves; octave_idx++) {
        const vector<Image>& img_octave = img_pyramid.octaves[octave_idx];
        vector<Image>& grad_octave = grad_pyramid.octaves[octave_idx];
        grad_octave.reserve(grad_pyramid.imgs_per_octave);

        // Compute tile info for this octave
        TileInfo tile;
        int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
        int base_height = base_tile.global_height * 2;
        tile.compute_for_octave(octave_idx, base_width, base_height, grid);

        // Check if this octave is in rank-0-only mode
        bool is_rank0_only = tile.is_rank0_only_mode(grid);

        // Create appropriate grid for this octave
        // For rank-0-only mode: use a grid with no neighbors (halo exchange becomes no-op)
        // For distributed mode: use the original grid
        CartesianGrid octave_grid = is_rank0_only ? CartesianGrid::create_rank0_only_grid(grid) : grid;

        // Non-rank-0 processes skip rank-0-only octaves
        if (is_rank0_only && octave_grid.rank != 0) {
            // Create empty gradient images for non-rank-0 processes
            for (int scale_idx = 0; scale_idx < img_pyramid.imgs_per_octave; scale_idx++) {
                grad_octave.push_back(Image(0, 0, 2));
            }
            cout << "Skipping rank-0-only octave " << octave_idx << " on rank " << grid.rank << endl;
            continue;
        }

        // Both distributed and rank-0-only modes use the same code path
        int width = tile.width;
        int height = tile.height;

        // Allocate halo buffers for 1-pixel boundary exchange
        HaloBuffers halo_buffers;
        halo_buffers.allocate(img_octave.size(), width, height, 1);

        // Exchange halos once per octave for all scales
        // For rank-0-only mode, octave_grid has no neighbors, so this becomes a no-op
        MPI_Request requests[8];
        int req_idx = 0;

        const float* top_recv = halo_buffers.top_recv.data();
        const float* bottom_recv = halo_buffers.bottom_recv.data();
        const float* left_recv = halo_buffers.left_recv.data();
        const float* right_recv = halo_buffers.right_recv.data();

        bool has_top = (octave_grid.neighbors[TOP] != MPI_PROC_NULL);
        bool has_bottom = (octave_grid.neighbors[BOTTOM] != MPI_PROC_NULL);
        bool has_left = (octave_grid.neighbors[LEFT] != MPI_PROC_NULL);
        bool has_right = (octave_grid.neighbors[RIGHT] != MPI_PROC_NULL);

        // Prepare pointers to all scale images in this octave
        const float* octave_data[8];  // Max 8 scales (generous upper bound)
        for (int s = 0; s < img_octave.size(); s++) {
            octave_data[s] = img_octave[s].data;
        }
        // octave_data[0] = img_pyramid.octaves[octave_idx][1].data;

        // Pack and exchange boundaries
        pack_boundaries(octave_data, width, height, 1, tile, halo_buffers);
        exchange_halos(octave_data, width, height, 1, tile, octave_grid, halo_buffers, requests, req_idx);

        vector<Image> grads_buffer;  // Temporary buffer for gradient images
        for (int s = 0; s < img_pyramid.imgs_per_octave; s++) {
            grads_buffer.push_back(Image(width, height, 2));
        }

        // Interior pixels
        // Parallelize gradient computation with OpenMP for better performance
        for (int scale_idx = 0; scale_idx < img_octave.size(); scale_idx++) {
            const Image& src = img_octave[scale_idx];
            Image& grad = grads_buffer[scale_idx];
            // Compute gradients with OpenMP parallelization
            #pragma omp parallel for collapse(2) schedule(static)
            for (int x = 1; x < width - 1; x++) {
                for (int y = 1; y < height - 1; y++) {
                    float gx = (src.get_pixel(x + 1, y, 0) - src.get_pixel(x - 1, y, 0)) * 0.5f;
                    float gy = (src.get_pixel(x, y + 1, 0) - src.get_pixel(x, y - 1, 0)) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    grad.set_pixel(x, y, 1, gy);
                }
            }
        }

        // Exchange halos are complete - proceed to boundary pixels
        wait_halos(requests, req_idx);

        // Boundary pixels
        for (int scale_idx = 0; scale_idx < img_octave.size(); scale_idx++) {
            const Image& src = img_octave[scale_idx];
            Image& grad = grads_buffer[scale_idx];

            const float* left_halo = &left_recv[scale_idx * height];
            const float* right_halo = &right_recv[scale_idx * height];
            const float* top_halo = &top_recv[scale_idx * width];
            const float* bottom_halo = &bottom_recv[scale_idx * width];

            // Top boundary
            if (has_top) {
                int y = 0;
                int x_start = has_left ? 0 : 1;
                int x_end = has_right ? width : width - 1;
                for (int x = x_start; x < x_end; x++) {
                    float gx, gy;
                    gx = (src.get_pixel(x + 1, y, 0) - src.get_pixel(x - 1, y, 0)) * 0.5f;
                    gy = (src.get_pixel(x, y + 1, 0) - top_halo[x]) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            // Bottom boundary
            if (has_bottom) {
                int y = height - 1;
                int x_start = has_left ? 0 : 1;
                int x_end = has_right ? width : width - 1;
                for (int x = x_start; x < x_end; x++) {
                    float gx, gy;
                    gx = (src.get_pixel(x + 1, y, 0) - src.get_pixel(x - 1, y, 0)) * 0.5f;
                    gy = (bottom_halo[x] - src.get_pixel(x, y - 1, 0)) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            // Left boundary
            if (has_left) {
                int x = 0;
                int y_start = has_top ? 0 : 1;
                int y_end = has_bottom ? height : height - 1;
                for (int y = y_start; y < y_end; y++) {
                    float gx, gy;
                    gx = (src.get_pixel(x + 1, y, 0) - left_halo[y]) * 0.5f;
                    gy = (src.get_pixel(x, y + 1, 0) - src.get_pixel(x, y - 1, 0)) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            // Right boundary
            if (has_right) {
                int x = width - 1;
                int y_start = has_top ? 0 : 1;
                int y_end = has_bottom ? height : height - 1;
                for (int y = y_start; y < y_end; y++) {
                    float gx, gy;
                    gx = (right_halo[y] - src.get_pixel(x - 1, y, 0)) * 0.5f;
                    gy = (src.get_pixel(x, y + 1, 0) - src.get_pixel(x, y - 1, 0)) * 0.5f;
                    grad.set_pixel(x, y, 0, gx);
                    grad.set_pixel(x, y, 1, gy);
                }
            }
        }

        for (int scale_idx = 0; scale_idx < img_pyramid.imgs_per_octave; scale_idx++) {
            Image& grad = grads_buffer[scale_idx];
            grad_pyramid.octaves[octave_idx].push_back(std::move(grad));
        }
        
        cout << "Done processing octave " << octave_idx << " on rank " << grid.rank << endl;
    }
    
    cout << "Done generating local gradient pyramid on rank " << grid.rank << endl;

    return grad_pyramid;
}

// Helper function to check if a point is extremum using local data and halo regions
// Returns true if the point is a local extremum in its 3x3x3 neighborhood
bool check_boundary_extremum(const vector<Image>& octave, int scale, int x, int y, const vector<float>& top_halo,
                             const vector<float>& bottom_halo, const vector<float>& left_halo,
                             const vector<float>& right_halo, int width, int height, bool has_top, bool has_bottom,
                             bool has_left, bool has_right) {
    const Image& img = octave[scale];
    const Image& prev = octave[scale - 1];
    const Image& next = octave[scale + 1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0);

    // Check 3x3x3 neighborhood
    for (int ds = -1; ds <= 1; ds++) {
        const Image& check_img = (ds == -1) ? prev : ((ds == 0) ? img : next);
        int check_scale = scale + ds;  // Actual scale index in the octave

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = x + dx;
                int ny = y + dy;
                float neighbor;

                // Determine where to read the neighbor pixel
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Within local tile
                    neighbor = check_img.get_pixel(nx, ny, 0);
                } else if (ny == -1 && has_top && nx >= 0 && nx < width) {
                    // Top halo: use check_scale as index
                    neighbor = top_halo[check_scale * width + nx];
                } else if (ny == height && has_bottom && nx >= 0 && nx < width) {
                    // Bottom halo: use check_scale as index
                    neighbor = bottom_halo[check_scale * width + nx];
                } else if (nx == -1 && has_left && ny >= 0 && ny < height) {
                    // Left halo: use check_scale as index
                    neighbor = left_halo[check_scale * height + ny];
                } else if (nx == width && has_right && ny >= 0 && ny < height) {
                    // Right halo: use check_scale as index
                    neighbor = right_halo[check_scale * height + ny];
                } else {
                    // Corner or outside boundary - clamp to edge
                    int cx = max(0, min(width - 1, nx));
                    int cy = max(0, min(height - 1, ny));
                    neighbor = check_img.get_pixel(cx, cy, 0);
                }

                if (neighbor > val) is_max = false;
                if (neighbor < val) is_min = false;
                if (!is_min && !is_max) return false;
            }
        }
    }
    return true;
}

// Parallel keypoint detection with boundary handling via halo exchange
// Each rank scans its local DoG tiles and detects keypoints
vector<Keypoint> find_keypoints_parallel(const ScaleSpacePyramid& dog_pyramid, const TileInfo& base_tile,
                                         const CartesianGrid& grid, int num_octaves, float contrast_thresh,
                                         float edge_thresh) {
    PROFILE_FUNCTION();

    vector<Keypoint> local_keypoints;

    // Process each octave
    for (int octave_idx = 0; octave_idx < dog_pyramid.num_octaves; octave_idx++) {
        const vector<Image>& octave = dog_pyramid.octaves[octave_idx];

        // Compute tile info for this octave
        TileInfo tile;
        int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
        int base_height = base_tile.global_height * 2;
        tile.compute_for_octave(octave_idx, base_width, base_height, grid);

        // Check if this octave is in rank-0-only mode
        bool is_rank0_only = tile.is_rank0_only_mode(grid);

        // Create appropriate grid for this octave
        // For rank-0-only mode: use a grid with no neighbors (halo exchange becomes no-op)
        // For distributed mode: use the original grid
        CartesianGrid octave_grid = is_rank0_only ? CartesianGrid::create_rank0_only_grid(grid) : grid;

        // Non-rank-0 processes skip rank-0-only octaves
        if (is_rank0_only && octave_grid.rank != 0) {
            continue;
        }

        int width = tile.width;
        int height = tile.height;

        // Prepare halo exchange for boundary keypoint detection
        // For rank-0-only mode, all neighbors are MPI_PROC_NULL, so this becomes a no-op
        bool has_top = (octave_grid.neighbors[TOP] != MPI_PROC_NULL);
        bool has_bottom = (octave_grid.neighbors[BOTTOM] != MPI_PROC_NULL);
        bool has_left = (octave_grid.neighbors[LEFT] != MPI_PROC_NULL);
        bool has_right = (octave_grid.neighbors[RIGHT] != MPI_PROC_NULL);

        // For each scale we process, we need 3 DoG scales (prev, curr, next)
        int num_dog_scales = dog_pyramid.imgs_per_octave;

        HaloBuffers halo_buffers;
        halo_buffers.allocate(num_dog_scales, width, height, 1);  // 1-pixel halo for 3-scale stack

        MPI_Request requests[8];
        int req_idx = 0;

        const float* octave_data[8];  // Max 8 scales (for typical SIFT settings)
        for (int s = 0; s < num_dog_scales; s++) {
            octave_data[s] = octave[s].data;
        }
        pack_boundaries(&octave_data[0], width, height, 1, tile, halo_buffers);

        // Exchange halos using MPI (once per octave)
        // For rank-0-only octaves, octave_grid has no neighbors, so this becomes a no-op
        exchange_halos(&octave_data[0], width, height, 1, tile, octave_grid, halo_buffers, requests, req_idx);
        wait_halos(requests, req_idx);

        // Now process each scale with the exchanged halos available
        // Scan scales (skip first and last scale - need neighbors for 3x3x3 check)
        const int boundary_ranges[4][2][2] = {
            {{1, 0}, {width - 2, 0}},                    // Top boundary (y=0, x in [1, width-2])
            {{1, height - 2}, {width - 2, height - 2}},  // Bottom boundary (y=height-2, x in [1, width-2])
            {{0, 1}, {0, height - 2}},                   // Left boundary (y in [1, height-2), x=0)
            {{width - 2, 1}, {width - 2, height - 2}}    // Right boundary (y in [1, height-2), x=width-2)
        };
        const int boundary_directions[4][2] = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};
        const bool boundary_exists[4] = {has_top, has_bottom, has_left, has_right};

        const int corner_points[4][2] = {
            {0, 0},                  // Top-left
            {width - 1, 0},          // Top-right
            {0, height - 1},         // Bottom-left
            {width - 1, height - 1}  // Bottom-right
        };
        const bool corner_exists[4] = {has_top && has_left, has_top && has_right, has_bottom && has_left,
                                       has_bottom && has_right};

        for (int scale = 1; scale < dog_pyramid.imgs_per_octave - 1; scale++) {
            const Image& img = octave[scale];

            // Step 1: Process interior pixels [1, width-2] x [1, height-2]
            for (int x = 1; x < width - 1; x++) {
                for (int y = 1; y < height - 1; y++) {
                    // Quick contrast threshold check
                    if (abs(img.get_pixel(x, y, 0)) < 0.8 * contrast_thresh) {
                        continue;
                    }

                    // Check if this is a local extremum (standard check)
                    if (point_is_extremum(octave, scale, x, y)) {
                        Keypoint kp = {x, y, octave_idx, scale, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid) {
                            // After refinement, kp.i and kp.j are refined discrete coords (tile-local)
                            // and kp.x and kp.y are continuous coords (tile-local, with sub-pixel offset)
                            // computed as: kp.x = MIN_PIX_DIST * 2^octave * (offset_x + kp.i)
                            // We need to convert to global coordinates by adding tile offset
                            kp.i += tile.x_start;
                            kp.j += tile.y_start;
                            float scale_factor = MIN_PIX_DIST * pow(2, octave_idx);
                            kp.x += tile.x_start * scale_factor;
                            kp.y += tile.y_start * scale_factor;
                            local_keypoints.push_back(kp);
                        }
                    }
                }
            }

            // Step 2: Process boundary pixels (edges, excluding corners)
            for (int b = 0; b < 4; b++) {
                if (!boundary_exists[b]) continue;

                int x = boundary_ranges[b][0][0];
                int y = boundary_ranges[b][0][1];
                int end_x = boundary_ranges[b][1][0];
                int end_y = boundary_ranges[b][1][1];
                int dir_x = boundary_directions[b][0];
                int dir_y = boundary_directions[b][1];

                for (; x <= end_x && y <= end_y; x += dir_x, y += dir_y) {
                    if (abs(img.get_pixel(x, y, 0)) < 0.8 * contrast_thresh) continue;
                    if (check_boundary_extremum(octave, scale, x, y, halo_buffers.top_recv, halo_buffers.bottom_recv,
                                                halo_buffers.left_recv, halo_buffers.right_recv, width, height, has_top,
                                                has_bottom, has_left, has_right)) {
                        Keypoint kp = {x, y, octave_idx, scale, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid) {
                            kp.i += tile.x_start;
                            kp.j += tile.y_start;
                            float min_pix_dist = MIN_PIX_DIST;
                            float scale_factor = min_pix_dist * pow(2, octave_idx);
                            kp.x += tile.x_start * scale_factor;
                            kp.y += tile.y_start * scale_factor;
                            local_keypoints.push_back(kp);
                        }
                    }
                }
            }

            // Step 3: Process corners
            for (int c = 0; c < 4; c++) {
                if (!corner_exists[c]) continue;

                int x = corner_points[c][0];
                int y = corner_points[c][1];

                if (abs(img.get_pixel(x, y, 0)) >= 0.8 * contrast_thresh) {
                    if (check_boundary_extremum(octave, scale, x, y, halo_buffers.top_recv, halo_buffers.bottom_recv,
                                                halo_buffers.left_recv, halo_buffers.right_recv, width, height, has_top,
                                                has_bottom, has_left, has_right)) {
                        Keypoint kp = {x, y, octave_idx, scale, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid) {
                            kp.i += tile.x_start;
                            kp.j += tile.y_start;
                            float min_pix_dist = MIN_PIX_DIST;
                            float scale_factor = min_pix_dist * pow(2, octave_idx);
                            kp.x += tile.x_start * scale_factor;
                            kp.y += tile.y_start * scale_factor;
                            local_keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
    }

    return local_keypoints;
}

// Parallel version of find_keypoints_and_descriptors
// Uses parallel Gaussian pyramid, then gathers to rank 0 for remaining serial processing
vector<Keypoint> find_keypoints_and_descriptors_parallel(const Image& img, const TileInfo& base_tile,
                                                         const CartesianGrid& grid, float sigma_min, int num_octaves,
                                                         int scales_per_octave, float contrast_thresh,
                                                         float edge_thresh, float lambda_ori, float lambda_desc) {
    // Prepare input image (convert to grayscale on rank 0 if needed)
    // On non-rank-0 processes, create an empty image as placeholder
    Image input_img;
    if (grid.rank == 0) {
        if (img.channels == 3) {
            PROFILE_SCOPE("rgb_to_grayscale");
            input_img = rgb_to_grayscale(img);
        } else {
            input_img = img;
        }
    }

    // Generate Gaussian pyramid in parallel (distributed across ranks)
    ScaleSpacePyramid local_gaussian_pyramid =
        generate_gaussian_pyramid_parallel(input_img, base_tile, grid, sigma_min, num_octaves, scales_per_octave);

    // Uncomment to save Gaussian pyramid for debugging/verification
    // save_gaussian_pyramid_parallel(local_gaussian_pyramid, base_tile, grid, "results/tmp");

    // Generate DoG pyramid in parallel (each rank processes its local tiles)
    ScaleSpacePyramid local_dog_pyramid = generate_dog_pyramid_parallel(local_gaussian_pyramid);
    // Parallel gradient pyramid generation: each rank processes its local tiles with halo exchange
    ScaleSpacePyramid local_grad_pyramid = generate_gradient_pyramid_parallel(local_gaussian_pyramid, base_tile, grid);

    // cout << "Done generating local gradient pyramid on rank " << grid.rank << endl;

    // Uncomment to save gradient pyramid for debugging/verification
    // save_gradient_pyramid_parallel(local_grad_pyramid, base_tile, grid, "results/tmp");

    // Parallel keypoint detection: each rank processes its local DoG tiles
    vector<Keypoint> local_keypoints =
        find_keypoints_parallel(local_dog_pyramid, base_tile, grid, num_octaves, contrast_thresh, edge_thresh);

    // Gather gradient pyramid to rank 0 for orientation/descriptor computation
    ScaleSpacePyramid full_grad_pyramid;
    {
        PROFILE_SCOPE("gather_gradient_pyramid");

        // Initialize the full gradient pyramid structure on rank 0
        if (grid.rank == 0) {
            full_grad_pyramid.num_octaves = local_grad_pyramid.num_octaves;
            full_grad_pyramid.imgs_per_octave = local_grad_pyramid.imgs_per_octave;
            full_grad_pyramid.octaves.resize(local_grad_pyramid.num_octaves);
        }

        // Gather each octave and scale
        for (int oct = 0; oct < local_grad_pyramid.num_octaves; oct++) {
            for (int scale = 0; scale < local_grad_pyramid.imgs_per_octave; scale++) {
                const Image& local_img = local_grad_pyramid.octaves[oct][scale];

                // Compute tile info to check if this octave was processed in rank-0-only mode
                TileInfo tile;
                int base_width = base_tile.global_width * 2;
                int base_height = base_tile.global_height * 2;
                tile.compute_for_octave(oct, base_width, base_height, grid);

                bool is_rank0_only = tile.is_rank0_only_mode(grid);

                if (is_rank0_only) {
                    // Rank 0 already has the full gradient image, no gather needed
                    if (grid.rank == 0) {
                        full_grad_pyramid.octaves[oct].push_back(Image(local_img));
                    }
                } else {
                    // Normal distributed processing: gather tiles from all ranks
                    // Prepare full image on rank 0
                    // Note: Gradient images have 2 channels (gx, gy)
                    Image full_img;
                    if (grid.rank == 0) {
                        full_img = Image(tile.global_width, tile.global_height, 2);
                    }

                    // Gather each channel separately (gradient images have 2 channels: gx, gy)
                    int local_pixels = tile.width * tile.height;
                    int global_pixels = (grid.rank == 0) ? tile.global_width * tile.global_height : 0;

                    for (int ch = 0; ch < 2; ch++) {
                        const float* local_channel_data = local_img.data + ch * local_pixels;
                        float* global_channel_data = (grid.rank == 0) ? (full_img.data + ch * global_pixels) : nullptr;

                        if (grid.rank == 0) {
                            gather_image_tiles(local_channel_data, global_channel_data, full_img.width, full_img.height,
                                               tile, grid);
                        } else {
                            gather_image_tiles(local_channel_data, nullptr, 0, 0, tile, grid);
                        }
                    }

                    if (grid.rank == 0) {
                        full_grad_pyramid.octaves[oct].push_back(std::move(full_img));
                    }
                }
            }
        }
    }

    // if (grid.rank == 0) {
    //     // Save gradient images for debugging/verification
    //     PROFILE_SCOPE("save_gradient_images");
    //     for (int oct = 0; oct < full_grad_pyramid.num_octaves; oct++) {
    //         for (int scale = 0; scale < full_grad_pyramid.imgs_per_octave; scale++) {
    //             string text_filename = "results/tmp/grad_" + to_string(oct) + "_" + to_string(scale) + ".txt";
    //             full_grad_pyramid.octaves[oct][scale].save_text(text_filename);
    //         }
    //     }
    // }

    // Gather keypoints from all ranks to rank 0
    vector<Keypoint> all_keypoints;
    {
        PROFILE_SCOPE("gather_keypoints");

        // Get count from each rank
        int local_count = local_keypoints.size();
        vector<int> counts;
        vector<int> displs;

        if (grid.rank == 0) {
            counts.resize(grid.size);
            displs.resize(grid.size);
        }

        // Gather counts
        {
            PROFILE_MPI("Gather_keypoint_counts");
            MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, grid.cart_comm);
        }

        // Compute displacements and total count on rank 0
        int total_count = 0;
        if (grid.rank == 0) {
            for (int i = 0; i < grid.size; i++) {
                displs[i] = total_count;
                total_count += counts[i];
            }
            all_keypoints.resize(total_count);

            // Convert counts and displs to bytes for MPI_Gatherv
            for (int i = 0; i < grid.size; i++) {
                counts[i] *= sizeof(Keypoint);
                displs[i] *= sizeof(Keypoint);
            }
        }

        // Gather keypoints (using MPI_BYTE to transfer raw struct data)
        // Note: Keypoint struct should be POD-compatible for MPI transfer
        {
            PROFILE_MPI("Gatherv_keypoints");
            int local_bytes = local_count * sizeof(Keypoint);
            MPI_Gatherv(local_keypoints.data(), local_bytes, MPI_BYTE, all_keypoints.data(),
                        grid.rank == 0 ? counts.data() : nullptr, grid.rank == 0 ? displs.data() : nullptr, MPI_BYTE, 0,
                        grid.cart_comm);
        }
    }

    // Export keypoints found in current stage
    // if (grid.rank == 0) {
    //     auto sorted_kps = all_keypoints;
    //     sort(sorted_kps.begin(), sorted_kps.end(), [](const Keypoint& a, const Keypoint& b) {
    //         if (a.octave != b.octave) return a.octave < b.octave;
    //         if (a.scale != b.scale) return a.scale < b.scale;
    //         if (a.x != b.x) return a.x < b.x;
    //         return a.y < b.y;
    //     });
    //     export_keypoints_discrete(sorted_kps, "results/kps.txt");
    // }

    // Continue with serial SIFT pipeline on rank 0
    vector<Keypoint> kps;
    if (grid.rank == 0) {
        // Use gathered keypoints and gradient pyramid
        vector<Keypoint> tmp_kps = all_keypoints;

        // Compute orientations and descriptors using the gathered gradient pyramid
        // Parallelized with OpenMP for multi-threaded processing
        {
            PROFILE_SCOPE("orientation_and_descriptor");
            
            // Use a thread-safe approach: each thread accumulates its own keypoints
            // Then we merge them at the end
            vector<vector<Keypoint>> thread_kps;
            
            #pragma omp parallel
            {
                int num_threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
                
                // Initialize thread-local storage on first thread only
                #pragma omp single
                {
                    thread_kps.resize(num_threads);
                }
                
                // Process keypoints with dynamic scheduling for load balancing
                #pragma omp for schedule(dynamic, 16)
                for (size_t i = 0; i < tmp_kps.size(); i++) {
                    Keypoint kp_tmp = tmp_kps[i];
                    vector<float> orientations =
                        find_keypoint_orientations(kp_tmp, full_grad_pyramid, lambda_ori, lambda_desc);
                    for (float theta : orientations) {
                        Keypoint kp = kp_tmp;
                        compute_keypoint_descriptor(kp, theta, full_grad_pyramid, lambda_desc);
                        thread_kps[thread_id].push_back(kp);
                    }
                }
            }
            
            // Merge results from all threads
            for (const auto& thread_vec : thread_kps) {
                kps.insert(kps.end(), thread_vec.begin(), thread_vec.end());
            }
        }
    }

    return kps;
}

// Helper function to gather and save Gaussian pyramid to disk
void save_gaussian_pyramid_parallel(const ScaleSpacePyramid& local_pyramid, const TileInfo& base_tile,
                                    const CartesianGrid& grid, const std::string& output_dir) {
    PROFILE_FUNCTION();

    // Create output directory on rank 0
    if (grid.rank == 0) {
        // Note: Directory should already exist, but we're not creating it here
        // to avoid including system headers. Assume it exists.
    }

    // Gather and save each octave and scale
    for (int oct = 0; oct < local_pyramid.num_octaves; oct++) {
        for (int scale = 0; scale < local_pyramid.imgs_per_octave; scale++) {
            const Image& local_img = local_pyramid.octaves[oct][scale];

            // Compute tile info to check if this octave was processed in rank-0-only mode
            TileInfo tile;
            int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
            int base_height = base_tile.global_height * 2;
            tile.compute_for_octave(oct, base_width, base_height, grid);

            bool is_rank0_only = tile.is_rank0_only_mode(grid);

            Image full_img;
            if (is_rank0_only) {
                // Rank 0 already has the full image, no gather needed
                if (grid.rank == 0) {
                    full_img = Image(local_img);
                }
            } else {
                // Normal distributed processing: gather tiles from all ranks
                if (grid.rank == 0) {
                    full_img = Image(tile.global_width, tile.global_height, 1);
                    gather_image_tiles(local_img.data, full_img.data, tile.global_width, tile.global_height, tile,
                                       grid);
                } else {
                    gather_image_tiles(local_img.data, nullptr, tile.global_width, tile.global_height, tile, grid);
                }
            }

            // Save on rank 0
            if (grid.rank == 0) {
                std::string filename = output_dir + "/" + std::to_string(oct) + "_" + std::to_string(scale) + ".txt";
                full_img.save_text(filename);
            }
        }
    }
}

// Helper function to gather and save gradient pyramid to disk
void save_gradient_pyramid_parallel(const ScaleSpacePyramid& local_pyramid, const TileInfo& base_tile,
                                    const CartesianGrid& grid, const std::string& output_dir) {
    PROFILE_FUNCTION();

    // Create output directory on rank 0
    if (grid.rank == 0) {
        // Note: Directory should already exist, but we're not creating it here
        // to avoid including system headers. Assume it exists.
    }

    // Gather and save each octave and scale
    for (int oct = 0; oct < local_pyramid.num_octaves; oct++) {
        for (int scale = 0; scale < local_pyramid.imgs_per_octave; scale++) {
            const Image& local_img = local_pyramid.octaves[oct][scale];

            // Compute tile info to check if this octave was processed in rank-0-only mode
            TileInfo tile;
            int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
            int base_height = base_tile.global_height * 2;
            tile.compute_for_octave(oct, base_width, base_height, grid);

            bool is_rank0_only = tile.is_rank0_only_mode(grid);

            Image full_img;
            if (is_rank0_only) {
                // Rank 0 already has the full gradient image, no gather needed
                if (grid.rank == 0) {
                    full_img = Image(local_img);
                }
            } else {
                // Normal distributed processing: gather tiles from all ranks
                // Note: Gradient images have 2 channels (gx, gy)
                if (grid.rank == 0) {
                    full_img = Image(tile.global_width, tile.global_height, 2);
                }

                // Gather each channel separately (gradient images have 2 channels: gx, gy)
                int local_pixels = tile.width * tile.height;
                int global_pixels = (grid.rank == 0) ? tile.global_width * tile.global_height : 0;

                for (int ch = 0; ch < 2; ch++) {
                    const float* local_channel_data = local_img.data + ch * local_pixels;
                    float* global_channel_data = (grid.rank == 0) ? (full_img.data + ch * global_pixels) : nullptr;

                    if (grid.rank == 0) {
                        gather_image_tiles(local_channel_data, global_channel_data, full_img.width, full_img.height,
                                           tile, grid);
                    } else {
                        gather_image_tiles(local_channel_data, nullptr, 0, 0, tile, grid);
                    }
                }
            }

            // Save on rank 0
            if (grid.rank == 0) {
                std::string filename =
                    output_dir + "/grad_" + std::to_string(oct) + "_" + std::to_string(scale) + ".txt";
                full_img.save_text(filename);
            }
        }
    }
}
