#include "sift.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

    // Broadcast input image dimensions (needed by all ranks to compute tile)
    int input_width, input_height;
    if (grid.rank == 0) {
        input_width = img.width;
        input_height = img.height;
    }
    {
        PROFILE_MPI("Bcast_input_dimensions");
        MPI_Bcast(&input_width, 1, MPI_INT, 0, grid.cart_comm);
        MPI_Bcast(&input_height, 1, MPI_INT, 0, grid.cart_comm);
    }

    // Parallel resize: each rank computes its tile of the 2x upscaled image
    int base_width = input_width * 2;
    int base_height = input_height * 2;

    // Compute tile for base octave (octave 0)
    TileInfo tile;
    tile.compute_for_octave(0, base_width, base_height, grid);

    // Each rank computes its tile of the upscaled image
    Image local_base = resize_parallel(img, base_width, base_height, Interpolation::BILINEAR, tile, grid);

    // Apply initial blur
    float sigma_diff = sqrt(base_sigma * base_sigma - 1.0f);
    local_base = gaussian_blur_parallel(local_base, sigma_diff, tile, grid);

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

    const int min_tile_size = 20;          // Need at least 20 pixels to handle max halo ~10
    int first_small_octave = num_octaves;  // Index of first octave that's too small

    // Process octaves in parallel until tiles become too small
    for (int i = 0; i < num_octaves; i++) {
        // Update tile for this octave
        tile.compute_for_octave(i, base_width, base_height, grid);

        // Check if tiles are large enough for distributed processing
        bool tile_large_enough = (tile.width >= min_tile_size && tile.height >= min_tile_size);

        // All ranks need to agree on processing mode (all-reduce)
        int local_ok = tile_large_enough ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, grid.cart_comm);

        if (global_ok == 0) {
            // Tiles too small - stop distributed processing
            first_small_octave = i;
            if (grid.rank == 0) {
                printf("Octave %d and beyond: tiles too small (min=%dx%d), switching to rank-0-only mode\n", i,
                       tile.width, tile.height);
            }
            break;
        }

        // Normal distributed processing
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(move(local_base));

        // Apply successive blurs
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            Image blurred = gaussian_blur_parallel(prev_img, sigma_vals[j], tile, grid);
            pyramid.octaves[i].push_back(move(blurred));
        }

        // Prepare base image for next octave (downsample by 2x)
        if (i < num_octaves - 1) {
            const Image& next_base_src = pyramid.octaves[i][imgs_per_octave - 3];

            PROFILE_SCOPE("parallel_downsample");

            // Get current and next octave dimensions
            int current_width = tile.global_width;
            int current_height = tile.global_height;
            int next_width = current_width / 2;
            int next_height = current_height / 2;

            // Compute tile for next octave
            TileInfo next_tile;
            next_tile.compute_for_octave(i + 1, base_width, base_height, grid);

            // Strategy: Gather current image to rank 0, then use resize_parallel
            // This is simpler than trying to downsample from distributed tiles directly
            Image full_img;
            if (grid.rank == 0) {
                full_img = Image(current_width, current_height, 1);
                gather_image_tiles(next_base_src.data, full_img.data, current_width, current_height, tile, grid);
            } else {
                // Non-root ranks participate in gather
                gather_image_tiles(next_base_src.data, nullptr, current_width, current_height, tile, grid);
            }

            // Parallel downsample: each rank computes its tile of the downsampled image
            local_base = resize_parallel(full_img, next_width, next_height, Interpolation::NEAREST, next_tile, grid);

            // Update tile for next iteration
            tile = next_tile;
        }
    }

    // Process remaining small octaves on rank 0 only
    if (first_small_octave < num_octaves) {
        // Need to gather tiles from last distributed octave and prepare base for first small octave
        if (first_small_octave > 0) {
            // Get the image from previous octave to use as base
            int prev_octave = first_small_octave - 1;
            const Image& next_base_src = pyramid.octaves[prev_octave][imgs_per_octave - 3];

            // Update tile info for the previous octave to gather correctly
            tile.compute_for_octave(prev_octave, base_width, base_height, grid);

            // Gather the full image and downsample on rank 0
            Image full_img;
            if (grid.rank == 0) {
                full_img = Image(tile.global_width, tile.global_height, 1);
                gather_image_tiles(next_base_src.data, full_img.data, tile.global_width, tile.global_height, tile,
                                   grid);

                // Downsample to get base for first small octave
                local_base = full_img.resize(tile.global_width / 2, tile.global_height / 2, Interpolation::NEAREST);
            } else {
                gather_image_tiles(next_base_src.data, nullptr, tile.global_width, tile.global_height, tile, grid);
            }
        }

        // Rank 0 processes remaining octaves sequentially
        if (grid.rank == 0) {
            for (int i = first_small_octave; i < num_octaves; i++) {
                pyramid.octaves[i].reserve(imgs_per_octave);
                pyramid.octaves[i].push_back(move(local_base));

                // Apply successive blurs sequentially
                for (int j = 1; j < sigma_vals.size(); j++) {
                    const Image& prev_img = pyramid.octaves[i].back();
                    Image blurred = gaussian_blur(prev_img, sigma_vals[j]);
                    pyramid.octaves[i].push_back(move(blurred));
                }

                // Prepare base for next octave (sequential downsample)
                if (i < num_octaves - 1) {
                    const Image& next_base_src = pyramid.octaves[i][imgs_per_octave - 3];
                    local_base =
                        next_base_src.resize(next_base_src.width / 2, next_base_src.height / 2, Interpolation::NEAREST);
                }
            }
        } else {
            // Other ranks create empty pyramids for these octaves
            for (int i = first_small_octave; i < num_octaves; i++) {
                pyramid.octaves[i].resize(imgs_per_octave);
                for (int j = 0; j < imgs_per_octave; j++) {
                    pyramid.octaves[i][j] = Image(0, 0, 1);
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
            // OpenMP is currently disabled, so this runs single-threaded per rank
            // Future: Add #pragma omp parallel for when re-enabling OpenMP
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j - 1].data[pix_idx];
            }

            dog_pyramid.octaves[i].push_back(move(diff));
        }
    }

    return dog_pyramid;
}

// Parallel keypoint detection
// Each rank scans its local DoG tiles and detects keypoints
vector<Keypoint> find_keypoints_parallel(const ScaleSpacePyramid& dog_pyramid, const TileInfo& base_tile,
                                         const CartesianGrid& grid, int num_octaves, float contrast_thresh,
                                         float edge_thresh) {
    PROFILE_FUNCTION();

    vector<Keypoint> local_keypoints;

    // Process each octave
    for (int octave_idx = 0; octave_idx < dog_pyramid.num_octaves; octave_idx++) {
        const vector<Image>& octave = dog_pyramid.octaves[octave_idx];

        // Check if this octave is empty (rank-0-only mode for small octaves)
        if (octave.empty() || octave[0].width == 0 || octave[0].height == 0) {
            continue;  // Skip empty octaves
        }

        // Compute tile info for this octave
        TileInfo tile;
        int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
        int base_height = base_tile.global_height * 2;
        tile.compute_for_octave(octave_idx, base_width, base_height, grid);

        // Define interior region (1 pixel inside tile boundary)
        // This ensures we own the keypoint and have all neighbors for 3x3x3 check
        int x_min = 1;               // 1 pixel from tile edge
        int x_max = tile.width - 1;  // 1 pixel from tile edge
        int y_min = 1;
        int y_max = tile.height - 1;

        // Scan scales (skip first and last scale - need neighbors for 3x3x3 check)
        for (int scale = 1; scale < dog_pyramid.imgs_per_octave - 1; scale++) {
            const Image& img = octave[scale];

            // Scan interior pixels in this tile
            // OpenMP disabled for now - will add later
            for (int x = x_min; x < x_max; x++) {
                for (int y = y_min; y < y_max; y++) {
                    // Quick contrast threshold check before expensive extremum test
                    if (abs(img.get_pixel(x, y, 0)) < 0.8 * contrast_thresh) {
                        continue;
                    }

                    // Check if this is a local extremum in 3x3x3 neighborhood
                    if (point_is_extremum(octave, scale, x, y)) {
                        // Create keypoint at discrete location
                        // Use local tile coordinates (x, y) which are relative to tile origin
                        Keypoint kp = {x, y, octave_idx, scale, -1, -1, -1, -1};

                        // Refine keypoint (Taylor expansion, contrast check, edge check)
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);

                        if (kp_is_valid) {
                            // Keypoint coordinates (kp.x, kp.y, kp.sigma) are now in global image space
                            // (computed by find_input_img_coords inside refine_or_discard_keypoint)
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

    // Generate DoG pyramid in parallel (each rank processes its local tiles)
    ScaleSpacePyramid local_dog_pyramid = generate_dog_pyramid_parallel(local_gaussian_pyramid);

    // Gather the pyramids to rank 0 for remaining serial processing
    // For now, we gather both Gaussian and DoG pyramids
    ScaleSpacePyramid full_gaussian_pyramid;
    ScaleSpacePyramid full_dog_pyramid;

    {
        PROFILE_SCOPE("gather_pyramids");
        // All ranks should have computed the same number of octaves now
        // Initialize the full pyramid structures on rank 0
        if (grid.rank == 0) {
            full_gaussian_pyramid.num_octaves = local_gaussian_pyramid.num_octaves;
            full_gaussian_pyramid.imgs_per_octave = local_gaussian_pyramid.imgs_per_octave;
            full_gaussian_pyramid.octaves.resize(local_gaussian_pyramid.num_octaves);

            full_dog_pyramid.num_octaves = local_dog_pyramid.num_octaves;
            full_dog_pyramid.imgs_per_octave = local_dog_pyramid.imgs_per_octave;
            full_dog_pyramid.octaves.resize(local_dog_pyramid.num_octaves);
        }

        for (int oct = 0; oct < local_gaussian_pyramid.num_octaves; oct++) {
            // Gather Gaussian pyramid for this octave
            for (int scale = 0; scale < local_gaussian_pyramid.imgs_per_octave; scale++) {
                // Get local image (should exist if we computed this many octaves)
                const Image& local_img = local_gaussian_pyramid.octaves[oct][scale];

                // Check if this octave was processed in rank-0-only mode
                // (indicated by 0×0 empty images on any rank)
                // All ranks must agree on this to avoid deadlock
                int local_size = local_img.width * local_img.height;
                int min_size;
                MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, grid.cart_comm);
                bool is_rank0_only = (min_size == 0);

                if (is_rank0_only) {
                    // For rank-0-only octaves, rank 0 already has the complete image
                    // No gathering needed; just copy from rank 0's local pyramid
                    if (grid.rank == 0) {
                        // Rank 0 already has the full image in local_gaussian_pyramid
                        // Just copy it to the full pyramid
                        full_gaussian_pyramid.octaves[oct].push_back(Image(local_img));
                    }
                    // Non-rank-0 processes do nothing for this octave
                } else {
                    // Normal distributed processing: gather tiles from all ranks
                    // Compute tile info for this octave
                    TileInfo tile;
                    int base_width = base_tile.global_width * 2;  // Account for initial 2x upscaling
                    int base_height = base_tile.global_height * 2;
                    tile.compute_for_octave(oct, base_width, base_height, grid);

                    // Prepare full image on rank 0
                    Image full_img;
                    if (grid.rank == 0) {
                        int octave_width = base_width / (1 << oct);
                        int octave_height = base_height / (1 << oct);
                        full_img = Image(octave_width, octave_height, 1);
                    }

                    // Gather tiles - ALL ranks must participate
                    if (grid.rank == 0) {
                        gather_image_tiles(local_img.data, full_img.data, full_img.width, full_img.height, tile, grid);
                        full_gaussian_pyramid.octaves[oct].push_back(move(full_img));
                    } else {
                        gather_image_tiles(local_img.data, nullptr, 0, 0, tile, grid);
                    }
                }
            }

            // Gather DoG pyramid for this octave
            // DoG has one fewer image per octave than Gaussian
            for (int scale = 0; scale < local_dog_pyramid.imgs_per_octave; scale++) {
                const Image& local_img = local_dog_pyramid.octaves[oct][scale];

                // Check if this octave was processed in rank-0-only mode
                int local_size = local_img.width * local_img.height;
                int min_size;
                MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, grid.cart_comm);
                bool is_rank0_only = (min_size == 0);

                if (is_rank0_only) {
                    if (grid.rank == 0) {
                        full_dog_pyramid.octaves[oct].push_back(Image(local_img));
                    }
                } else {
                    // Compute tile info for this octave
                    TileInfo tile;
                    int base_width = base_tile.global_width * 2;
                    int base_height = base_tile.global_height * 2;
                    tile.compute_for_octave(oct, base_width, base_height, grid);

                    // Prepare full image on rank 0
                    Image full_img;
                    if (grid.rank == 0) {
                        int octave_width = base_width / (1 << oct);
                        int octave_height = base_height / (1 << oct);
                        full_img = Image(octave_width, octave_height, 1);
                    }

                    // Gather tiles - ALL ranks must participate
                    if (grid.rank == 0) {
                        gather_image_tiles(local_img.data, full_img.data, full_img.width, full_img.height, tile, grid);
                        full_dog_pyramid.octaves[oct].push_back(move(full_img));
                    } else {
                        gather_image_tiles(local_img.data, nullptr, 0, 0, tile, grid);
                    }
                }
            }
        }
    }

    // Save pyramid images for debugging/verification (only on rank 0)
    // if (grid.rank == 0) {
    //     PROFILE_SCOPE("save_pyramid_images");
    //     for (int oct = 0; oct < full_gaussian_pyramid.num_octaves; oct++) {
    //         for (int scale = 0; scale < full_gaussian_pyramid.imgs_per_octave; scale++) {
    //             string filename = "results/tmp/" + to_string(oct) + "_" + to_string(scale) + ".txt";
    //             full_gaussian_pyramid.octaves[oct][scale].save_text(filename);
    //         }
    //     }
    // }

    // Save DoG images for debugging/verification (only on rank 0)
    // if (grid.rank == 0) {
    //     PROFILE_SCOPE("save_dog_images");
    //     for (int oct = 0; oct < full_dog_pyramid.num_octaves; oct++) {
    //         for (int scale = 0; scale < full_dog_pyramid.imgs_per_octave; scale++) {
    //             string filename = "results/tmp/dog_" + to_string(oct) + "_" + to_string(scale) + ".txt";
    //             full_dog_pyramid.octaves[oct][scale].save_text(filename);
    //         }
    //     }
    // }

    // Parallel keypoint detection: each rank processes its local DoG tiles
    vector<Keypoint> local_keypoints =
        find_keypoints_parallel(local_dog_pyramid, base_tile, grid, num_octaves, contrast_thresh, edge_thresh);

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

    // Continue with serial SIFT pipeline on rank 0
    vector<Keypoint> kps;
    if (grid.rank == 0) {
        // Use gathered keypoints directly
        vector<Keypoint> tmp_kps = all_keypoints;

        // Generate gradient pyramid
        ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(full_gaussian_pyramid);

        // Compute orientations and descriptors
        {
            PROFILE_SCOPE("orientation_and_descriptor");
            for (Keypoint& kp_tmp : tmp_kps) {
                vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid, lambda_ori, lambda_desc);
                for (float theta : orientations) {
                    Keypoint kp = kp_tmp;
                    compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                    kps.push_back(kp);
                }
            }
        }
    }

    return kps;
}
