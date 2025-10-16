#include "sift.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
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

    // For now, we'll resize on rank 0 and scatter
    // TODO: Parallelize resize operation
    Image base_img;
    if (grid.rank == 0) {
        base_img = img.resize(img.width * 2, img.height * 2, Interpolation::BILINEAR);
    }

    // Broadcast dimensions
    int base_width, base_height;
    if (grid.rank == 0) {
        base_width = base_img.width;
        base_height = base_img.height;
    }
    MPI_Bcast(&base_width, 1, MPI_INT, 0, grid.cart_comm);
    MPI_Bcast(&base_height, 1, MPI_INT, 0, grid.cart_comm);

    // Compute tile for this octave
    TileInfo tile;
    tile.compute_for_octave(0, base_width, base_height, grid);

    // Scatter base image to all processes
    Image local_base(tile.width, tile.height, 1);
    if (grid.rank == 0) {
        scatter_image_tiles(base_img.data, local_base.data, base_width, base_height, tile, grid);
    } else {
        scatter_image_tiles(nullptr, local_base.data, base_width, base_height, tile, grid);
    }

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

    for (int i = 0; i < num_octaves; i++) {
        // Update tile for this octave
        tile.compute_for_octave(i, base_width, base_height, grid);

        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(move(local_base));

        // Apply successive blurs
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            Image blurred = gaussian_blur_parallel(prev_img, sigma_vals[j], tile, grid);
            pyramid.octaves[i].push_back(move(blurred));
        }

        // Prepare base image for next octave (downsample by 2x)
        const Image& next_base_src = pyramid.octaves[i][imgs_per_octave - 3];
        int next_width = next_base_src.width / 2;
        int next_height = next_base_src.height / 2;

        // Simple nearest-neighbor downsample (local operation, no communication)
        local_base = Image(next_width, next_height, 1);
#pragma omp parallel for collapse(2)
        for (int y = 0; y < next_height; y++) {
            for (int x = 0; x < next_width; x++) {
                local_base.data[y * next_width + x] = next_base_src.data[(y * 2) * next_base_src.width + (x * 2)];
            }
        }
    }

    return pyramid;
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

    // Gather the pyramid to rank 0 for serial processing
    // For now, we'll gather each octave/scale separately
    ScaleSpacePyramid full_gaussian_pyramid;

    {
        PROFILE_SCOPE("gather_pyramid");
        // All ranks should have computed the same number of octaves now
        // Initialize the full pyramid structure on rank 0
        if (grid.rank == 0) {
            full_gaussian_pyramid.num_octaves = local_gaussian_pyramid.num_octaves;
            full_gaussian_pyramid.imgs_per_octave = local_gaussian_pyramid.imgs_per_octave;
            full_gaussian_pyramid.octaves.resize(local_gaussian_pyramid.num_octaves);
        }

        for (int oct = 0; oct < local_gaussian_pyramid.num_octaves; oct++) {
            for (int scale = 0; scale < local_gaussian_pyramid.imgs_per_octave; scale++) {
                // Get local image (should exist if we computed this many octaves)
                const Image& local_img = local_gaussian_pyramid.octaves[oct][scale];

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
    }

    // Continue with serial SIFT pipeline on rank 0
    vector<Keypoint> kps;
    if (grid.rank == 0) {
        // Generate DoG pyramid
        ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(full_gaussian_pyramid);

        // Find keypoints
        vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);

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
