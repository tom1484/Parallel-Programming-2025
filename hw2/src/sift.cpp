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

        // Skip if tile is too small
        if (tile.is_too_small(16)) {
            // For very small octaves, could switch to serial or subset of ranks
            // For now, just skip
            break;
        }

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
