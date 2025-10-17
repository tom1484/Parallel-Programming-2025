#include "image.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "mpi_utils.hpp"
#include "profiler.hpp"
#include "sequential/image.hpp"

using namespace std;

// Parallel Gaussian blur with MPI halo exchange (OpenMP disabled for debugging)
Image gaussian_blur_parallel(const Image& img, float sigma, const TileInfo& tile, const CartesianGrid& grid) {
    PROFILE_FUNCTION();
    assert(img.channels == 1);

    // Compute kernel
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    int halo_width = center;  // Halo width = kernel radius
    
    // Debug: Print tile info for problematic cases
    if (tile.global_width < 100 || tile.global_height < 100 || tile.width < 20 || tile.height < 20) {
        printf("[Rank %d] blur sigma=%.3f, kernel_size=%d, halo=%d, tile=%dx%d (global=%dx%d)\n",
               grid.rank, sigma, size, halo_width, tile.width, tile.height, 
               tile.global_width, tile.global_height);
    }

    vector<float> kernel(size);
    float sum = 0;
    for (int k = -size / 2; k <= size / 2; k++) {
        float val = exp(-(k * k) / (2 * sigma * sigma));
        kernel[center + k] = val;
        sum += val;
    }
    for (int k = 0; k < size; k++) kernel[k] /= sum;

    int width = tile.width;
    int height = tile.height;

    // Allocate buffers for halo exchange
    HaloBuffers buffers;
    buffers.allocate(width, height, halo_width);

    // Temporary image for intermediate result (after vertical pass)
    Image tmp(width, height, 1);

    // Output image
    Image filtered(width, height, 1);

    // ========== VERTICAL CONVOLUTION ==========

    // Step 1: Post nonblocking receives for top/bottom halos
    MPI_Request requests[8];
    int num_requests = 0;

    if (grid.neighbors[TOP] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.top_recv.data(), width * halo_width, MPI_FLOAT, grid.neighbors[TOP], 0, grid.cart_comm,
                  &requests[num_requests++]);
    }
    if (grid.neighbors[BOTTOM] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.bottom_recv.data(), width * halo_width, MPI_FLOAT, grid.neighbors[BOTTOM], 1, grid.cart_comm,
                  &requests[num_requests++]);
    }

    // Step 2: Pack and send top/bottom boundaries (clamp for small tiles)
    if (width > 0 && height > 0) {
        // Pack top rows (clamped)
        for (int h = 0; h < halo_width; h++) {
            int src_row = h < height ? h : height - 1;  // clamp to [0, height-1]
            for (int x = 0; x < width; x++) {
                buffers.top_send[h * width + x] = img.data[src_row * width + x];
            }
        }
        // Pack bottom rows (clamped)
        for (int h = 0; h < halo_width; h++) {
            int src_row = height - halo_width + h;
            if (src_row < 0) src_row = 0;
            if (src_row >= height) src_row = height - 1;
            for (int x = 0; x < width; x++) {
                buffers.bottom_send[h * width + x] = img.data[src_row * width + x];
            }
        }
    }

    if (grid.neighbors[TOP] != MPI_PROC_NULL) {
        MPI_Isend(buffers.top_send.data(), width * halo_width, MPI_FLOAT, grid.neighbors[TOP], 1, grid.cart_comm,
                  &requests[num_requests++]);
    }
    if (grid.neighbors[BOTTOM] != MPI_PROC_NULL) {
        MPI_Isend(buffers.bottom_send.data(), width * halo_width, MPI_FLOAT, grid.neighbors[BOTTOM], 0, grid.cart_comm,
                  &requests[num_requests++]);
    }

    // Step 3: Compute interior rows (overlap with communication)
    int interior_row_start = std::min(std::max(halo_width, 0), height);
    int interior_row_end = std::max(height - halo_width, interior_row_start);
    for (int y = interior_row_start; y < interior_row_end; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                int src_y = y + dy;
                // Interior rows, no clamping needed
                sum += img.data[src_y * width + x] * kernel[k];
            }
            tmp.data[y * width + x] = sum;
        }
    }

    // Step 4: Wait for halo exchange to complete
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    // Step 5: Compute border rows with received halos
    // Top border
    int top_row_end = std::min(halo_width, height);
    for (int y = 0; y < top_row_end; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                int src_y = y + dy;
                float val;
                if (src_y < 0) {
                    // Use halo from top neighbor (or clamp if at boundary)
                    if (grid.neighbors[TOP] != MPI_PROC_NULL) {
                        val = buffers.top_recv[(halo_width + src_y) * width + x];
                    } else {
                        val = img.data[x];  // Clamp to edge
                    }
                } else {
                    val = img.data[src_y * width + x];
                }
                sum += val * kernel[k];
            }
            tmp.data[y * width + x] = sum;
        }
    }

    // Bottom border
    int bottom_row_start = std::max(0, height - halo_width);
    for (int y = bottom_row_start; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                int src_y = y + dy;
                float val;
                if (src_y >= height) {
                    // Use halo from bottom neighbor (or clamp if at boundary)
                    if (grid.neighbors[BOTTOM] != MPI_PROC_NULL) {
                        val = buffers.bottom_recv[(src_y - height) * width + x];
                    } else {
                        val = img.data[(height - 1) * width + x];  // Clamp to edge
                    }
                } else {
                    val = img.data[src_y * width + x];
                }
                sum += val * kernel[k];
            }
            tmp.data[y * width + x] = sum;
        }
    }

    // ========== HORIZONTAL CONVOLUTION ==========

    // Need to exchange left/right halos from tmp
    num_requests = 0;

    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.left_recv.data(), halo_width * height, MPI_FLOAT, grid.neighbors[LEFT], 2, grid.cart_comm,
                  &requests[num_requests++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.right_recv.data(), halo_width * height, MPI_FLOAT, grid.neighbors[RIGHT], 3, grid.cart_comm,
                  &requests[num_requests++]);
    }

    // Pack left/right columns from tmp (clamp for small tiles)
    if (width > 0 && height > 0) {
        for (int y = 0; y < height; y++) {
            for (int h = 0; h < halo_width; h++) {
                int left_x = h < width ? h : width - 1;  // clamp to [0, width-1]
                int right_x = width - halo_width + h;
                if (right_x < 0) right_x = 0;
                if (right_x >= width) right_x = width - 1;
                buffers.left_send[y * halo_width + h] = tmp.data[y * width + left_x];
                buffers.right_send[y * halo_width + h] = tmp.data[y * width + right_x];
            }
        }
    }

    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.left_send.data(), halo_width * height, MPI_FLOAT, grid.neighbors[LEFT], 3, grid.cart_comm,
                  &requests[num_requests++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.right_send.data(), halo_width * height, MPI_FLOAT, grid.neighbors[RIGHT], 2, grid.cart_comm,
                  &requests[num_requests++]);
    }

    // Compute interior columns (overlap with communication)
    int interior_col_start = std::min(std::max(halo_width, 0), width);
    int interior_col_end = std::max(width - halo_width, interior_col_start);
    for (int y = 0; y < height; y++) {
        for (int x = interior_col_start; x < interior_col_end; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                int src_x = x + dx;
                sum += tmp.data[y * width + src_x] * kernel[k];
            }
            filtered.data[y * width + x] = sum;
        }
    }

    // Wait for halo exchange
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    // Compute border columns
    // Left border
    int left_col_end = std::min(halo_width, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < left_col_end; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                int src_x = x + dx;
                float val;
                if (src_x < 0) {
                    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
                        val = buffers.left_recv[y * halo_width + (halo_width + src_x)];
                    } else {
                        val = tmp.data[y * width];  // Clamp to edge
                    }
                } else {
                    val = tmp.data[y * width + src_x];
                }
                sum += val * kernel[k];
            }
            filtered.data[y * width + x] = sum;
        }
    }

    // Right border
    int right_col_start = std::max(0, width - halo_width);
    for (int y = 0; y < height; y++) {
        for (int x = right_col_start; x < width; x++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                int src_x = x + dx;
                float val;
                if (src_x >= width) {
                    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
                        val = buffers.right_recv[y * halo_width + (src_x - width)];
                    } else {
                        val = tmp.data[y * width + (width - 1)];  // Clamp to edge
                    }
                } else {
                    val = tmp.data[y * width + src_x];
                }
                sum += val * kernel[k];
            }
            filtered.data[y * width + x] = sum;
        }
    }

    return filtered;
}

// Helper function for coordinate mapping (same as sequential version)
static float map_coordinate(float new_max, float current_max, float coord) {
    float a = new_max / current_max;
    float b = -0.5 + a * 0.5;
    return a * coord + b;
}

// Parallel image resize with MPI distribution
// Strategy: Broadcast source image, each rank computes its tile of the output
// src_img: Source image (only valid on rank 0)
// new_w, new_h: Target dimensions
// tile: Tile info for OUTPUT image (must be pre-computed for new_w, new_h)
// grid: Cartesian grid
Image resize_parallel(const Image& src_img, int new_w, int new_h, Interpolation method,
                     const TileInfo& tile, const CartesianGrid& grid) {
    PROFILE_FUNCTION();
    
    // Get source dimensions - need to broadcast from rank 0
    int src_w, src_h, channels;
    if (grid.rank == 0) {
        src_w = src_img.width;
        src_h = src_img.height;
        channels = src_img.channels;
    }
    
    // Broadcast source dimensions and channels (needed by all ranks)
    {
        PROFILE_MPI("Bcast_dimensions");
        MPI_Bcast(&src_w, 1, MPI_INT, 0, grid.cart_comm);
        MPI_Bcast(&src_h, 1, MPI_INT, 0, grid.cart_comm);
        MPI_Bcast(&channels, 1, MPI_INT, 0, grid.cart_comm);
    }
    
    // Verify tile is valid
    if (tile.global_width != new_w || tile.global_height != new_h) {
        if (grid.rank == 0) {
            printf("ERROR: tile dimensions (%dx%d) don't match target (%dx%d)\n",
                   tile.global_width, tile.global_height, new_w, new_h);
        }
        MPI_Abort(grid.cart_comm, 1);
    }
    
    // Each rank allocates its output tile
    Image local_output(tile.width, tile.height, channels);
    
    // For resize, we need access to the entire source image
    // Strategy: Broadcast full source image to all ranks
    // (resize is not the bottleneck, and this is simpler than computing required regions)
    
    Image full_src;
    if (grid.rank == 0) {
        full_src = src_img;
    } else {
        full_src = Image(src_w, src_h, channels);
    }
    
    // Broadcast full source image to all ranks
    {
        PROFILE_SCOPE("broadcast_source_image");
        PROFILE_MPI("Bcast_image_data");
        MPI_Bcast(full_src.data, src_w * src_h * channels, MPI_FLOAT, 0, grid.cart_comm);
    }
    
    // Each rank computes its output tile
    {
        PROFILE_SCOPE("compute_resize");
        for (int local_x = 0; local_x < tile.width; local_x++) {
            // Convert local coordinate to global coordinate in output image
            int global_x = tile.x_start + local_x;
            
            for (int local_y = 0; local_y < tile.height; local_y++) {
                int global_y = tile.y_start + local_y;
                
                // Map global output coordinates to source coordinates
                float src_x = map_coordinate(src_w, new_w, global_x);
                float src_y = map_coordinate(src_h, new_h, global_y);
                
                for (int c = 0; c < channels; c++) {
                    float value;
                    if (method == Interpolation::BILINEAR) {
                        value = bilinear_interpolate(full_src, src_x, src_y, c);
                    } else {  // NEAREST
                        value = nn_interpolate(full_src, src_x, src_y, c);
                    }
                    local_output.set_pixel(local_x, local_y, c, value);
                }
            }
        }
    }
    
    return local_output;
}
