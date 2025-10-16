#include "image.hpp"

#include <mpi.h>
#include <omp.h>

#include <cassert>
#include <cmath>
#include <vector>

#include "mpi_utils.hpp"
#include "profiler.hpp"
#include "sequential/image.hpp"

using namespace std;

// Parallel Gaussian blur with MPI+OpenMP and halo exchange
Image gaussian_blur_parallel(const Image& img, float sigma, const TileInfo& tile, const CartesianGrid& grid) {
    PROFILE_FUNCTION();
    assert(img.channels == 1);

    // Compute kernel
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    int halo_width = center;  // Halo width = kernel radius

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

    // Step 2: Pack and send top/bottom boundaries
    // Pack top rows
    for (int h = 0; h < halo_width; h++) {
        for (int x = 0; x < width; x++) {
            buffers.top_send[h * width + x] = img.data[h * width + x];
        }
    }
    // Pack bottom rows
    for (int h = 0; h < halo_width; h++) {
        for (int x = 0; x < width; x++) {
            buffers.bottom_send[h * width + x] = img.data[(height - halo_width + h) * width + x];
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
#pragma omp parallel for schedule(static)
    for (int y = halo_width; y < height - halo_width; y++) {
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
#pragma omp parallel for schedule(static)
    for (int y = 0; y < halo_width; y++) {
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
#pragma omp parallel for schedule(static)
    for (int y = height - halo_width; y < height; y++) {
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

    // Pack left/right columns from tmp
    for (int y = 0; y < height; y++) {
        for (int h = 0; h < halo_width; h++) {
            buffers.left_send[y * halo_width + h] = tmp.data[y * width + h];
            buffers.right_send[y * halo_width + h] = tmp.data[y * width + (width - halo_width + h)];
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
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = halo_width; x < width - halo_width; x++) {
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
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < halo_width; x++) {
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
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = width - halo_width; x < width; x++) {
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
