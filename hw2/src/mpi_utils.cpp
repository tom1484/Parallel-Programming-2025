#include "mpi_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

// CartesianGrid implementation
CartesianGrid::CartesianGrid() : cart_comm(MPI_COMM_NULL), rank(-1), size(0), comm_freed(false) {
    dims[0] = dims[1] = 0;
    coords[0] = coords[1] = 0;
    for (int i = 0; i < 4; i++) neighbors[i] = MPI_PROC_NULL;
}

CartesianGrid::~CartesianGrid() {
    // Don't free here - call finalize() explicitly before MPI_Finalize
    // This prevents the "MPI_Comm_free after MPI_FINALIZE" error
}

void CartesianGrid::finalize() {
    if (comm_freed) return;
    if (cart_comm != MPI_COMM_NULL && cart_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&cart_comm);
        cart_comm = MPI_COMM_NULL;
    }
    comm_freed = true;
}

void CartesianGrid::init(int px, int py) {
    dims[0] = px;
    dims[1] = py;
    comm_freed = false;
    int periods[2] = {0, 0};  // Non-periodic boundaries
    int reorder = 1;          // Allow MPI to reorder ranks for better topology

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    if (cart_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(cart_comm, &rank);
        MPI_Comm_size(cart_comm, &size);
        MPI_Cart_coords(cart_comm, rank, 2, coords);

        // Get neighbor ranks (MPI_PROC_NULL for boundaries)
        MPI_Cart_shift(cart_comm, 0, 1, &neighbors[LEFT], &neighbors[RIGHT]);
        MPI_Cart_shift(cart_comm, 1, 1, &neighbors[TOP], &neighbors[BOTTOM]);
    } else {
        // Fallback: use world communicator (shouldn't happen with valid dims)
        cart_comm = MPI_COMM_WORLD;
        MPI_Comm_rank(cart_comm, &rank);
        MPI_Comm_size(cart_comm, &size);
        coords[0] = coords[1] = 0;
        neighbors[TOP] = neighbors[BOTTOM] = neighbors[LEFT] = neighbors[RIGHT] = MPI_PROC_NULL;
    }
}

void CartesianGrid::get_optimal_dims(int nprocs, int& px, int& py) {
    // Try to make grid as square as possible
    px = static_cast<int>(sqrt(nprocs));
    while (nprocs % px != 0) px--;
    py = nprocs / px;

    // Prefer more rows than columns (images are typically wider)
    if (px > py) swap(px, py);
}

// TileInfo implementation
TileInfo::TileInfo()
    : x_start(0), x_end(0), y_start(0), y_end(0), width(0), height(0), global_width(0), global_height(0) {}

TileInfo::TileInfo(int gw, int gh, const CartesianGrid& grid) : global_width(gw), global_height(gh) {
    compute_for_octave(0, gw, gh, grid);
}

void TileInfo::compute_for_octave(int octave, int base_width, int base_height, const CartesianGrid& grid) {
    // Image dimensions for this octave
    global_width = base_width >> octave;  // Divide by 2^octave
    global_height = base_height >> octave;

    // Compute tile bounds based on process coordinates
    x_start = (grid.coords[0] * global_width) / grid.dims[0];
    x_end = ((grid.coords[0] + 1) * global_width) / grid.dims[0];
    y_start = (grid.coords[1] * global_height) / grid.dims[1];
    y_end = ((grid.coords[1] + 1) * global_height) / grid.dims[1];

    width = x_end - x_start;
    height = y_end - y_start;
}

bool TileInfo::is_too_small(int min_size) const { return width < min_size || height < min_size; }

// HaloBuffers implementation
void HaloBuffers::allocate(int _stack_size, int width, int height, int halo_width) {
    // For odd-sized images split across ranks, neighbors may have different tile sizes
    // Allocate conservatively - use max possible size to handle any neighbor
    // Top/bottom: use maximum possible width
    // Left/right: use maximum possible height

    // Conservative allocation: add safety margin for odd-sized tiles
    // In practice, difference is at most 1 pixel per direction
    stack_size = _stack_size;
    int max_width_variation = 2;  // Account for ±1 pixel difference
    int max_height_variation = 2;

    // Top/bottom: width elements × halo_width rows (neighbors may have slightly different width)
    top_send.resize((width + max_width_variation) * halo_width * stack_size);
    top_recv.resize((width + max_width_variation) * halo_width * stack_size);
    bottom_send.resize((width + max_width_variation) * halo_width * stack_size);
    bottom_recv.resize((width + max_width_variation) * halo_width * stack_size);

    // Left/right: halo_width elements × height rows (neighbors may have slightly different height)
    left_send.resize(halo_width * (height + max_height_variation) * stack_size);
    left_recv.resize(halo_width * (height + max_height_variation) * stack_size);
    right_send.resize(halo_width * (height + max_height_variation) * stack_size);
    right_recv.resize(halo_width * (height + max_height_variation) * stack_size);
}

void HaloBuffers::allocate(int width, int height, int halo_width) {
    allocate(1, width, height, halo_width);  // Default stack size
}

void pack_boundaries_vertical(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers) {
    int pixels = width * halo_width;
    for (int s = 0; s < buffers.stack_size; s++) {
        const float* stack_data = data[s];
        // Pack top rows
        for (int h = 0; h < halo_width; h++) {
            for (int x = 0; x < width; x++) {
                buffers.top_send[s * pixels + h * width + x] = stack_data[h * width + x];
            }
        }
        // Pack bottom rows
        for (int h = 0; h < halo_width; h++) {
            for (int x = 0; x < width; x++) {
                buffers.bottom_send[s * pixels + h * width + x] = stack_data[(height - halo_width + h) * width + x];
            }
        }
    }
}

// Pack boundary data for halo exchange
void pack_boundaries_horizontal(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers) {
    int pixels = halo_width * height;
    for (int s = 0; s < buffers.stack_size; s++) {
        const float* stack_data = data[s];
        // Pack left columns
        for (int y = 0; y < height; y++) {
            for (int h = 0; h < halo_width; h++) {
                buffers.left_send[s * pixels + y * halo_width + h] = stack_data[y * width + h];
            }
        }
        // Pack right columns
        for (int y = 0; y < height; y++) {
            for (int h = 0; h < halo_width; h++) {
                buffers.right_send[s * pixels + y * halo_width + h] = stack_data[y * width + (width - halo_width + h)];
            }
        }
    }
}

// Pack boundary data for halo exchange
void pack_boundaries(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers) {
    pack_boundaries_vertical(data, width, height, halo_width, tile, buffers);
    pack_boundaries_horizontal(data, width, height, halo_width, tile, buffers);
}

void pack_boundaries_vertical(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers) {
    pack_boundaries_vertical(&data, width, height, halo_width, tile, buffers);
}

void pack_boundaries_horizontal(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers) {
    pack_boundaries_horizontal(&data, width, height, halo_width, tile, buffers);
}

void pack_boundaries(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers) {
    pack_boundaries(&data, width, height, halo_width, tile, buffers);
}

void exchange_halos_vertical(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                             const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    // Post receives with correct sizes
    // Top/bottom: neighbor has same width as me, but might have different height
    // Left/right: neighbor has same height as me, but might have different width
    if (grid.neighbors[TOP] != MPI_PROC_NULL) {
           // Top neighbor sends me their bottom rows for ALL stacked scales
           // Receive width * halo_width elements per scale
           MPI_Irecv(buffers.top_recv.data(), width * halo_width * buffers.stack_size, MPI_FLOAT, grid.neighbors[TOP], 0,
                   grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[BOTTOM] != MPI_PROC_NULL) {
           MPI_Irecv(buffers.bottom_recv.data(), width * halo_width * buffers.stack_size, MPI_FLOAT,
                   grid.neighbors[BOTTOM], 1, grid.cart_comm, &requests[req_idx++]);
    }

    // Pack and send
    pack_boundaries_vertical(data, width, height, halo_width, tile, buffers);

    if (grid.neighbors[TOP] != MPI_PROC_NULL) {
           MPI_Isend(buffers.top_send.data(), width * halo_width * buffers.stack_size, MPI_FLOAT, grid.neighbors[TOP], 1,
                   grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[BOTTOM] != MPI_PROC_NULL) {
           MPI_Isend(buffers.bottom_send.data(), width * halo_width * buffers.stack_size, MPI_FLOAT,
                   grid.neighbors[BOTTOM], 0, grid.cart_comm, &requests[req_idx++]);
    }
}

// Perform nonblocking halo exchange
void exchange_halos_horizontal(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                               const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    // Post receives with correct sizes
    // Top/bottom: neighbor has same width as me, but might have different height
    // Left/right: neighbor has same height as me, but might have different width
    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
           // Left neighbor sends me their right columns for all stacked scales
           MPI_Irecv(buffers.left_recv.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[LEFT], 2,
                   grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
           MPI_Irecv(buffers.right_recv.data(), halo_width * height * buffers.stack_size, MPI_FLOAT,
                   grid.neighbors[RIGHT], 3, grid.cart_comm, &requests[req_idx++]);
    }

    // Pack and send
    pack_boundaries_horizontal(data, width, height, halo_width, tile, buffers);

    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
           MPI_Isend(buffers.left_send.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[LEFT], 3,
                   grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
           MPI_Isend(buffers.right_send.data(), halo_width * height * buffers.stack_size, MPI_FLOAT,
                   grid.neighbors[RIGHT], 2, grid.cart_comm, &requests[req_idx++]);
    }
}

void exchange_halos(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                    const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    exchange_halos_vertical(data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
    exchange_halos_horizontal(data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
}

void exchange_halos_vertical(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                             const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    exchange_halos_vertical(&data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
}

void exchange_halos_horizontal(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                               const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    exchange_halos_horizontal(&data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
}

void exchange_halos(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                    const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    exchange_halos(&data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
}

void wait_halos(MPI_Request* requests, int num_requests) { MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE); }

// Scatter image tiles from root to all processes
void scatter_image_tiles(const float* global_data, float* local_data, int global_width, int global_height,
                         const TileInfo& tile, const CartesianGrid& grid) {
    // For simplicity, use root-based scatter
    // More efficient: use MPI-IO with subarray views (future optimization)

    if (grid.rank == 0) {
        // Root sends tiles to all processes
        for (int p = 0; p < grid.size; p++) {
            int coords[2];
            MPI_Cart_coords(grid.cart_comm, p, 2, coords);

            // Compute tile bounds for process p
            int xs = (coords[0] * global_width) / grid.dims[0];
            int xe = ((coords[0] + 1) * global_width) / grid.dims[0];
            int ys = (coords[1] * global_height) / grid.dims[1];
            int ye = ((coords[1] + 1) * global_height) / grid.dims[1];
            int tw = xe - xs;
            int th = ye - ys;

            if (p == 0) {
                // Copy to local buffer
                for (int y = 0; y < th; y++) {
                    for (int x = 0; x < tw; x++) {
                        local_data[y * tw + x] = global_data[(ys + y) * global_width + (xs + x)];
                    }
                }
            } else {
                // Send to other process
                vector<float> tile_data(tw * th);
                for (int y = 0; y < th; y++) {
                    for (int x = 0; x < tw; x++) {
                        tile_data[y * tw + x] = global_data[(ys + y) * global_width + (xs + x)];
                    }
                }
                MPI_Send(tile_data.data(), tw * th, MPI_FLOAT, p, 0, grid.cart_comm);
            }
        }
    } else {
        // Non-root receives its tile
        MPI_Recv(local_data, tile.width * tile.height, MPI_FLOAT, 0, 0, grid.cart_comm, MPI_STATUS_IGNORE);
    }
}

// Gather image tiles from all processes to root
void gather_image_tiles(const float* local_data, float* global_data, int global_width, int global_height,
                        const TileInfo& tile, const CartesianGrid& grid) {
    if (grid.rank == 0) {
        // Root receives tiles from all processes
        for (int p = 0; p < grid.size; p++) {
            int coords[2];
            MPI_Cart_coords(grid.cart_comm, p, 2, coords);

            // Compute tile bounds for process p
            int xs = (coords[0] * global_width) / grid.dims[0];
            int xe = ((coords[0] + 1) * global_width) / grid.dims[0];
            int ys = (coords[1] * global_height) / grid.dims[1];
            int ye = ((coords[1] + 1) * global_height) / grid.dims[1];
            int tw = xe - xs;
            int th = ye - ys;

            if (p == 0) {
                // Copy from local buffer
                for (int y = 0; y < th; y++) {
                    for (int x = 0; x < tw; x++) {
                        global_data[(ys + y) * global_width + (xs + x)] = local_data[y * tw + x];
                    }
                }
            } else {
                // Receive from other process
                vector<float> tile_data(tw * th);
                MPI_Recv(tile_data.data(), tw * th, MPI_FLOAT, p, 0, grid.cart_comm, MPI_STATUS_IGNORE);
                for (int y = 0; y < th; y++) {
                    for (int x = 0; x < tw; x++) {
                        global_data[(ys + y) * global_width + (xs + x)] = tile_data[y * tw + x];
                    }
                }
            }
        }
    } else {
        // Non-root sends its tile
        MPI_Send(local_data, tile.width * tile.height, MPI_FLOAT, 0, 0, grid.cart_comm);
    }
}
