#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include <mpi.h>

#include <vector>

// Neighbor indices for Cartesian grid
enum Neighbor { TOP = 0, BOTTOM = 1, LEFT = 2, RIGHT = 3 };

// 2D Cartesian MPI process grid
struct CartesianGrid {
    MPI_Comm cart_comm;  // Cartesian communicator
    int rank;            // Rank in cart_comm
    int size;            // Total number of processes
    int dims[2];         // Grid dimensions [Px, Py]
    int coords[2];       // This process's coordinates [px, py]
    int neighbors[4];    // Neighbor ranks: TOP, BOTTOM, LEFT, RIGHT
    bool comm_freed;     // Track if communicator has been freed

    CartesianGrid();
    ~CartesianGrid();

    // Initialize Cartesian grid with specified dimensions
    void init(int px, int py);

    // Manually free the communicator (call before MPI_Finalize)
    void finalize();

    // Get optimal grid dimensions for given number of processes
    static void get_optimal_dims(int nprocs, int& px, int& py);
};

// Information about a tile owned by a process
struct TileInfo {
    int x_start, x_end;  // Local tile x bounds [x_start, x_end)
    int y_start, y_end;  // Local tile y bounds [y_start, y_end)
    int width, height;   // Tile dimensions
    int global_width;    // Global image width
    int global_height;   // Global image height

    TileInfo();
    TileInfo(int gw, int gh, const CartesianGrid& grid);

    // Compute tile bounds for a specific octave (images shrink by 2x per octave)
    void compute_for_octave(int octave, int base_width, int base_height, const CartesianGrid& grid);

    // Check if tile is too small to be useful
    bool is_too_small(int min_size = 32) const;
};

// Buffer for halo exchange
struct HaloBuffers {
    std::vector<float> top_send, top_recv;
    std::vector<float> bottom_send, bottom_recv;
    std::vector<float> left_send, left_recv;
    std::vector<float> right_send, right_recv;

    void allocate(int width, int height, int halo_width);
};

// Pack boundary data for sending
void pack_boundaries(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers);

// Unpack received halo data
void unpack_boundaries(float* data, int width, int height, int halo_width, const TileInfo& tile,
                       const HaloBuffers& buffers);

// Perform nonblocking halo exchange
void exchange_halos(float* data, int width, int height, int halo_width, const TileInfo& tile, const CartesianGrid& grid,
                    HaloBuffers& buffers, MPI_Request* requests);

// Wait for halo exchange to complete
void wait_halos(MPI_Request* requests, int num_requests);

// Scatter image tiles from root to all processes
void scatter_image_tiles(const float* global_data, float* local_data, int global_width, int global_height,
                         const TileInfo& tile, const CartesianGrid& grid);

// Gather image tiles from all processes to root
void gather_image_tiles(const float* local_data, float* global_data, int global_width, int global_height,
                        const TileInfo& tile, const CartesianGrid& grid);

#endif  // MPI_UTILS_HPP
