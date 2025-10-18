#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include <mpi.h>

#include <vector>

using namespace std;

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
    int stack_size;
    vector<float> top_send, top_recv;
    vector<float> bottom_send, bottom_recv;
    vector<float> left_send, left_recv;
    vector<float> right_send, right_recv;

    void allocate(int stack_size, int width, int height, int halo_width);
    void allocate(int width, int height, int halo_width);
};

// Pack boundary data for sending
void pack_boundaries(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers);
void pack_boundaries_vertical(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers);
void pack_boundaries_horizontal(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers);
void pack_boundaries(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers);
void pack_boundaries_vertical(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers);
void pack_boundaries_horizontal(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers);

// Perform nonblocking halo exchange
void exchange_halos(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                    const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);
void exchange_halos_vertical(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                             const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);
void exchange_halos_horizontal(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                               const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);
void exchange_halos(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                    const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);
void exchange_halos_vertical(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                             const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);
void exchange_halos_horizontal(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                               const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx);

// Wait for halo exchange to complete
void wait_halos(MPI_Request* requests, int num_requests);

// Scatter image tiles from root to all processes
void scatter_image_tiles(const float* global_data, float* local_data, int global_width, int global_height,
                         const TileInfo& tile, const CartesianGrid& grid);

// Gather image tiles from all processes to root
void gather_image_tiles(const float* local_data, float* global_data, int global_width, int global_height,
                        const TileInfo& tile, const CartesianGrid& grid);

#endif  // MPI_UTILS_HPP
