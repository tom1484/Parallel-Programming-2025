#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP

#include <mpi.h>

#include <vector>

using namespace std;

// Global minimum tile size for distributed processing
// Tiles smaller than this will be allocated as zero-size (rank-0-only mode)
constexpr int MIN_TILE_SIZE = 20;

// Neighbor indices for Cartesian grid (8-connected)
enum Neighbor { TOP = 0, BOTTOM = 1, LEFT = 2, RIGHT = 3, TOPLEFT = 4, TOPRIGHT = 5, BOTTOMLEFT = 6, BOTTOMRIGHT = 7 };

// 2D Cartesian MPI process grid
struct CartesianGrid {
    MPI_Comm cart_comm;  // Cartesian communicator
    int rank;            // Rank in cart_comm
    int size;            // Total number of processes
    int dims[2];         // Grid dimensions [Px, Py]
    int coords[2];       // This process's coordinates [px, py]
    int neighbors[8];    // Neighbor ranks: TOP, BOTTOM, LEFT, RIGHT, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT
    bool comm_freed;     // Track if communicator has been freed

    CartesianGrid();
    ~CartesianGrid();

    // Initialize Cartesian grid with specified dimensions
    void init(int px, int py);

    // Manually free the communicator (call before MPI_Finalize)
    void finalize();

    // Get optimal grid dimensions for given number of processes
    static void get_optimal_dims(int nprocs, int& px, int& py);

    // Create a rank-0-only grid (for small octaves where only rank 0 processes)
    // This grid has all neighbors set to MPI_PROC_NULL, making halo exchange a no-op
    static CartesianGrid create_rank0_only_grid(const CartesianGrid& base_grid);
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

    // Check if this is rank-0-only mode (zero-size tiles for non-rank-0, full image for rank-0)
    bool is_rank0_only_mode(const CartesianGrid& grid) const;
};

// Buffer for halo exchange (8-neighbor support)
struct HaloBuffers {
    int stack_size;
    // Edge buffers (4 cardinal directions)
    vector<float> top_send, top_recv;
    vector<float> bottom_send, bottom_recv;
    vector<float> left_send, left_recv;
    vector<float> right_send, right_recv;
    // Corner buffers (4 diagonal directions)
    vector<float> topleft_send, topleft_recv;
    vector<float> topright_send, topright_recv;
    vector<float> bottomleft_send, bottomleft_recv;
    vector<float> bottomright_send, bottomright_recv;

    void allocate(int stack_size, int width, int height, int halo_width);
    void allocate(int width, int height, int halo_width);
};

// Pack boundary data for sending
void pack_boundaries(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers);
void pack_boundaries_corner(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                            HaloBuffers& buffers);
void pack_boundaries_vertical(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers);
void pack_boundaries_horizontal(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers);
void pack_boundaries(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers);
void pack_boundaries_corner(const float* data, int width, int height, int halo_width, const TileInfo& tile,
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
