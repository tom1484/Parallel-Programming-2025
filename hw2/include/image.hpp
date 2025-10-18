#ifndef IMAGE_H
#define IMAGE_H
#include <mpi.h>

#include <map>
#include <string>
#include <vector>

using namespace std;

// Enable/disable profiling globally - only in DEBUG builds
#ifdef DEBUG
#define PROFILE_FUNCTION() ScopedTimer _timer(__FUNCTION__)
#define PROFILE_SCOPE(name) ScopedTimer _timer(name)
#define PROFILE_MPI(name) ScopedTimer _mpi_timer(string("MPI_") + name)
#else
#define PROFILE_FUNCTION()
#define PROFILE_SCOPE(name)
#define PROFILE_MPI(name)
#endif

// Column widths
// #define

struct TimingData {
    double total_time_ms;
    int call_count;
    int depth;
    vector<string> children;  // Call paths of children
    double min_time_ms;
    double max_time_ms;
    bool is_mpi;          // Flag to identify MPI communication sections
    string display_name;  // Simple name for display (without path)
    string parent_path;   // Full path of parent for hierarchy
};

class Profiler {
   public:
    static Profiler& getInstance();

    void startSection(const string& name);
    void endSection(const string& name);
    void report() const;
    void reset();

    // MPI/OpenMP support
    void initializeMPI(int rank, int size);
    void gatherAndReport();  // Gather from all ranks and report on rank 0

    // Get total MPI communication time
    double getTotalMPITime() const;

   private:
    Profiler() : current_depth_(0), total_program_time_ms_(0.0), mpi_rank_(0), mpi_size_(1) {}
    ~Profiler() = default;

    // Prevent copying
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    struct SectionTimer {
        string name;       // Display name
        string call_path;  // Full call path for unique identification
        chrono::high_resolution_clock::time_point start_time;
        int depth;
        int thread_id;
    };

    map<string, TimingData> timings_;  // Key is call_path, not just name
    vector<SectionTimer> timer_stack_;
    int current_depth_;
    double total_program_time_ms_;
    string root_section_;  // Call path of root section

    // MPI info
    int mpi_rank_;
    int mpi_size_;

    // Thread safety
    mutex timings_mutex_;

    // Per-thread timer stacks (for OpenMP support)
    map<int, vector<SectionTimer>> thread_timer_stacks_;
    map<int, int> thread_depths_;

    void printAggregatedSection(const string& name, const map<string, TimingData>& aggregated_timings,
                                double parent_time_ms, string prefix, bool last, double total_time) const;
};

// RAII timer class for automatic scope-based timing
class ScopedTimer {
   public:
    explicit ScopedTimer(const string& name) : name_(name) { Profiler::getInstance().startSection(name_); }

    ~ScopedTimer() { Profiler::getInstance().endSection(name_); }

   private:
    string name_;
};

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

enum Interpolation { BILINEAR, NEAREST };

struct Image {
    explicit Image(string file_path);
    Image(int w, int h, int c);
    Image();
    ~Image();
    Image(const Image& other);
    Image& operator=(const Image& other);
    Image(Image&& other);
    Image& operator=(Image&& other);
    int width;
    int height;
    int channels;
    int size;
    float* data;
    bool save(string file_path);
    bool save_png(string file_path);
    bool save_text(string file_path);
    void set_pixel(int x, int y, int c, float val);
    float get_pixel(int x, int y, int c) const;
    void clamp();
    Image resize(int new_w, int new_h, Interpolation method = BILINEAR) const;
};

float bilinear_interpolate(const Image& img, float x, float y, int c);
float nn_interpolate(const Image& img, float x, float y, int c);

Image rgb_to_grayscale(const Image& img);
Image grayscale_to_rgb(const Image& img);

Image gaussian_blur(const Image& img, float sigma);

void draw_point(Image& img, int x, int y, int size = 3);
void draw_line(Image& img, int x1, int y1, int x2, int y2);

// Parallel version of Gaussian blur with MPI+OpenMP
// Uses halo exchange for boundary communication and overlaps computation with communication
Image gaussian_blur_parallel(const Image& img, float sigma, const TileInfo& tile, const CartesianGrid& grid);

// Parallel version of image resize with MPI+OpenMP
// Distributes resize work across ranks, each computing its tile of the output
// src_img: Full source image (only valid on rank 0, others can pass empty image)
// new_w, new_h: Target dimensions for the resized image
// tile: Tile information for the output image
// grid: Cartesian grid for MPI communication
Image resize_parallel(const Image& src_img, int new_w, int new_h, Interpolation method, const TileInfo& tile,
                      const CartesianGrid& grid);

#endif