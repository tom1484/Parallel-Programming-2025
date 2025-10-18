#include "image.hpp"

#include <omp.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iomanip>

#include "stb/image_write.h"

using namespace std;

Image::Image(string file_path) {
    unsigned char* img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char* error_msg = stbi_failure_reason();
        cerr << "Failed to load image: " << file_path.c_str() << "\n";
        cerr << "Error msg (stb_image): " << error_msg << "\n";
        exit(1);
    }

    size = width * height * channels;
    data = new float[size];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y * width * channels + x * channels + c;
                int dst_idx = c * height * width + y * width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4) channels = 3;  // ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c) : width{w}, height{h}, channels{c}, size{w * h * c}, data{new float[w * h * c]()} {}

Image::Image() : width{0}, height{0}, channels{0}, size{0}, data{nullptr} {}

Image::~Image() { delete[] this->data; }

Image::Image(const Image& other)
    : width{other.width},
      height{other.height},
      channels{other.channels},
      size{other.size},
      data{new float[other.size]} {
    // cout << "copy constructor\n";
    for (int i = 0; i < size; i++) data[i] = other.data[i];
}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        delete[] data;
        // cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++) data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    : width{other.width}, height{other.height}, channels{other.channels}, size{other.size}, data{other.data} {
    // cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other) {
    // cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

// save image as jpg file
bool Image::save(string file_path) {
    unsigned char* out_data = new unsigned char[width * height * channels];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y * width * channels + x * channels + c;
                int src_idx = c * height * width + y * width + x;
                out_data[dst_idx] = roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success) cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

bool Image::save_png(string file_path) {
    unsigned char* out_data = new unsigned char[width * height * channels];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y * width * channels + x * channels + c;
                int src_idx = c * height * width + y * width + x;
                out_data[dst_idx] = roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_png(file_path.c_str(), width, height, channels, out_data, width * channels);
    if (!success) cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return success;
}

bool Image::save_text(string file_path) {
    FILE* f = fopen(file_path.c_str(), "w");
    if (!f) {
        cerr << "Failed to open file for writing: " << file_path << "\n";
        return false;
    }

    // Write 2D float data (for grayscale images)
    // Data is stored in row-major order: height x width
    for (int i = 0; i < height; i++) {
        string line;
        for (int j = 0; j < width; j++) {
            line += to_string(data[i * width + j]) + " ";
        }
        fprintf(f, "%s\n", line.c_str());
    }

    fclose(f);
    return true;
}

void Image::set_pixel(int x, int y, int c, float val) {
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        cerr << "set_pixel() error: Index out of bounds.\n";
        exit(1);
    }
    data[c * width * height + y * width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const {
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    return data[c * width * height + y * width + x];
}

void Image::clamp() {
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

// map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord) {
    float a = new_max / current_max;
    float b = -0.5 + a * 0.5;
    return a * coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const {
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c) {
    float p1, p2, p3, p4, q1, q2;
    float x_floor = floor(x), y_floor = floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil - y) * p1 + (y - y_floor) * p3;
    q2 = (y_ceil - y) * p2 + (y - y_floor) * p4;
    return (x_ceil - x) * q1 + (x - x_floor) * q2;
}

float nn_interpolate(const Image& img, float x, float y, int c) { return img.get_pixel(round(x), round(y), c); }

Image rgb_to_grayscale(const Image& img) {
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299 * red + 0.587 * green + 0.114 * blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img) {
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma) {
    PROFILE_FUNCTION();
    assert(img.channels == 1);

    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size / 2; k <= size / 2; k++) {
        float val = exp(-(k * k) / (2 * sigma * sigma));
        kernel.set_pixel(center + k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++) kernel.data[k] /= sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    // convolve vertical
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                sum += img.get_pixel(x, y + dy, 0) * kernel.data[k];
            }
            tmp.set_pixel(x, y, 0, sum);
        }
    }
    // convolve horizontal
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                sum += tmp.get_pixel(x + dx, y, 0) * kernel.data[k];
            }
            filtered.set_pixel(x, y, 0, sum);
        }
    }
    return filtered;
}

void draw_point(Image& img, int x, int y, int size) {
    for (int i = x - size / 2; i <= x + size / 2; i++) {
        for (int j = y - size / 2; j <= y + size / 2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (abs(i - x) + abs(j - y) > size / 2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2) {
    if (x2 < x1) {
        swap(x1, x2);
        swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy * (x - x1) / dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}

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

        // Get corner neighbors manually (not provided by MPI_Cart_shift)
        int corner_coords[2];
        // Top-left
        corner_coords[0] = coords[0] - 1;
        corner_coords[1] = coords[1] - 1;
        if (corner_coords[0] >= 0 && corner_coords[1] >= 0) {
            MPI_Cart_rank(cart_comm, corner_coords, &neighbors[TOPLEFT]);
        } else {
            neighbors[TOPLEFT] = MPI_PROC_NULL;
        }
        // Top-right
        corner_coords[0] = coords[0] + 1;
        corner_coords[1] = coords[1] - 1;
        if (corner_coords[0] < dims[0] && corner_coords[1] >= 0) {
            MPI_Cart_rank(cart_comm, corner_coords, &neighbors[TOPRIGHT]);
        } else {
            neighbors[TOPRIGHT] = MPI_PROC_NULL;
        }
        // Bottom-left
        corner_coords[0] = coords[0] - 1;
        corner_coords[1] = coords[1] + 1;
        if (corner_coords[0] >= 0 && corner_coords[1] < dims[1]) {
            MPI_Cart_rank(cart_comm, corner_coords, &neighbors[BOTTOMLEFT]);
        } else {
            neighbors[BOTTOMLEFT] = MPI_PROC_NULL;
        }
        // Bottom-right
        corner_coords[0] = coords[0] + 1;
        corner_coords[1] = coords[1] + 1;
        if (corner_coords[0] < dims[0] && corner_coords[1] < dims[1]) {
            MPI_Cart_rank(cart_comm, corner_coords, &neighbors[BOTTOMRIGHT]);
        } else {
            neighbors[BOTTOMRIGHT] = MPI_PROC_NULL;
        }
    } else {
        // Fallback: use world communicator (shouldn't happen with valid dims)
        cart_comm = MPI_COMM_WORLD;
        MPI_Comm_rank(cart_comm, &rank);
        MPI_Comm_size(cart_comm, &size);
        coords[0] = coords[1] = 0;
        for (int i = 0; i < 8; i++) {
            neighbors[i] = MPI_PROC_NULL;
        }
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

CartesianGrid CartesianGrid::create_rank0_only_grid(const CartesianGrid& base_grid) {
    // Create a grid configuration for rank-0-only processing
    // All ranks keep their communicator, but set neighbors to MPI_PROC_NULL
    // This allows the same halo exchange code to work (it becomes a no-op)
    CartesianGrid rank0_grid;

    rank0_grid.cart_comm = base_grid.cart_comm;  // Reuse the same communicator
    rank0_grid.rank = base_grid.rank;
    rank0_grid.size = base_grid.size;
    rank0_grid.dims[0] = base_grid.dims[0];
    rank0_grid.dims[1] = base_grid.dims[1];
    rank0_grid.coords[0] = base_grid.coords[0];
    rank0_grid.coords[1] = base_grid.coords[1];
    rank0_grid.comm_freed = false;

    // Set all neighbors to MPI_PROC_NULL so halo exchange becomes a no-op
    for (int i = 0; i < 8; i++) {
        rank0_grid.neighbors[i] = MPI_PROC_NULL;
    }

    return rank0_grid;
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

    int tile_width = x_end - x_start;
    int tile_height = y_end - y_start;

    // Check if tile is too small for distributed processing
    // If so, allocate zero-size tile (rank-0-only mode)
    // Rank 0 keeps the full image dimensions
    if (tile_width < MIN_TILE_SIZE || tile_height < MIN_TILE_SIZE) {
        if (grid.rank == 0) {
            // Rank 0 processes the entire image
            x_start = 0;
            x_end = global_width;
            y_start = 0;
            y_end = global_height;
            width = global_width;
            height = global_height;
        } else {
            // Other ranks get zero-size tiles
            x_start = 0;
            x_end = 0;
            y_start = 0;
            y_end = 0;
            width = 0;
            height = 0;
        }
    } else {
        // Normal distributed processing
        width = tile_width;
        height = tile_height;
    }
}

bool TileInfo::is_too_small(int min_size) const { return width < min_size || height < min_size; }

bool TileInfo::is_rank0_only_mode(const CartesianGrid& grid) const {
    // In rank-0-only mode:
    // - Rank 0 has the full image (width == global_width && height == global_height)
    // - Other ranks have zero-size tiles (width == 0 && height == 0)
    if (grid.rank == 0) {
        return (width == global_width && height == global_height);
    } else {
        return (width == 0 && height == 0);
    }
}

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

    // Corners: halo_width × halo_width regions (small, but needed for diagonal neighbors)
    int corner_size = halo_width * halo_width * stack_size;
    topleft_send.resize(corner_size);
    topleft_recv.resize(corner_size);
    topright_send.resize(corner_size);
    topright_recv.resize(corner_size);
    bottomleft_send.resize(corner_size);
    bottomleft_recv.resize(corner_size);
    bottomright_send.resize(corner_size);
    bottomright_recv.resize(corner_size);
}

void HaloBuffers::allocate(int width, int height, int halo_width) {
    allocate(1, width, height, halo_width);  // Default stack size
}

// Pack boundary data for halo exchange
void pack_boundaries(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers) {
    pack_boundaries_vertical(data, width, height, halo_width, tile, buffers);
    pack_boundaries_horizontal(data, width, height, halo_width, tile, buffers);
    pack_boundaries_corner(data, width, height, halo_width, tile, buffers);
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

// Pack corner regions for halo exchange (8-neighbor support)
void pack_boundaries_corner(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                            HaloBuffers& buffers) {
    int corner_pixels = halo_width * halo_width;
    for (int s = 0; s < buffers.stack_size; s++) {
        const float* stack_data = data[s];

        // Top-left corner
        for (int dy = 0; dy < halo_width; dy++) {
            for (int dx = 0; dx < halo_width; dx++) {
                buffers.topleft_send[s * corner_pixels + dy * halo_width + dx] = stack_data[dy * width + dx];
            }
        }

        // Top-right corner
        for (int dy = 0; dy < halo_width; dy++) {
            for (int dx = 0; dx < halo_width; dx++) {
                buffers.topright_send[s * corner_pixels + dy * halo_width + dx] =
                    stack_data[dy * width + (width - halo_width + dx)];
            }
        }

        // Bottom-left corner
        for (int dy = 0; dy < halo_width; dy++) {
            for (int dx = 0; dx < halo_width; dx++) {
                buffers.bottomleft_send[s * corner_pixels + dy * halo_width + dx] =
                    stack_data[(height - halo_width + dy) * width + dx];
            }
        }

        // Bottom-right corner
        for (int dy = 0; dy < halo_width; dy++) {
            for (int dx = 0; dx < halo_width; dx++) {
                buffers.bottomright_send[s * corner_pixels + dy * halo_width + dx] =
                    stack_data[(height - halo_width + dy) * width + (width - halo_width + dx)];
            }
        }
    }
}

void pack_boundaries(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                     HaloBuffers& buffers) {
    pack_boundaries(&data, width, height, halo_width, tile, buffers);
}

void pack_boundaries_vertical(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                              HaloBuffers& buffers) {
    pack_boundaries_vertical(&data, width, height, halo_width, tile, buffers);
}

void pack_boundaries_horizontal(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                                HaloBuffers& buffers) {
    pack_boundaries_horizontal(&data, width, height, halo_width, tile, buffers);
}

void pack_boundaries_corner(const float* data, int width, int height, int halo_width, const TileInfo& tile,
                            HaloBuffers& buffers) {
    pack_boundaries_corner(&data, width, height, halo_width, tile, buffers);
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
        MPI_Irecv(buffers.left_recv.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[LEFT],
                  2, grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.right_recv.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[RIGHT],
                  3, grid.cart_comm, &requests[req_idx++]);
    }

    // Pack and send
    pack_boundaries_horizontal(data, width, height, halo_width, tile, buffers);

    if (grid.neighbors[LEFT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.left_send.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[LEFT],
                  3, grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[RIGHT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.right_send.data(), halo_width * height * buffers.stack_size, MPI_FLOAT, grid.neighbors[RIGHT],
                  2, grid.cart_comm, &requests[req_idx++]);
    }
}

// Exchange corner halos (8-neighbor support)
void exchange_halos_corners(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                            const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    int corner_size = halo_width * halo_width * buffers.stack_size;

    // Top-left corner
    if (grid.neighbors[TOPLEFT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.topleft_recv.data(), corner_size, MPI_FLOAT, grid.neighbors[TOPLEFT], 7, grid.cart_comm,
                  &requests[req_idx++]);
    }
    if (grid.neighbors[BOTTOMRIGHT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.bottomright_send.data(), corner_size, MPI_FLOAT, grid.neighbors[BOTTOMRIGHT], 7,
                  grid.cart_comm, &requests[req_idx++]);
    }

    // Top-right corner
    if (grid.neighbors[TOPRIGHT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.topright_recv.data(), corner_size, MPI_FLOAT, grid.neighbors[TOPRIGHT], 8, grid.cart_comm,
                  &requests[req_idx++]);
    }
    if (grid.neighbors[BOTTOMLEFT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.bottomleft_send.data(), corner_size, MPI_FLOAT, grid.neighbors[BOTTOMLEFT], 8, grid.cart_comm,
                  &requests[req_idx++]);
    }

    // Bottom-left corner
    if (grid.neighbors[BOTTOMLEFT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.bottomleft_recv.data(), corner_size, MPI_FLOAT, grid.neighbors[BOTTOMLEFT], 9, grid.cart_comm,
                  &requests[req_idx++]);
    }
    if (grid.neighbors[TOPRIGHT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.topright_send.data(), corner_size, MPI_FLOAT, grid.neighbors[TOPRIGHT], 9, grid.cart_comm,
                  &requests[req_idx++]);
    }

    // Bottom-right corner
    if (grid.neighbors[BOTTOMRIGHT] != MPI_PROC_NULL) {
        MPI_Irecv(buffers.bottomright_recv.data(), corner_size, MPI_FLOAT, grid.neighbors[BOTTOMRIGHT], 10,
                  grid.cart_comm, &requests[req_idx++]);
    }
    if (grid.neighbors[TOPLEFT] != MPI_PROC_NULL) {
        MPI_Isend(buffers.topleft_send.data(), corner_size, MPI_FLOAT, grid.neighbors[TOPLEFT], 10, grid.cart_comm,
                  &requests[req_idx++]);
    }
}

void exchange_halos(const float** data, int width, int height, int halo_width, const TileInfo& tile,
                    const CartesianGrid& grid, HaloBuffers& buffers, MPI_Request* requests, int& req_idx) {
    exchange_halos_vertical(data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
    exchange_halos_horizontal(data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
    exchange_halos_corners(data, width, height, halo_width, tile, grid, buffers, requests, req_idx);
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
    // if (tile.global_width < 100 || tile.global_height < 100 || tile.width < 20 || tile.height < 20) {
    //     printf("[Rank %d] blur sigma=%.3f, kernel_size=%d, halo=%d, tile=%dx%d (global=%dx%d)\n",
    //            grid.rank, sigma, size, halo_width, tile.width, tile.height,
    //            tile.global_width, tile.global_height);
    // }

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
    MPI_Request requests[4];
    buffers.allocate(width, height, halo_width);

    // Temporary image for intermediate result (after vertical pass)
    Image tmp(width, height, 1);
    // Output image
    Image filtered(width, height, 1);

    // ========== VERTICAL CONVOLUTION ==========

    // // Step 1: Post nonblocking receives for top/bottom halos
    int req_idx = 0;
    exchange_halos_vertical((const float*)img.data, width, height, halo_width, tile, grid, buffers, requests, req_idx);

    // Step 3: Compute interior rows (overlap with communication)
    {
        PROFILE_SCOPE("vertical_interior");
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
    }

    // Step 4: Wait for halo exchange to complete
    // MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    {
        PROFILE_SCOPE("halo_exchange_vertical");
        wait_halos(requests, req_idx);
    }

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

    req_idx = 0;
    exchange_halos_horizontal((const float*)tmp.data, width, height, halo_width, tile, grid, buffers, requests,
                              req_idx);

    {
        PROFILE_SCOPE("horizontal_interior");
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
    }

    // Wait for halo exchange
    {
        PROFILE_SCOPE("halo_exchange_horizontal");
        wait_halos(requests, req_idx);
    }

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

// WARNING: This doesn't consider the boundary halos
// Parallel image resize with MPI distribution
// Strategy: Broadcast source image, each rank computes its tile of the output
// src_img: Source image (only valid on rank 0)
// new_w, new_h: Target dimensions
// tile: Tile info for OUTPUT image (must be pre-computed for new_w, new_h)
// grid: Cartesian grid
Image resize_parallel(const Image& src_img, int new_w, int new_h, Interpolation method, const TileInfo& tile,
                      const CartesianGrid& grid) {
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
            printf("ERROR: tile dimensions (%dx%d) don't match target (%dx%d)\n", tile.global_width, tile.global_height,
                   new_w, new_h);
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
