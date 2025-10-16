#ifndef IMAGE_PARALLEL_HPP
#define IMAGE_PARALLEL_HPP

#include "mpi_utils.hpp"
#include "sequential/image.hpp"

// Parallel version of Gaussian blur with MPI+OpenMP
// Uses halo exchange for boundary communication and overlaps computation with communication
Image gaussian_blur_parallel(const Image& img, float sigma, const TileInfo& tile, const CartesianGrid& grid);

#endif  // IMAGE_PARALLEL_HPP
