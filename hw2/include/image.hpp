#ifndef IMAGE_PARALLEL_HPP
#define IMAGE_PARALLEL_HPP

#include "mpi_utils.hpp"
#include "sequential/image.hpp"

// Parallel version of Gaussian blur with MPI+OpenMP
// Uses halo exchange for boundary communication and overlaps computation with communication
Image gaussian_blur_parallel(const Image& img, float sigma, const TileInfo& tile, const CartesianGrid& grid);

// Parallel version of image resize with MPI+OpenMP
// Distributes resize work across ranks, each computing its tile of the output
// src_img: Full source image (only valid on rank 0, others can pass empty image)
// new_w, new_h: Target dimensions for the resized image
// tile: Tile information for the output image
// grid: Cartesian grid for MPI communication
Image resize_parallel(const Image& src_img, int new_w, int new_h, Interpolation method,
                     const TileInfo& tile, const CartesianGrid& grid);

#endif  // IMAGE_PARALLEL_HPP
