#ifndef SIFT_PARALLEL_HPP
#define SIFT_PARALLEL_HPP

#include "mpi_utils.hpp"
#include "sequential/image.hpp"
#include "sequential/sift.hpp"

// Parallel version of Gaussian pyramid generation with MPI+OpenMP
// Each rank processes a tile of the image at each octave level
ScaleSpacePyramid generate_gaussian_pyramid_parallel(const Image& img, const TileInfo& base_tile,
                                                     const CartesianGrid& grid, float sigma_min = SIGMA_MIN,
                                                     int num_octaves = N_OCT, int scales_per_octave = N_SPO);

#endif  // SIFT_PARALLEL_HPP
