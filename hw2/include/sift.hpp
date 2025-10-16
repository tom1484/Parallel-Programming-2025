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

// Parallel version of find_keypoints_and_descriptors
// Currently uses parallel Gaussian pyramid generation, then gathers to rank 0 for serial processing
std::vector<Keypoint> find_keypoints_and_descriptors_parallel(const Image& img, const TileInfo& base_tile,
                                                              const CartesianGrid& grid, float sigma_min = SIGMA_MIN,
                                                              int num_octaves = N_OCT, int scales_per_octave = N_SPO,
                                                              float contrast_thresh = C_DOG, float edge_thresh = C_EDGE,
                                                              float lambda_ori = LAMBDA_ORI,
                                                              float lambda_desc = LAMBDA_DESC);

#endif  // SIFT_PARALLEL_HPP
