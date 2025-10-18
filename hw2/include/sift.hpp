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

// Parallel version of DoG pyramid generation
// Each rank processes its local tiles from the Gaussian pyramid
// This is embarrassingly parallel - no communication needed
ScaleSpacePyramid generate_dog_pyramid_parallel(const ScaleSpacePyramid& img_pyramid);

// Parallel version of gradient pyramid generation
// Each rank processes its local tiles from the Gaussian pyramid
// Computes gradients using central differences with halo exchange for boundary pixels
ScaleSpacePyramid generate_gradient_pyramid_parallel(const ScaleSpacePyramid& img_pyramid, const TileInfo& base_tile,
                                                     const CartesianGrid& grid);

// Parallel version of keypoint detection
// Each rank scans its local DoG tiles for extrema and refines them
// Uses interior ownership rule: keeps keypoints 1 pixel inside tile boundary
std::vector<Keypoint> find_keypoints_parallel(const ScaleSpacePyramid& dog_pyramid, const TileInfo& base_tile,
                                              const CartesianGrid& grid, int num_octaves, float contrast_thresh = C_DOG,
                                              float edge_thresh = C_EDGE);

// Parallel version of find_keypoints_and_descriptors
// Currently uses parallel Gaussian pyramid generation, then gathers to rank 0 for serial processing
std::vector<Keypoint> find_keypoints_and_descriptors_parallel(const Image& img, const TileInfo& base_tile,
                                                              const CartesianGrid& grid, float sigma_min = SIGMA_MIN,
                                                              int num_octaves = N_OCT, int scales_per_octave = N_SPO,
                                                              float contrast_thresh = C_DOG, float edge_thresh = C_EDGE,
                                                              float lambda_ori = LAMBDA_ORI,
                                                              float lambda_desc = LAMBDA_DESC);

// Helper function to gather and save Gaussian pyramid to disk
// Gathers distributed pyramid tiles to rank 0 and saves each scale to ./results/tmp/<octave>_<scale>.txt
void save_gaussian_pyramid_parallel(const ScaleSpacePyramid& local_pyramid, const TileInfo& base_tile,
                                    const CartesianGrid& grid, const std::string& output_dir = "results/tmp");

// Helper function to gather and save gradient pyramid to disk
// Gathers distributed gradient tiles to rank 0 and saves each scale to ./results/tmp/grad_<octave>_<scale>.txt
void save_gradient_pyramid_parallel(const ScaleSpacePyramid& local_pyramid, const TileInfo& base_tile,
                                    const CartesianGrid& grid, const std::string& output_dir = "results/tmp");

#endif  // SIFT_PARALLEL_HPP
