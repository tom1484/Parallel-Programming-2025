#ifndef SIFT_H
#define SIFT_H

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"

using namespace std;

struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    vector<vector<Image>> octaves;
};

struct Keypoint {
    // discrete coordinates
    int i;
    int j;
    int octave;
    int scale;  // index of gaussian image inside the octave

    // continuous coordinates (interpolated)
    float x;
    float y;
    float sigma;
    float extremum_val;  // value of interpolated DoG extremum

    array<uint8_t, 128> descriptor;
};

//*******************************************
// SIFT algorithm parameters, used by default
//*******************************************

// digital scale space configuration and keypoint detection
const int MAX_REFINEMENT_ITERS = 5;
const float SIGMA_MIN = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = 8;
const int N_SPO = 5;
const float C_DOG = 0.015;
const float C_EDGE = 10;

// computation of the SIFT descriptor
const int N_BINS = 36;
const float LAMBDA_ORI = 1.5;
const int N_HIST = 4;
const int N_ORI = 8;
const float LAMBDA_DESC = 6;

// feature matching
const float THRESH_ABSOLUTE = 350;
const float THRESH_RELATIVE = 0.7;

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min = SIGMA_MIN, int num_octaves = N_OCT,
                                            int scales_per_octave = N_SPO);

ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid);

vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh = C_DOG,
                                float edge_thresh = C_EDGE);

// Helper functions for keypoint detection (exposed for parallel implementation)
bool point_is_extremum(const vector<Image>& octave, int scale, int x, int y);

bool refine_or_discard_keypoint(Keypoint& kp, const vector<Image>& octave, float contrast_thresh, float edge_thresh);

ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid);

vector<float> find_keypoint_orientations(Keypoint& kp, const ScaleSpacePyramid& grad_pyramid,
                                         float lambda_ori = LAMBDA_ORI, float lambda_desc = LAMBDA_DESC);

void compute_keypoint_descriptor(Keypoint& kp, float theta, const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc = LAMBDA_DESC);

vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min = SIGMA_MIN, int num_octaves = N_OCT,
                                                int scales_per_octave = N_SPO, float contrast_thresh = C_DOG,
                                                float edge_thresh = C_EDGE, float lambda_ori = LAMBDA_ORI,
                                                float lambda_desc = LAMBDA_DESC);

Image draw_keypoints(const Image& img, const vector<Keypoint>& kps);

void export_keypoints_discrete(const vector<Keypoint>& kps, const string& file_path);

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

#endif