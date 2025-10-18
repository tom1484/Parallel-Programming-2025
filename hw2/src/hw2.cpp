#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "image.hpp"
#include "sift.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        cerr << "MPI_THREAD_FUNNELED not supported\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    if (rank == 0) {
        cerr << "Running with " << size << " MPI ranks and " << omp_get_max_threads() << " OpenMP threads per rank\n";
    }
#endif

    // Create Cartesian grid
    CartesianGrid grid;
    int px, py;
    CartesianGrid::get_optimal_dims(size, px, py);
    grid.init(px, py);

#ifdef DEBUG
    if (grid.rank == 0) {
        cerr << "Using " << px << " x " << py << " process grid\n";
    }
#endif

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc != 4) {
        if (rank == 0) {
            cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    string input_img = argv[1];
    string output_img = argv[2];
    string output_txt = argv[3];

    // Load image on rank 0
    Image img;
    int img_width = 0, img_height = 0;

    if (grid.rank == 0) {
        img = Image(input_img);
        {
            PROFILE_SCOPE("rgb_to_grayscale");
            img = img.channels == 1 ? img : rgb_to_grayscale(img);
        }
        img_width = img.width;
        img_height = img.height;
    }

    // Broadcast image dimensions
    MPI_Bcast(&img_width, 1, MPI_INT, 0, grid.cart_comm);
    MPI_Bcast(&img_height, 1, MPI_INT, 0, grid.cart_comm);

    // Create base tile info
    TileInfo base_tile(img_width, img_height, grid);

    vector<Keypoint> kps;
    {
        PROFILE_SCOPE("SIFT_TOTAL");
        // Use parallel version that leverages parallel Gaussian pyramid generation
        kps = find_keypoints_and_descriptors_parallel(img, base_tile, grid);
    }

    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
    if (grid.rank == 0) {
        ofstream ofs(output_txt);
        if (!ofs) {
            cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
    }
    /////////////////////////////////////////////////////////////

    // Free the Cartesian communicator before MPI_Finalize
    grid.finalize();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}