#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "image.hpp"
#include "profiler.hpp"
#include "sift.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc != 4) {
        cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        return 1;
    }

    string input_img = argv[1];
    string output_img = argv[2];
    string output_txt = argv[3];

    auto start = chrono::high_resolution_clock::now();

    Image img(input_img);
    {
        PROFILE_SCOPE("rgb_to_grayscale");
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }

    vector<Keypoint> kps;
    {
        PROFILE_SCOPE("SIFT_TOTAL");
        kps = find_keypoints_and_descriptors(img);
    }

    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
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
    /////////////////////////////////////////////////////////////

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "Execution time: " << duration.count() << " ms\n";

    cout << "Found " << kps.size() << " keypoints.\n";

    // Print profiling report
    Profiler::getInstance().report();

    return 0;
}