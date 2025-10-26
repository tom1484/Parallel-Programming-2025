// Your cuda program :)

#ifndef SUBMIT
#include <chrono>
#endif

#include <iostream>

using namespace std;

int main(int argc, char** argv) {
#ifndef SUBMIT
    auto start = chrono::high_resolution_clock::now();
#endif

#ifndef SUBMIT
    auto end = chrono::high_resolution_clock::now();
    auto elapsed_us = chrono::duration_cast<chrono::microseconds>(end - start);
    cerr << "Elapsed: " << elapsed_us.count() << " us" << endl;
#endif

    return 0;
}