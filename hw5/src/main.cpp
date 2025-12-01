#ifndef SUBMIT
#include <chrono>
#include <iostream>
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel.hpp"
#include "utils.hpp"

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) { return m0 + 0.5 * m0 * fabs(sin(t / 6000)); }
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

hipError_t _hip_error;

void set_type_int(int n, const std::vector<std::string>& type, std::vector<int>& type_int) {
    type_int.resize(n);
    for (int i = 0; i < n; ++i) type_int[i] = (type[i] == "device" ? 1 : 0);
}

void allocate_device_arrays(DeviceArrays& dev, int n) {
    CHECK(hipMalloc(&dev.qx, n * sizeof(double)));
    CHECK(hipMalloc(&dev.qy, n * sizeof(double)));
    CHECK(hipMalloc(&dev.qz, n * sizeof(double)));

    CHECK(hipMalloc(&dev.vx, n * sizeof(double)));
    CHECK(hipMalloc(&dev.vy, n * sizeof(double)));
    CHECK(hipMalloc(&dev.vz, n * sizeof(double)));

    CHECK(hipMalloc(&dev.m, n * sizeof(double)));
    CHECK(hipMalloc(&dev.type, n * sizeof(int)));
}

void copy_host_to_device(DeviceArrays& dev, int n, const std::vector<double>& qx, const std::vector<double>& qy,
                         const std::vector<double>& qz, const std::vector<double>& vx, const std::vector<double>& vy,
                         const std::vector<double>& vz, const std::vector<double>& m,
                         const std::vector<int>& type_int) {
    CHECK(hipMemcpy(dev.qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));

    CHECK(hipMemcpy(dev.vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));

    CHECK(hipMemcpy(dev.m, m.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.type, type_int.data(), n * sizeof(int), hipMemcpyHostToDevice));
}

void copy_device_to_host(DeviceArrays& dev, int n, std::vector<double>& qx, std::vector<double>& qy,
                         std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
                         std::vector<double>& vz) {
    CHECK(hipMemcpy(qx.data(), dev.qx, n * sizeof(double), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(qy.data(), dev.qy, n * sizeof(double), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(qz.data(), dev.qz, n * sizeof(double), hipMemcpyDeviceToHost));

    CHECK(hipMemcpy(vx.data(), dev.vx, n * sizeof(double), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(vy.data(), dev.vy, n * sizeof(double), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(vz.data(), dev.vz, n * sizeof(double), hipMemcpyDeviceToHost));
}

double distance_host(int i, int j, const std::vector<double>& qx, const std::vector<double>& qy,
                     const std::vector<double>& qz) {
    double dx = qx[i] - qx[j];
    double dy = qy[i] - qy[j];
    double dz = qz[i] - qz[j];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

int main(int argc, char** argv) {
#ifndef SUBMIT
    auto __start = std::chrono::high_resolution_clock::now();
#endif
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    std::vector<int> type_int;  // add this

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // ---------------------
    // Problem 1 (GPU version)
    // ---------------------
    double min_dist = std::numeric_limits<double>::infinity();

    // Load input
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    set_type_int(n, type, type_int);

    // Allocate GPU arrays
    DeviceArrays dev1;
    allocate_device_arrays(dev1, n);

    // Allocate device memory for results (large enough for various result types)
    void* d_result;
    CHECK(hipMalloc(&d_result, 2 * sizeof(double)));  // Enough for 2 doubles or 4 ints

    // Copy initial state to GPU
    copy_host_to_device(dev1, n, qx, qy, qz, vx, vy, vz, m, type_int);

    // Run entire simulation on GPU in a single kernel launch
    min_dist = run_simulation_problem1(param::n_steps, n, planet, asteroid, dev1, (double*)d_result);

    // ---------------------
    // Problem 2 (GPU version)
    // ---------------------
    int hit_time_step = -2;

    // Reload input (fresh initial conditions)
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    set_type_int(n, type, type_int);

    // Reconvert type strings → integers
    for (int i = 0; i < n; i++) type_int[i] = (type[i] == "device" ? 1 : 0);

    // Allocate GPU arrays
    DeviceArrays dev2;
    allocate_device_arrays(dev2, n);

    // Copy initial state to GPU
    copy_host_to_device(dev2, n, qx, qy, qz, vx, vy, vz, m, type_int);

    // Run entire simulation on GPU in a single kernel launch
    hit_time_step = run_simulation_problem2(param::n_steps, n, planet, asteroid,
                                            param::planet_radius, dev2, (int*)d_result);

    // ---------------------
    // Problem 3 (Multi-GPU optimized)
    // ---------------------
    int gravity_device_id = -1;
    double missile_cost = -1.0;

    // If there is no collision even with all devices (Problem 2),
    // there's no need to launch a missile.
    if (hit_time_step >= 0) {
        // Read input once to get device list
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        type_int.resize(n);
        for (int i = 0; i < n; ++i) type_int[i] = (type[i] == "device" ? 1 : 0);

        // Collect all gravity devices
        std::vector<int> devices;
        for (int i = 0; i < n; ++i) {
            if (type[i] == "device") {
                devices.push_back(i);
            }
        }

        // Use multi-GPU implementation
        int best_device_idx = -1;
        int best_hit_step = -1;

        run_problem3_multi_gpu(
            param::n_steps, n, planet, asteroid,
            devices.data(), (int)devices.size(),
            param::planet_radius, param::missile_speed, param::dt,
            qx.data(), qy.data(), qz.data(),
            vx.data(), vy.data(), vz.data(),
            m.data(), type_int.data(),
            &best_device_idx, &best_hit_step
        );

        if (best_device_idx >= 0 && best_hit_step >= 0) {
            gravity_device_id = devices[best_device_idx];
            double t_hit = best_hit_step * param::dt;
            missile_cost = param::get_missile_cost(t_hit);
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

#ifndef SUBMIT
    auto __end = std::chrono::high_resolution_clock::now();
    auto __elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(__end - __start);
    std::cerr << "Elapsed: " << __elapsed_us.count() << " us" << std::endl;
#endif

    return 0;
}
