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

    // Copy initial state to GPU
    copy_host_to_device(dev1, n, qx, qy, qz, vx, vy, vz, m, type_int);

    // Simulation loop (GPU)
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step_gpu(step, n, dev1,
                         /*ignore_devices=*/true,
                         /*disabled_device=*/-1);
        }

        // Copy positions back to host
        copy_device_to_host(dev1, n, qx, qy, qz, vx, vy, vz);

        double dist = distance_host(planet, asteroid, qx, qy, qz);
        if (dist < min_dist) min_dist = dist;
    }

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

    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step_gpu(step, n, dev2,
                         /*ignore_devices=*/false,
                         /*disabled_device=*/-1);
        }

        // Retrieve updated planet & asteroid positions
        copy_device_to_host(dev2, n, qx, qy, qz, vx, vy, vz);

        double dist = distance_host(planet, asteroid, qx, qy, qz);

        if (dist < param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // ---------------------
    // Problem 3 (GPU version, simple full-copy)
    // ---------------------
    int gravity_device_id = -1;
    double missile_cost = -1.0;

    // If there is no collision even with all devices (Problem 2),
    // there's no need to launch a missile.
    if (hit_time_step >= 0) {
        // First, read input again to identify all devices
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

        // Allocate GPU arrays for Problem 3 simulations
        DeviceArrays dev3;
        allocate_device_arrays(dev3, n);

        double best_cost = std::numeric_limits<double>::infinity();
        int best_device = -1;

        // Try destroying each device one by one
        for (int device_id : devices) {
            // Reset system state from original input
            read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
            set_type_int(n, type, type_int);

            // Copy this fresh state to GPU
            copy_host_to_device(dev3, n, qx, qy, qz, vx, vy, vz, m, type_int);

            int missile_hit_step = -1;
            bool saved = true;

            for (int step = 0; step <= param::n_steps; ++step) {
                if (step > 0) {
                    // If missile already hit, disable this device's mass via disabled_device
                    int disabled = (missile_hit_step >= 0 ? device_id : -1);

                    run_step_gpu(step, n, dev3,
                                 /*ignore_devices=*/false,
                                 /*disabled_device=*/disabled);
                }

                // Get full state back to CPU (simple version)
                copy_device_to_host(dev3, n, qx, qy, qz, vx, vy, vz);

                // Check planet–asteroid collision
                double dist_pa = distance_host(planet, asteroid, qx, qy, qz);
                if (dist_pa < param::planet_radius) {
                    saved = false;
                    break;  // this device cannot save the planet
                }

                // Missile–device logic (only until it hits once)
                if (missile_hit_step < 0) {
                    double missile_dist = step * param::dt * param::missile_speed;
                    double dist_pd = distance_host(planet, device_id, qx, qy, qz);

                    if (missile_dist > dist_pd) {
                        // missile hits this device at this step
                        missile_hit_step = step;
                        // from next step onward, we pass disabled_device = device_id
                        // to run_step_gpu so its mass is treated as zero
                    }
                }
            }

            // If the planet is safe and the missile actually hit the device, compute cost
            if (saved && missile_hit_step >= 0) {
                double t_hit = missile_hit_step * param::dt;
                double cost = param::get_missile_cost(t_hit);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_device = device_id;
                }
            }
        }

        if (best_device != -1) {
            gravity_device_id = best_device;
            missile_cost = best_cost;
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
