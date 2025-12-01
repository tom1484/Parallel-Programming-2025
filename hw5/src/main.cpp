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

void read_input(const char* filename, int& n, int& planet, int& asteroid, std::vector<double>& qx,
                std::vector<double>& qy, std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
                std::vector<double>& vz, std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step, int gravity_device_id,
                  double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
              std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz, const std::vector<double>& m,
              const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 = pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
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

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = -1.0;

    // If there is no collision even with all devices, no need to launch a missile
    if (hit_time_step >= 0) {
        // We will search over all gravity devices and find the cheapest one
        // whose destruction prevents the collision.
        double best_cost = std::numeric_limits<double>::infinity();
        int best_device = -1;

        // First, reload the initial state so we have the original types
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

        // Collect all gravity devices
        std::vector<int> devices;
        for (int i = 0; i < n; ++i) {
            if (type[i] == "device") {
                devices.push_back(i);
            }
        }

        // Try destroying each device, one at a time
        for (int device_id : devices) {
            // Reset system state for this simulation
            read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

            int missile_hit_step = -1;
            bool saved = true;

            for (int step = 0; step <= param::n_steps; ++step) {
                // advance simulation
                if (step > 0) {
                    run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
                }

                // check planet–asteroid collision
                double dx = qx[planet] - qx[asteroid];
                double dy = qy[planet] - qy[asteroid];
                double dz = qz[planet] - qz[asteroid];
                double dist2 = dx * dx + dy * dy + dz * dz;
                if (dist2 < param::planet_radius * param::planet_radius) {
                    saved = false;
                    break;  // collision still happens; this device is not enough
                }

                // missile–device interaction (only until it hits once)
                if (missile_hit_step < 0) {
                    double missile_dist = step * param::dt * param::missile_speed;
                    double pd_dist = distance(planet, device_id);

                    if (missile_dist > pd_dist) {
                        // missile hits this device at this step
                        missile_hit_step = step;
                        // after hit, the device's mass becomes zero permanently
                        m[device_id] = 0.0;
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
