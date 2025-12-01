#include "kernel.hpp"

__device__ double gravity_device_mass_dev(double m0, double t) { return m0 + 0.5 * m0 * fabs(sin(t / 6000.0)); }

// ============================================================================
// Problem 1: Mega-kernel - Run ALL steps in a single kernel, track min distance
// Single block version (n <= blockSize)
// ============================================================================
__global__ void simulate_problem1_kernel(
    int n, int n_steps, int planet, int asteroid,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* m, const int* type,
    double* result_min_dist
) {
    const double dt = 60.0;
    const double eps = 1e-3;
    const double G = 6.674e-11;

    int i = threadIdx.x;

    // Shared memory for minimum distance tracking
    __shared__ double s_min_dist;

    if (i == 0) {
        s_min_dist = 1e300;
    }
    __syncthreads();

    // Check initial distance (step 0)
    if (i == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < s_min_dist) s_min_dist = dist;
    }
    __syncthreads();

    // Run simulation for all steps
    for (int step = 1; step <= n_steps; step++) {
        double t = step * dt;

        // Each thread handles one body
        if (i < n) {
            double qxi = qx[i];
            double qyi = qy[i];
            double qzi = qz[i];

            double axi = 0.0, ayi = 0.0, azi = 0.0;

            // Compute acceleration from all other bodies
            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double mj = m[j];
                // Problem 1: ignore devices (treat mass as 0)
                if (type[j] == 1) {
                    mj = 0.0;
                }

                double dx = qx[j] - qxi;
                double dy = qy[j] - qyi;
                double dz = qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }

            // Update velocity
            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            // Update position
            qx[i] = qxi + vxi * dt;
            qy[i] = qyi + vyi * dt;
            qz[i] = qzi + vzi * dt;

            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;
        }

        __syncthreads();  // All threads must finish updating before distance check

        // Thread 0 computes and updates minimum distance
        if (i == 0) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < s_min_dist) s_min_dist = dist;
        }

        __syncthreads();  // Ensure min_dist is updated before next iteration
    }

    // Write final result
    if (i == 0) {
        *result_min_dist = s_min_dist;
    }
}

double run_simulation_problem1(int n_steps, int n, int planet, int asteroid,
                               DeviceArrays& dev, double* d_result) {
    // Use single block with enough threads for all bodies
    int blockSize = ((n + 31) / 32) * 32;  // Round up to multiple of 32
    if (blockSize < 32) blockSize = 32;
    if (blockSize > 1024) blockSize = 1024;  // Max block size

    hipLaunchKernelGGL(
        simulate_problem1_kernel,
        dim3(1), dim3(blockSize), 0, 0,
        n, n_steps, planet, asteroid,
        dev.qx, dev.qy, dev.qz,
        dev.vx, dev.vy, dev.vz,
        dev.m, dev.type,
        d_result
    );

    CHECK(hipDeviceSynchronize());

    double result;
    CHECK(hipMemcpy(&result, d_result, sizeof(double), hipMemcpyDeviceToHost));
    return result;
}

// ============================================================================
// Problem 2: Mega-kernel - Run until collision detected (with early exit)
// ============================================================================
__global__ void simulate_problem2_kernel(
    int n, int n_steps, int planet, int asteroid,
    double planet_radius,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* m, const int* type,
    int* result_collision_step
) {
    const double dt = 60.0;
    const double eps = 1e-3;
    const double G = 6.674e-11;

    int i = threadIdx.x;

    // Shared memory for collision detection
    __shared__ int s_collision_step;

    if (i == 0) {
        s_collision_step = -2;  // -2 means no collision yet
    }
    __syncthreads();

    // Check initial distance (step 0)
    if (i == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < planet_radius) {
            s_collision_step = 0;
        }
    }
    __syncthreads();

    // Run simulation for all steps (with early exit on collision)
    for (int step = 1; step <= n_steps; step++) {
        // Early exit if collision already detected
        if (s_collision_step >= 0) break;

        double t = step * dt;

        // Each thread handles one body
        if (i < n) {
            double qxi = qx[i];
            double qyi = qy[i];
            double qzi = qz[i];

            double axi = 0.0, ayi = 0.0, azi = 0.0;

            // Compute acceleration from all other bodies
            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double mj = m[j];
                // Problem 2: include device mass (fluctuating)
                if (type[j] == 1) {
                    mj = gravity_device_mass_dev(mj, t);
                }

                double dx = qx[j] - qxi;
                double dy = qy[j] - qyi;
                double dz = qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }

            // Update velocity
            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            // Update position
            qx[i] = qxi + vxi * dt;
            qy[i] = qyi + vyi * dt;
            qz[i] = qzi + vzi * dt;

            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;
        }

        __syncthreads();

        // Thread 0 checks for collision
        if (i == 0) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < planet_radius && s_collision_step < 0) {
                s_collision_step = step;
            }
        }

        __syncthreads();
    }

    // Write final result
    if (i == 0) {
        *result_collision_step = s_collision_step;
    }
}

int run_simulation_problem2(int n_steps, int n, int planet, int asteroid,
                            double planet_radius, DeviceArrays& dev, int* d_result) {
    int blockSize = ((n + 31) / 32) * 32;
    if (blockSize < 32) blockSize = 32;
    if (blockSize > 1024) blockSize = 1024;

    hipLaunchKernelGGL(
        simulate_problem2_kernel,
        dim3(1), dim3(blockSize), 0, 0,
        n, n_steps, planet, asteroid, planet_radius,
        dev.qx, dev.qy, dev.qz,
        dev.vx, dev.vy, dev.vz,
        dev.m, dev.type,
        d_result
    );

    CHECK(hipDeviceSynchronize());

    int result;
    CHECK(hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    return result;
}

// ============================================================================
// Problem 3: Mega-kernel - Run with missile logic
// ============================================================================
__global__ void simulate_problem3_kernel(
    int n, int n_steps, int planet, int asteroid, int device_id,
    double planet_radius, double missile_speed, double dt_param,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* m, const int* type,
    int* result  // result[0] = saved (0/1), result[1] = missile_hit_step
) {
    const double dt = 60.0;
    const double eps = 1e-3;
    const double G = 6.674e-11;

    int i = threadIdx.x;

    __shared__ int s_missile_hit_step;
    __shared__ int s_collision_step;

    if (i == 0) {
        s_missile_hit_step = -1;
        s_collision_step = -1;
    }
    __syncthreads();

    // Check initial conditions (step 0)
    if (i == 0) {
        // Check collision
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < planet_radius) {
            s_collision_step = 0;
        }

        // Check missile (at step 0, missile_dist = 0)
        // No hit at step 0
    }
    __syncthreads();

    for (int step = 1; step <= n_steps; step++) {
        // Early exit if collision detected
        if (s_collision_step >= 0) break;

        double t = step * dt;
        int missile_hit = s_missile_hit_step;

        if (i < n) {
            double qxi = qx[i];
            double qyi = qy[i];
            double qzi = qz[i];

            double axi = 0.0, ayi = 0.0, azi = 0.0;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double mj = m[j];

                // If missile hit the device, its mass becomes 0
                if (j == device_id && missile_hit >= 0) {
                    mj = 0.0;
                } else if (type[j] == 1) {
                    mj = gravity_device_mass_dev(mj, t);
                }

                double dx = qx[j] - qxi;
                double dy = qy[j] - qyi;
                double dz = qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }

            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            qx[i] = qxi + vxi * dt;
            qy[i] = qyi + vyi * dt;
            qz[i] = qzi + vzi * dt;

            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;
        }

        __syncthreads();

        if (i == 0) {
            // Check collision
            double dx_pa = qx[planet] - qx[asteroid];
            double dy_pa = qy[planet] - qy[asteroid];
            double dz_pa = qz[planet] - qz[asteroid];
            double dist_pa = sqrt(dx_pa * dx_pa + dy_pa * dy_pa + dz_pa * dz_pa);
            if (dist_pa < planet_radius && s_collision_step < 0) {
                s_collision_step = step;
            }

            // Check missile hit (only if not already hit)
            if (s_missile_hit_step < 0) {
                double missile_dist = step * dt_param * missile_speed;
                double dx_pd = qx[planet] - qx[device_id];
                double dy_pd = qy[planet] - qy[device_id];
                double dz_pd = qz[planet] - qz[device_id];
                double dist_pd = sqrt(dx_pd * dx_pd + dy_pd * dy_pd + dz_pd * dz_pd);
                if (missile_dist > dist_pd) {
                    s_missile_hit_step = step;
                }
            }
        }

        __syncthreads();
    }

    if (i == 0) {
        result[0] = (s_collision_step < 0) ? 1 : 0;  // saved = no collision
        result[1] = s_missile_hit_step;
    }
}

Problem3Result run_simulation_problem3(int n_steps, int n, int planet, int asteroid,
                                       int device_id, double planet_radius,
                                       double missile_speed, double dt,
                                       DeviceArrays& dev, int* d_result) {
    int blockSize = ((n + 31) / 32) * 32;
    if (blockSize < 32) blockSize = 32;
    if (blockSize > 1024) blockSize = 1024;

    hipLaunchKernelGGL(
        simulate_problem3_kernel,
        dim3(1), dim3(blockSize), 0, 0,
        n, n_steps, planet, asteroid, device_id,
        planet_radius, missile_speed, dt,
        dev.qx, dev.qy, dev.qz,
        dev.vx, dev.vy, dev.vz,
        dev.m, dev.type,
        d_result
    );

    CHECK(hipDeviceSynchronize());

    int results[2];
    CHECK(hipMemcpy(results, d_result, 2 * sizeof(int), hipMemcpyDeviceToHost));

    Problem3Result res;
    res.saved = (results[0] == 1);
    res.missile_hit_step = results[1];
    res.collision_step = res.saved ? -1 : 0;  // We don't track exact collision step
    return res;
}
