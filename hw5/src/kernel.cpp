#include "kernel.hpp"

__device__ double gravity_device_mass_dev(double m0, double t) { return m0 + 0.5 * m0 * fabs(sin(t / 6000.0)); }

// Maximum bodies we can fit in shared memory
// Each body needs: 3 doubles (position) + 1 double (mass) = 32 bytes
// Plus type (4 bytes) = 36 bytes per body
// For 1024 bodies: 36KB shared memory (most GPUs have 48-64KB)
#define MAX_BODIES_SHARED 1024

// ============================================================================
// Problem 1: Mega-kernel with SHARED MEMORY
// Load all body data into shared memory once per step
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
    int blockSize = blockDim.x;

    // Shared memory for body data (loaded once per step)
    __shared__ double s_qx[MAX_BODIES_SHARED];
    __shared__ double s_qy[MAX_BODIES_SHARED];
    __shared__ double s_qz[MAX_BODIES_SHARED];
    __shared__ double s_m[MAX_BODIES_SHARED];
    __shared__ int s_type[MAX_BODIES_SHARED];
    __shared__ double s_min_dist;

    // Initialize min_dist
    if (i == 0) {
        s_min_dist = 1e300;
    }

    // Load initial positions and masses into shared memory
    if (i < n) {
        s_qx[i] = qx[i];
        s_qy[i] = qy[i];
        s_qz[i] = qz[i];
        s_m[i] = m[i];
        s_type[i] = type[i];
    }
    __syncthreads();

    // Check initial distance (step 0)
    if (i == 0) {
        double dx = s_qx[planet] - s_qx[asteroid];
        double dy = s_qy[planet] - s_qy[asteroid];
        double dz = s_qz[planet] - s_qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < s_min_dist) s_min_dist = dist;
    }
    __syncthreads();

    // Run simulation for all steps
    for (int step = 1; step <= n_steps; step++) {
        double axi = 0.0, ayi = 0.0, azi = 0.0;

        // Each thread computes acceleration for its body using shared memory
        if (i < n) {
            double qxi = s_qx[i];
            double qyi = s_qy[i];
            double qzi = s_qz[i];

            // Compute acceleration from all other bodies (all in shared memory!)
            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                // Problem 1: ignore devices (treat mass as 0)
                double mj = (s_type[j] == 1) ? 0.0 : s_m[j];

                double dx = s_qx[j] - qxi;
                double dy = s_qy[j] - qyi;
                double dz = s_qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }
        }

        __syncthreads();  // Ensure all threads finished reading shared memory

        // Update velocities and positions (write to global, then update shared)
        if (i < n) {
            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            double new_qx = s_qx[i] + vxi * dt;
            double new_qy = s_qy[i] + vyi * dt;
            double new_qz = s_qz[i] + vzi * dt;

            // Update global memory (for final state)
            qx[i] = new_qx;
            qy[i] = new_qy;
            qz[i] = new_qz;
            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;

            // Update shared memory for next iteration
            s_qx[i] = new_qx;
            s_qy[i] = new_qy;
            s_qz[i] = new_qz;
        }

        __syncthreads();  // Ensure all shared memory updated before next iteration

        // Thread 0 computes minimum distance (from shared memory)
        if (i == 0) {
            double dx = s_qx[planet] - s_qx[asteroid];
            double dy = s_qy[planet] - s_qy[asteroid];
            double dz = s_qz[planet] - s_qz[asteroid];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < s_min_dist) s_min_dist = dist;
        }
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
// Problem 2: Mega-kernel with SHARED MEMORY - Run until collision detected
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

    // Shared memory for body data
    __shared__ double s_qx[MAX_BODIES_SHARED];
    __shared__ double s_qy[MAX_BODIES_SHARED];
    __shared__ double s_qz[MAX_BODIES_SHARED];
    __shared__ double s_m[MAX_BODIES_SHARED];
    __shared__ int s_type[MAX_BODIES_SHARED];
    __shared__ int s_collision_step;

    // Initialize collision step
    if (i == 0) {
        s_collision_step = -2;
    }

    // Load initial data into shared memory
    if (i < n) {
        s_qx[i] = qx[i];
        s_qy[i] = qy[i];
        s_qz[i] = qz[i];
        s_m[i] = m[i];
        s_type[i] = type[i];
    }
    __syncthreads();

    // Check initial distance (step 0)
    if (i == 0) {
        double dx = s_qx[planet] - s_qx[asteroid];
        double dy = s_qy[planet] - s_qy[asteroid];
        double dz = s_qz[planet] - s_qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < planet_radius) {
            s_collision_step = 0;
        }
    }
    __syncthreads();

    // Run simulation
    for (int step = 1; step <= n_steps; step++) {
        if (s_collision_step >= 0) break;

        double t = step * dt;
        double axi = 0.0, ayi = 0.0, azi = 0.0;

        if (i < n) {
            double qxi = s_qx[i];
            double qyi = s_qy[i];
            double qzi = s_qz[i];

            // Compute acceleration using shared memory
            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                // Problem 2: include device mass (fluctuating)
                double mj = s_m[j];
                if (s_type[j] == 1) {
                    mj = gravity_device_mass_dev(mj, t);
                }

                double dx = s_qx[j] - qxi;
                double dy = s_qy[j] - qyi;
                double dz = s_qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }
        }

        __syncthreads();

        // Update velocities and positions
        if (i < n) {
            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            double new_qx = s_qx[i] + vxi * dt;
            double new_qy = s_qy[i] + vyi * dt;
            double new_qz = s_qz[i] + vzi * dt;

            qx[i] = new_qx;
            qy[i] = new_qy;
            qz[i] = new_qz;
            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;

            s_qx[i] = new_qx;
            s_qy[i] = new_qy;
            s_qz[i] = new_qz;
        }

        __syncthreads();

        // Check collision
        if (i == 0) {
            double dx = s_qx[planet] - s_qx[asteroid];
            double dy = s_qy[planet] - s_qy[asteroid];
            double dz = s_qz[planet] - s_qz[asteroid];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < planet_radius && s_collision_step < 0) {
                s_collision_step = step;
            }
        }

        __syncthreads();
    }

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
// Problem 3: Mega-kernel with SHARED MEMORY - Run with missile logic
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

    // Shared memory for body data
    __shared__ double s_qx[MAX_BODIES_SHARED];
    __shared__ double s_qy[MAX_BODIES_SHARED];
    __shared__ double s_qz[MAX_BODIES_SHARED];
    __shared__ double s_m[MAX_BODIES_SHARED];
    __shared__ int s_type[MAX_BODIES_SHARED];
    __shared__ int s_missile_hit_step;
    __shared__ int s_collision_step;

    // Initialize
    if (i == 0) {
        s_missile_hit_step = -1;
        s_collision_step = -1;
    }

    // Load initial data into shared memory
    if (i < n) {
        s_qx[i] = qx[i];
        s_qy[i] = qy[i];
        s_qz[i] = qz[i];
        s_m[i] = m[i];
        s_type[i] = type[i];
    }
    __syncthreads();

    // Check initial conditions (step 0)
    if (i == 0) {
        double dx = s_qx[planet] - s_qx[asteroid];
        double dy = s_qy[planet] - s_qy[asteroid];
        double dz = s_qz[planet] - s_qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < planet_radius) {
            s_collision_step = 0;
        }
    }
    __syncthreads();

    for (int step = 1; step <= n_steps; step++) {
        if (s_collision_step >= 0) break;

        double t = step * dt;
        int missile_hit = s_missile_hit_step;
        double axi = 0.0, ayi = 0.0, azi = 0.0;

        if (i < n) {
            double qxi = s_qx[i];
            double qyi = s_qy[i];
            double qzi = s_qz[i];

            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double mj = s_m[j];

                // If missile hit the device, its mass becomes 0
                if (j == device_id && missile_hit >= 0) {
                    mj = 0.0;
                } else if (s_type[j] == 1) {
                    mj = gravity_device_mass_dev(mj, t);
                }

                double dx = s_qx[j] - qxi;
                double dy = s_qy[j] - qyi;
                double dz = s_qz[j] - qzi;

                double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
                double dist3 = dist2 * sqrt(dist2);

                double coef = G * mj / dist3;
                axi += coef * dx;
                ayi += coef * dy;
                azi += coef * dz;
            }
        }

        __syncthreads();

        if (i < n) {
            double vxi = vx[i] + axi * dt;
            double vyi = vy[i] + ayi * dt;
            double vzi = vz[i] + azi * dt;

            double new_qx = s_qx[i] + vxi * dt;
            double new_qy = s_qy[i] + vyi * dt;
            double new_qz = s_qz[i] + vzi * dt;

            qx[i] = new_qx;
            qy[i] = new_qy;
            qz[i] = new_qz;
            vx[i] = vxi;
            vy[i] = vyi;
            vz[i] = vzi;

            s_qx[i] = new_qx;
            s_qy[i] = new_qy;
            s_qz[i] = new_qz;
        }

        __syncthreads();

        if (i == 0) {
            // Check collision
            double dx_pa = s_qx[planet] - s_qx[asteroid];
            double dy_pa = s_qy[planet] - s_qy[asteroid];
            double dz_pa = s_qz[planet] - s_qz[asteroid];
            double dist_pa = sqrt(dx_pa * dx_pa + dy_pa * dy_pa + dz_pa * dz_pa);
            if (dist_pa < planet_radius && s_collision_step < 0) {
                s_collision_step = step;
            }

            // Check missile hit
            if (s_missile_hit_step < 0) {
                double missile_dist = step * dt_param * missile_speed;
                double dx_pd = s_qx[planet] - s_qx[device_id];
                double dy_pd = s_qy[planet] - s_qy[device_id];
                double dz_pd = s_qz[planet] - s_qz[device_id];
                double dist_pd = sqrt(dx_pd * dx_pd + dy_pd * dy_pd + dz_pd * dz_pd);
                if (missile_dist > dist_pd) {
                    s_missile_hit_step = step;
                }
            }
        }

        __syncthreads();
    }

    if (i == 0) {
        result[0] = (s_collision_step < 0) ? 1 : 0;
        result[1] = s_missile_hit_step;
    }
}

// ============================================================================
// GPU memory management helpers
// ============================================================================
void allocate_device_arrays_gpu(DeviceArrays& dev, int n) {
    CHECK(hipMalloc(&dev.qx, n * sizeof(double)));
    CHECK(hipMalloc(&dev.qy, n * sizeof(double)));
    CHECK(hipMalloc(&dev.qz, n * sizeof(double)));
    CHECK(hipMalloc(&dev.vx, n * sizeof(double)));
    CHECK(hipMalloc(&dev.vy, n * sizeof(double)));
    CHECK(hipMalloc(&dev.vz, n * sizeof(double)));
    CHECK(hipMalloc(&dev.m, n * sizeof(double)));
    CHECK(hipMalloc(&dev.type, n * sizeof(int)));
}

void free_device_arrays(DeviceArrays& dev) {
    CHECK(hipFree(dev.qx));
    CHECK(hipFree(dev.qy));
    CHECK(hipFree(dev.qz));
    CHECK(hipFree(dev.vx));
    CHECK(hipFree(dev.vy));
    CHECK(hipFree(dev.vz));
    CHECK(hipFree(dev.m));
    CHECK(hipFree(dev.type));
}

void copy_host_to_device_arrays(DeviceArrays& dev, int n,
                                const double* qx, const double* qy, const double* qz,
                                const double* vx, const double* vy, const double* vz,
                                const double* m, const int* type) {
    CHECK(hipMemcpy(dev.qx, qx, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.qy, qy, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.qz, qz, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.vx, vx, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.vy, vy, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.vz, vz, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.m, m, n * sizeof(double), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.type, type, n * sizeof(int), hipMemcpyHostToDevice));
}

// ============================================================================
// Multi-GPU Problem 3 with HIP Streams: Overlap multiple device simulations
// ============================================================================
#include <omp.h>
#include <vector>
#include <algorithm>

// Async device-to-device copy using streams
void copy_device_arrays_async(DeviceArrays& dst, const DeviceArrays& src, int n, hipStream_t stream) {
    hipMemcpyAsync(dst.qx, src.qx, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(dst.qy, src.qy, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(dst.qz, src.qz, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(dst.vx, src.vx, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(dst.vy, src.vy, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(dst.vz, src.vz, n * sizeof(double), hipMemcpyDeviceToDevice, stream);
}

// Number of concurrent streams per GPU
#define NUM_STREAMS 4

void run_problem3_multi_gpu(int n_steps, int n, int planet, int asteroid,
                            const int* device_ids, int num_devices,
                            double planet_radius, double missile_speed, double dt,
                            const double* h_qx, const double* h_qy, const double* h_qz,
                            const double* h_vx, const double* h_vy, const double* h_vz,
                            const double* h_m, const int* h_type,
                            int* out_best_device_idx, int* out_best_hit_step) {
    
    // Get number of available GPUs
    int num_gpus = 0;
    CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus > 2) num_gpus = 2;  // Use at most 2 GPUs
    if (num_gpus < 1) num_gpus = 1;

    int blockSize = ((n + 31) / 32) * 32;
    if (blockSize < 32) blockSize = 32;
    if (blockSize > 1024) blockSize = 1024;

    // Arrays to store per-GPU results
    std::vector<int> gpu_best_device_idx(num_gpus, -1);
    std::vector<int> gpu_best_hit_step(num_gpus, -1);

    // Run on multiple GPUs in parallel using OpenMP
    #pragma omp parallel num_threads(num_gpus)
    {
        int tid = omp_get_thread_num();
        int gpu_id = tid;

        // Set current GPU
        CHECK(hipSetDevice(gpu_id));

        // Create streams for this GPU
        hipStream_t streams[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; s++) {
            CHECK(hipStreamCreate(&streams[s]));
        }

        // Allocate initial state array (shared across all streams)
        DeviceArrays dev_initial;
        allocate_device_arrays_gpu(dev_initial, n);
        copy_host_to_device_arrays(dev_initial, n, h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, h_type);

        // Allocate work arrays for each stream
        DeviceArrays dev_work[NUM_STREAMS];
        int* d_results[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; s++) {
            allocate_device_arrays_gpu(dev_work[s], n);
            // Copy constant data (m and type) once
            CHECK(hipMemcpy(dev_work[s].m, dev_initial.m, n * sizeof(double), hipMemcpyDeviceToDevice));
            CHECK(hipMemcpy(dev_work[s].type, dev_initial.type, n * sizeof(int), hipMemcpyDeviceToDevice));
            CHECK(hipMalloc(&d_results[s], 4 * sizeof(int)));
        }

        int local_best_device_idx = -1;
        int local_best_hit_step = -1;

        // Collect devices assigned to this GPU
        std::vector<int> my_devices;
        for (int d = tid; d < num_devices; d += num_gpus) {
            my_devices.push_back(d);
        }

        // Process devices in batches of NUM_STREAMS
        for (size_t batch_start = 0; batch_start < my_devices.size(); batch_start += NUM_STREAMS) {
            int batch_size = std::min((int)NUM_STREAMS, (int)(my_devices.size() - batch_start));

            // Launch all simulations in this batch asynchronously
            for (int s = 0; s < batch_size; s++) {
                int d = my_devices[batch_start + s];
                int device_id = device_ids[d];

                // Reset working arrays from initial state (async)
                copy_device_arrays_async(dev_work[s], dev_initial, n, streams[s]);

                // Run simulation for this device (async)
                hipLaunchKernelGGL(
                    simulate_problem3_kernel,
                    dim3(1), dim3(blockSize), 0, streams[s],
                    n, n_steps, planet, asteroid, device_id,
                    planet_radius, missile_speed, dt,
                    dev_work[s].qx, dev_work[s].qy, dev_work[s].qz,
                    dev_work[s].vx, dev_work[s].vy, dev_work[s].vz,
                    dev_work[s].m, dev_work[s].type,
                    d_results[s]
                );
            }

            // Wait for all streams in this batch and collect results
            for (int s = 0; s < batch_size; s++) {
                CHECK(hipStreamSynchronize(streams[s]));

                int results[2];
                CHECK(hipMemcpy(results, d_results[s], 2 * sizeof(int), hipMemcpyDeviceToHost));

                bool saved = (results[0] == 1);
                int missile_hit_step = results[1];

                int d = my_devices[batch_start + s];

                // If saved and missile hit, check if this is better
                if (saved && missile_hit_step >= 0) {
                    if (local_best_hit_step < 0 || missile_hit_step < local_best_hit_step) {
                        local_best_device_idx = d;
                        local_best_hit_step = missile_hit_step;
                    }
                }
            }
        }

        // Store local results
        gpu_best_device_idx[tid] = local_best_device_idx;
        gpu_best_hit_step[tid] = local_best_hit_step;

        // Cleanup GPU resources
        for (int s = 0; s < NUM_STREAMS; s++) {
            CHECK(hipFree(d_results[s]));
            free_device_arrays(dev_work[s]);
            CHECK(hipStreamDestroy(streams[s]));
        }
        free_device_arrays(dev_initial);
    }

    // Merge results from all GPUs (find minimum hit step)
    *out_best_device_idx = -1;
    *out_best_hit_step = -1;

    for (int g = 0; g < num_gpus; g++) {
        if (gpu_best_device_idx[g] >= 0 && gpu_best_hit_step[g] >= 0) {
            if (*out_best_hit_step < 0 || gpu_best_hit_step[g] < *out_best_hit_step) {
                *out_best_device_idx = gpu_best_device_idx[g];
                *out_best_hit_step = gpu_best_hit_step[g];
            }
        }
    }

    // Reset to GPU 0
    CHECK(hipSetDevice(0));
}
