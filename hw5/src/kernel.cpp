#include "kernel.hpp"

__device__ double gravity_device_mass_dev(double m0, double t) { return m0 + 0.5 * m0 * fabs(sin(t / 6000.0)); }

__global__ void nbody_step_kernel(int n, int step, double* qx, double* qy, double* qz, double* vx, double* vy,
                                  double* vz, const double* m, const int* type, bool ignore_devices,
                                  int disabled_device) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double dt = 60.0;      // param::dt
    const double eps = 1e-3;     // param::eps
    const double G = 6.674e-11;  // param::G

    double t = step * dt;

    // local copies of position (maybe help register reuse)
    double qxi = qx[i];
    double qyi = qy[i];
    double qzi = qz[i];

    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    // Compute accelerations
    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        double mj = m[j];

        // Determine mj based on rules
        if (ignore_devices && type[j] == 1) {
            mj = 0.0;  // Problem 1 case
        } else if (j == disabled_device) {
            mj = 0.0;  // Problem 3 missile disabled
        } else if (type[j] == 1) {
            // fluctuating device mass
            mj = gravity_device_mass_dev(mj, t);
        }

        double dx = qx[j] - qxi;
        double dy = qy[j] - qyi;
        double dz = qz[j] - qzi;

        double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
        double dist3 = dist2 * sqrt(dist2);  // (dist^2)^(3/2)

        double coef = G * mj / dist3;
        axi += coef * dx;
        ayi += coef * dy;
        azi += coef * dz;
    }

    // Update velocities
    double vxi = vx[i] + axi * dt;
    double vyi = vy[i] + ayi * dt;
    double vzi = vz[i] + azi * dt;

    // Update positions
    qx[i] = qxi + vxi * dt;
    qy[i] = qyi + vyi * dt;
    qz[i] = qzi + vzi * dt;

    // Write updated velocities
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}