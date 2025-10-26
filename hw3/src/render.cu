#include <iostream>

#include "render.hpp"
#include "schedule.hpp"

extern uint width;        // image width
extern uint height;       // image height
extern vec2 iResolution;  // just for convenience of calculation

extern vec3 camera_pos;  // camera position in 3D space (x, y, z)
extern vec3 target_pos;  // target position in 3D space (x, y, z)

extern ScheduleDim dim;

__constant__ uint d_width;
__constant__ uint d_height;
__constant__ vec2 d_iResolution;
__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ ScheduleDim d_dim;

void copy_constants_to_device() {
    cudaMemcpyToSymbol(d_width, &width, sizeof(uint));
    cudaMemcpyToSymbol(d_height, &height, sizeof(uint));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_dim, &dim, sizeof(ScheduleDim));
}

__device__ float _estimate(vec3 pos, float* trap) {
    vec3 v = pos;
    float dr = 1.f;            // |v'|
    float r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    *trap = r;

    for (int i = 0; i < md_iter; ++i) {
        float theta = glm::atan(v.y, v.x) * power;
        float phi = glm::asin(v.z / r) * power;

        // update vk+1 and dr
        dr = power * glm::pow(r, power - 1.f) * dr + 1.f;
        v = pos + glm::pow(r, power) * vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));

        // orbit trap for coloring
        *trap = glm::min(*trap, r);

        r = glm::length(v);      // update r
        if (r > bailout) break;  // if escaped
    }

    return 0.5f * log(r) * r / dr;  // mandelbulb's DE function
}

__device__ float _map(vec3 pos, float* trap) {
    vec2 rt = vec2(cos(PI / 2.f), sin(PI / 2.f));
    // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * pos;
    return _estimate(rp, trap);
}

__device__ float _trace_ray(vec3 origin, vec3 direction, float* trap) {
    float total_dis = 0;  // total distance
    float len = 0;        // current distance

    for (int i = 0; i < ray_step; ++i) {
        // get minimum distance from current ray position to the object's surface
        len = _map(origin + direction * total_dis, trap);
        if (glm::abs(len) < eps || total_dis > far_plane) break;
        total_dis += len * ray_multiplier;
    }
    // If exceeds the far plane then return -1 which means the ray missed a shot
    return total_dis < far_plane ? total_dis : -1.f;
}

__global__ void _render_pixel(uchar* buffer) {
    int x = d_dim.n_threads_x * blockIdx.x + threadIdx.x;
    int y = d_dim.n_threads_y * blockIdx.y + threadIdx.y;

    int pixel_index = (y * d_dim.batch_size_x + x) * 4;
    buffer[pixel_index] = 0;

    for (int m = 0; m < AA; ++m) {
        for (int n = 0; n < AA; ++n) {
            vec2 p = vec2(x, y) + vec2(m, n) / (float)AA;

            // Convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-d_iResolution.xy() + 2.f * p) / d_iResolution.y;
            uv.y *= -1;  // flip upside down

            // Create camera
            vec3 origin = d_camera_pos;                                                // ray (camera) origin
            vec3 target = d_target_pos;                                                // target position
            vec3 forward = glm::normalize(target - origin);                            // forward vector
            vec3 side = glm::normalize(glm::cross(forward, vec3(0.f, 1.f, 0.f)));      // right (side) vector
            vec3 up = glm::normalize(glm::cross(side, forward));                       // up vector
            vec3 direction = glm::normalize(uv.x * side + uv.y * up + FOV * forward);  // ray direction

            float trap;
            float d = _trace_ray(origin, direction, &trap);
        }
    }
}

void render_batch(uchar* buffer[], int size_x, int size_y) {
    // Launch kernel
    dim3 gridDim(dim.n_blocks_x, dim.n_blocks_y);
    dim3 blockDim(dim.n_threads_x, dim.n_threads_y);

    // Allocate device buffer pointer array
    uchar* d_buffer;
    cudaMalloc((void**)&d_buffer, dim.batch_size_x * dim.batch_size_y * 4);

    _render_pixel<<<gridDim, blockDim>>>(d_buffer);
    // Copy each row to host
    for (int by = 0; by < size_y; ++by) {
        cudaMemcpy(buffer[by], d_buffer + (by * dim.batch_size_x * 4), size_x * 4, cudaMemcpyDeviceToHost);
    }
}
