#include <cstring>
#include <iostream>

#include "arithmic.hpp"
#include "render.hpp"
#include "schedule.hpp"
#include "utils.hpp"

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
    float dr = 1.f;         // |v'|
    float r = __length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    *trap = r;

    for (int i = 0; i < md_iter; ++i) {
        float theta = atan2f(v.y, v.x) * power;
        float phi = asinf(v.z / r) * power;

        // update vk+1 and dr
        dr = __fma(power * __pow(r, power - 1.f), dr, 1.f);
        v = __fma(vec3(__cosf(theta) * __cosf(phi), __cosf(phi) * __sinf(theta), -__sinf(phi)), __pow(r, power), pos);

        // orbit trap for coloring
        *trap = __min(*trap, r);

        r = __length(v);         // update r
        if (r > bailout) break;  // if escaped
    }

    return 0.5f * logf(r) * r / dr;  // mandelbulb's DE function
}

__device__ float _map_trap(vec3 pos, float* trap) {
    vec2 rt = vec2(0.f, 1.f);
    // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * pos;
    return _estimate(rp, trap);
}

__device__ float _map(vec3 pos) {
    float _trap;  // dummy
    return _map_trap(pos, &_trap);
}

__device__ vec3 _palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return __fma(__cos(2.f * PI * __fma(c, t, d)), b, a);
}

__device__ float _softshadow(vec3 origin, vec3 direction, float k) {
    float res = 1.0f;
    float t = 0.f;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        float h = _map(__fma(direction, t, origin));
        res = __min(res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02f) return 0.02f;
        t += __clamp(h, .001f, step_limiter);  // move ray
    }
    return __clamp(res, .02f, 1.f);
}

__device__ float _trace_ray(vec3 origin, vec3 direction, float* trap) {
    float total_dis = 0;  // total distance
    float len = 0;        // current distance

    for (int i = 0; i < ray_step; ++i) {
        // get minimum distance from current ray position to the object's surface
        len = _map_trap(__fma(direction, total_dis, origin), trap);
        if (__abs(len) < eps || total_dis > far_plane) break;
        total_dis = __fma(len, ray_multiplier, total_dis);
    }
    // If exceeds the far plane then return -1 which means the ray missed a shot
    return total_dis < far_plane ? total_dis : -1.f;
}

// use gradient to calc surface normal
__device__ vec3 calculate_norm(vec3 p) {
    vec2 e = vec2(eps, 0.f);
    return __normalize(vec3(_map(p + e.xyy()) - _map(p - e.xyy()),  // dx
                            _map(p + e.yxy()) - _map(p - e.yxy()),  // dy
                            _map(p + e.yyx()) - _map(p - e.yyx())   // dz
                            ));
}

__global__ void __launch_bounds__(256, 8) _render_pixel(uchar* buffer) {
    int x = d_dim.n_threads_x * blockIdx.x + threadIdx.x;
    int y = d_dim.n_threads_y * blockIdx.y + threadIdx.y;
    if (x >= d_width || y >= d_height) return;

    int pixel_index = (y * d_width + x) * 4;

    float final_color_r = 0.0f;
    float final_color_g = 0.0f;
    float final_color_b = 0.0f;

    for (int m = 0; m < AA; ++m) {
        for (int n = 0; n < AA; ++n) {
            vec2 pos = vec2(x, y) + vec2(m, n) / (float)AA;

            // Convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-d_iResolution.xy() + 2.f * pos) / d_iResolution.y;
            uv.y *= -1;  // flip upside down

            // Create camera
            vec3 origin = d_camera_pos;                                             // ray (camera) origin
            vec3 target = d_target_pos;                                             // target position
            vec3 forward = __normalize(target - origin);                            // forward vector
            vec3 side = __normalize(__cross(forward, vec3(0.f, 1.f, 0.f)));         // right (side) vector
            vec3 up = __normalize(__cross(side, forward));                          // up vector
            vec3 direction = __normalize(uv.x * side + uv.y * up + FOV * forward);  // ray direction

            float trap;
            float depth = _trace_ray(origin, direction, &trap);

            // Lighting
            vec3 color(0.f);                             // color
            vec3 light_dir = __normalize(d_camera_pos);  // sun direction (directional light)
            vec3 light_color = vec3(1.f, .9f, .717f);    // light color

            // Coloring
            if (depth < 0.f) {      // miss (hit sky)
                color = vec3(0.f);  // sky color (black)
            } else {
                vec3 pos = origin + direction * depth;          // hit position
                vec3 nr = calculate_norm(pos);                  // get surface normal
                vec3 hal = __normalize(light_dir - direction);  // blinn-phong lighting model (vector h)

                // use orbit trap to get the color
                color = _palette(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f), vec3(.0f, .1f, .2f));  // diffuse color
                vec3 ambient_color = vec3(0.3f);                                                     // ambient color
                float gloss = 32.f;                                                                  // specular gloss

                // simple blinn phong lighting model
                float ambient = (0.7f + 0.3f * nr.y) *
                                (0.2f + 0.8f * __clamp(0.05f * (float)logf(trap), 0.0f, 1.0f));  // self occlution
                float shadow = _softshadow(pos + .001f * nr, light_dir, 16.f);                   // shadow
                float diffuse = __clamp(__dot(light_dir, nr), 0.f, 1.f) * shadow;                // diffuse
                float specular = __pow(__clamp(__dot(nr, hal), 0.f, 1.f), gloss) * diffuse;      // self shadow

                vec3 lin(0.f);
                lin += ambient_color * (.05f + .95f * ambient);  // ambient color * ambient
                lin += light_color * diffuse * 0.8f;             // diffuse * light color * light intensity
                color *= lin;

                color = __pow(color, vec3(.7f, .9f, 1.f));  // fake SSS (subsurface scattering)
                color += specular * 0.8f;                   // specular
            }

            color = __clamp(__pow(color, vec3(.4545f)), 0.f, 1.f);  // gamma correction
            // fcol += vec4(col, 1.f);
            final_color_r += color.r;
            final_color_g += color.g;
            final_color_b += color.b;
        }
    }
    // fcol /= (float)(AA * AA);
    final_color_r /= (float)(AA * AA);
    final_color_g /= (float)(AA * AA);
    final_color_b /= (float)(AA * AA);
    // convert float (0~1) to unsigned char (0~255)
    // fcol *= 255.0f;
    final_color_r *= 255.0f;
    final_color_g *= 255.0f;
    final_color_b *= 255.0f;

    buffer[pixel_index + 0] = (unsigned char)final_color_r;  // r
    buffer[pixel_index + 1] = (unsigned char)final_color_g;  // g
    buffer[pixel_index + 2] = (unsigned char)final_color_b;  // b
    buffer[pixel_index + 3] = 255;                           // a
}

void render(uchar* raw_image) {
    uchar* d_buffer;
    cudaMalloc((void**)&d_buffer, width * height * 4);
    copy_constants_to_device();

    // Set dimensions
    dim3 gridDim(dim.n_blocks_x, dim.n_blocks_y);
    dim3 blockDim(dim.n_threads_x, dim.n_threads_y);

#ifdef DEBUG
    int block_size = dim.n_threads_x * dim.n_threads_y;
    estimate_occupancy((void*)_render_pixel, block_size, 0);
#endif

    _render_pixel<<<gridDim, blockDim>>>(d_buffer);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(raw_image, d_buffer, width * height * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_buffer);
}
