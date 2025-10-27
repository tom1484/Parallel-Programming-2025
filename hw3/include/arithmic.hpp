#include "common.hpp"

__device__ inline float __min(const float& a, const float& b) { return a < b ? a : b; }

__device__ inline float __abs(const float& x) { return x < 0.f ? -x : x; }

__device__ inline float __clamp(const float& x, const float& a, const float& b) { return x < a ? a : (x > b ? b : x); }
__device__ inline vec2 __clamp(const vec2& v, const float& a, const float& b) {
    return vec2(__clamp(v.x, a, b), __clamp(v.y, a, b));
}
__device__ inline vec3 __clamp(const vec3& v, const float& a, const float& b) {
    return vec3(__clamp(v.x, a, b), __clamp(v.y, a, b), __clamp(v.z, a, b));
}

__device__ inline vec2 __add(const vec2& a, const vec2& b) { return vec2(a.x + b.x, a.y + b.y); }
__device__ inline vec3 __add(const vec3& a, const vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }

__device__ inline vec2 __sub(const vec2& a, const vec2& b) { return vec2(a.x - b.x, a.y - b.y); }
__device__ inline vec3 __sub(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }

__device__ inline vec2 __mul(const vec2& a, const vec2& b) { return vec2(a.x * b.x, a.y * b.y); }
__device__ inline vec3 __mul(const vec3& a, const vec3& b) { return vec3(a.x * b.x, a.y * b.y, a.z * b.z); }

__device__ inline vec2 __div(const vec2& a, const vec2& b) { return vec2(a.x / b.x, a.y / b.y); }
__device__ inline vec3 __div(const vec3& a, const vec3& b) { return vec3(a.x / b.x, a.y / b.y, a.z / b.z); }

__device__ inline float __cos(const float& a) { return __cosf(a); }
__device__ inline vec2 __cos(const vec2& a) { return vec2(__cosf(a.x), __cosf(a.y)); }
__device__ inline vec3 __cos(const vec3& a) { return vec3(__cosf(a.x), __cosf(a.y), __cosf(a.z)); }

__device__ inline float __sin(const float& a) { return __sinf(a); }
__device__ inline vec2 __sin(const vec2& a) { return vec2(__sinf(a.x), __sinf(a.y)); }
__device__ inline vec3 __sin(const vec3& a) { return vec3(__sinf(a.x), __sinf(a.y), __sinf(a.z)); }

__device__ inline float __fma(const float& a, const float& b, const float& c) { return a * b + c; }
__device__ inline vec2 __fma(const vec2& a, const vec2& b, const vec2& c) {
    return vec2(a.x * b.x + c.x, a.y * b.y + c.y);
}
__device__ inline vec2 __fma(const vec2& a, const float& b, const vec2& c) {
    return vec2(a.x * b + c.x, a.y * b + c.y);
}
__device__ inline vec3 __fma(const vec3& a, const vec3& b, const vec3& c) {
    return vec3(a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z);
}
__device__ inline vec3 __fma(const vec3& a, const float& b, const vec3& c) {
    return vec3(a.x * b + c.x, a.y * b + c.y, a.z * b + c.z);
}

__device__ inline float __length(const vec2& v) { return sqrtf(v.x * v.x + v.y * v.y); }
__device__ inline float __length(const vec3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }

__device__ inline vec2 __normalize(const vec2& v) {
    float inv_len = rsqrtf(v.x * v.x + v.y * v.y);
    return vec2(v.x * inv_len, v.y * inv_len);
}
__device__ inline vec3 __normalize(const vec3& v) {
    float inv_len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

__device__ inline vec3 __cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float __dot(const vec2& a, const vec2& b) { return a.x * b.x + a.y * b.y; }
__device__ inline float __dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float __pow(const float& x, const float& y) { return powf(x, y); }
__device__ inline vec2 __pow(const vec2& a, const vec2& b) { return vec2(__powf(a.x, b.x), __powf(a.y, b.y)); }
__device__ inline vec3 __pow(const vec3& a, const vec3& b) {
    return vec3(__powf(a.x, b.x), __powf(a.y, b.y), __powf(a.z, b.z));
}