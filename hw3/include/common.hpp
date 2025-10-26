#ifndef COMMON_HPP
#define COMMON_HPP

#include <lodepng.h>
#define GLM_FORCE_SWIZZLE

#include <glm/glm.hpp>

#define PI 3.1415926535897932384626433832795

#define AA 3.0              // anti-aliasing
#define power 8.0           // the power of the mandelbulb equation
#define md_iter 24.0        // the iteration count of the mandelbulb
#define ray_step 10000.0    // maximum step of ray marching
#define shadow_step 1500.0  // maximum step of shadow casting
#define step_limiter 0.2    // the limit of each step length
#define ray_multiplier 0.1  // prevent over-shooting, lower value for higher quality
#define bailout 2.0         // escape radius
#define eps 0.0005          // precision
#define FOV 1.5             // fov ~66deg
#define far_plane 100.0     // scene depth

typedef glm::dvec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3;  // 3x3 matrix

#endif  // COMMON_HPP