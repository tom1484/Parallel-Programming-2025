#ifndef COMMON_HPP
#define COMMON_HPP

#include <lodepng.h>
#define GLM_FORCE_SWIZZLE

#include <glm/glm.hpp>

#define PI 3.1415926535897932384626433832795f
#define H_PI 1.5707963267948966192313216916398f

#define AA 3                 // anti-aliasing
#define HALF_AA (AA * 0.5f)  // half of AA
#define power 8.0f           // the power of the mandelbulb equation
#define md_iter 24           // the iteration count of the mandelbulb
#define ray_step 10000       // maximum step of ray marching
#define shadow_step 1500     // maximum step of shadow casting
#define step_limiter 0.2f    // the limit of each step length
#define ray_multiplier 0.1f  // prevent over-shooting, lower value for higher quality
#define bailout 2.0f         // escape radius
#define bailout2 4.0f        // escape radius
#define eps 0.0005f          // precision
#define FOV 1.5f             // fov ~66deg
#define far_plane 100.0f     // scene depth

typedef glm::vec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::vec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::vec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::mat3 mat3;  // 3x3 matrix

typedef unsigned char uchar;  // 8-bit unsigned integer
typedef unsigned int uint;    // 32-bit unsigned integer

#endif  // COMMON_HPP