#include "common.hpp"
#include "schedule.hpp"

uint width;        // image width
uint height;       // image height
vec2 iResolution;          // just for convenience of calculation

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

ScheduleDim dim;
