#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE  // vec3.xyz(), vec3.xyx() ...ect, these are called "Swizzle".
// https://glm.g-truc.net/0.9f.1f/api/a00002.html
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9f.9f/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)

#define PI 3.1415926535897932384626433832795f

typedef glm::vec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::vec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::vec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::mat3 mat3;  // 3x3 matrix

unsigned int num_threads;  // number of thread
unsigned int width;        // image width
unsigned int height;       // image height
vec2 iResolution;          // just for convenience of calculation

int AA = 3;  // anti-aliasing

float power = 8.0f;           // the power of the mandelbulb equation
float md_iter = 24;          // the iteration count of the mandelbulb
float ray_step = 10000;      // maximum step of ray marching
float shadow_step = 1500;    // maximum step of shadow casting
float step_limiter = 0.2f;    // the limit of each step length
float ray_multiplier = 0.1f;  // prevent over-shooting, lower value for higher quality
float bailout = 2.0f;         // escape radius
float eps = 0.0005f;          // precision
float FOV = 1.5f;             // fov ~66deg
float far_plane = 100.f;      // scene depth

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

unsigned char* raw_image;  // 1D image
unsigned char** image;     // 2D image

// save raw_image to PNG file
void write_png(const char* filename) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
float md(vec3 p, float& trap) {
    vec3 v = p;
    float dr = 1.f;             // |v'|
    float r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        float theta = glm::atan(v.y, v.x) * power;
        float phi = glm::asin(v.z / r) * power;
        dr = power * glm::pow(r, power - 1.f) * dr + 1.f;
        v = p + glm::pow(r, power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v);      // update r
        if (r > bailout) break;  // if escaped
    }
    return 0.5f * log(r) * r / dr;  // mandelbulb's DE function
}

// scene mapping
float map(vec3 p, float& trap, int& ID) {
    vec2 rt = vec2(cos(PI / 2.f), sin(PI / 2.f));
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) *
              p;  // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
float map(vec3 p) {
    float dmy;  // dummy
    int dmy2;    // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2.f * PI * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softshadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0f;
    float t = 0.f;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        float h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02f) return 0.02f;
        t += glm::clamp(h, .001f, step_limiter);  // move ray
    }
    return glm::clamp(res, .02f, 1.f);
}

// use gradient to calc surface normal
vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.f);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
        map(p + e.yxy()) - map(p - e.yxy()),                    // dy
        map(p + e.yyx()) - map(p - e.yyx())                     // dz
        ));
}

// first march: find object's surface
float trace(vec3 ro, vec3 rd, float& trap, int& ID) {
    float t = 0;    // total distance
    float len = 0;  // current distance

    for (int i = 0; i < ray_step; ++i) {
        len = map(ro + rd * t, trap,
            ID);  // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }
    return t < far_plane
               ? t
               : -1.f;  // if exceeds the far plane then return -1 which means the ray missed a shot
}

int main(int argc, char** argv) {
    // ./hw3 [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    // x1 y1 z1: camera position in 3D space
    // x2 y2 z2: target position in 3D space
    // width height: image size
    // filename: filename
    assert(argc == 10);

    //---init arguments
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);

    float total_pixel = width * height;
    float current_pixel = 0;

    iResolution = vec2(width, height);
    //---

    //---create image
    raw_image = new unsigned char[width * height * 4];
    image = new unsigned char*[height];

    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * 4;
    }
    //---

    //---start rendering
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // vec4 fcol(0.f);  // final color (RGBA 0 ~ 1)
            float fcol_r = 0.0f;
            float fcol_g = 0.0f;
            float fcol_b = 0.0f;

            // anti aliasing
            for (int m = 0; m < AA; ++m) {
                for (int n = 0; n < AA; ++n) {
                    vec2 p = vec2(j, i) + vec2(m, n) / (float)AA;

                    //---convert screen space coordinate to (-ap~ap, -1~1)
                    // ap = aspect ratio = width/height
                    vec2 uv = (-iResolution.xy() + 2.f * p) / iResolution.y;
                    uv.y *= -1;  // flip upside down
                    //---

                    //---create camera
                    vec3 ro = camera_pos;               // ray (camera) origin
                    vec3 ta = target_pos;               // target position
                    vec3 cf = glm::normalize(ta - ro);  // forward vector
                    vec3 cs =
                        glm::normalize(glm::cross(cf, vec3(0.f, 1.f, 0.f)));  // right (side) vector
                    vec3 cu = glm::normalize(glm::cross(cs, cf));          // up vector
                    vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);  // ray direction
                    //---

                    //---marching
                    float trap;  // orbit trap
                    int objID;    // the object id intersected with
                    float d = trace(ro, rd, trap, objID);
                    //---

                    //---lighting
                    vec3 col(0.f);                          // color
                    vec3 sd = glm::normalize(camera_pos);  // sun direction (directional light)
                    vec3 sc = vec3(1.f, .9f, .717f);          // light color
                    //---

                    //---coloring
                    if (d < 0.f) {        // miss (hit sky)
                        col = vec3(0.f);  // sky color (black)
                    } else {
                        vec3 pos = ro + rd * d;              // hit position
                        vec3 nr = calcNor(pos);              // get surface normal
                        vec3 hal = glm::normalize(sd - rd);  // blinn-phong lighting model (vector
                                                             // h)
                        // for more info:
                        // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

                        // use orbit trap to get the color
                        col = pal(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f),
                            vec3(.0f, .1f, .2f));  // diffuse color
                        vec3 ambc = vec3(0.3f);  // ambient color
                        float gloss = 32.f;     // specular gloss

                        // simple blinn phong lighting model
                        float amb =
                            (0.7f + 0.3f * nr.y) *
                            (0.2f + 0.8f * glm::clamp(0.05f * (float)log(trap), 0.0f, 1.0f));  // self occlution
                        float sdw = softshadow(pos + .001f * nr, sd, 16.f);         // shadow
                        float dif = glm::clamp(glm::dot(sd, nr), 0.f, 1.f) * sdw;   // diffuse
                        float spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0.f, 1.f), gloss) *
                                     dif;  // self shadow

                        vec3 lin(0.f);
                        lin += ambc * (.05f + .95f * amb);  // ambient color * ambient
                        lin += sc * dif * 0.8f;            // diffuse * light color * light intensity
                        col *= lin;

                        col = glm::pow(col, vec3(.7f, .9f, 1.f));  // fake SSS (subsurface scattering)
                        col += spe * 0.8f;                       // specular
                    }
                    //---

                    col = glm::clamp(glm::pow(col, vec3(.4545f)), 0.f, 1.f);  // gamma correction
                    // fcol += vec4(col, 1.f);
                    fcol_r += col.r;
                    fcol_g += col.g;
                    fcol_b += col.b;
                }
            }

            // fcol /= (float)(AA * AA);
            fcol_r /= (float)(AA * AA);
            fcol_g /= (float)(AA * AA);
            fcol_b /= (float)(AA * AA);
            // convert float (0~1) to unsigned char (0~255)
            // fcol *= 255.0f;
            fcol_r *= 255.0f;
            fcol_g *= 255.0f;
            fcol_b *= 255.0f;
            image[i][4 * j + 0] = (unsigned char)fcol_r;  // r
            image[i][4 * j + 1] = (unsigned char)fcol_g;  // g
            image[i][4 * j + 2] = (unsigned char)fcol_b;  // b
            image[i][4 * j + 3] = 255;                    // a

            // print progress
            // printf("rendering...%5.2lf%%\r", current_pixel / total_pixel * 100.f);
            current_pixel++;
        }
    }
    //---

    //---saving image
    write_png(argv[9]);
    //---

    //---finalize
    delete[] raw_image;
    delete[] image;
    //---

    return 0;
}
