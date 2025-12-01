#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>

void read_input(const char* filename, int& n, int& planet, int& asteroid, std::vector<double>& qx,
                std::vector<double>& qy, std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
                std::vector<double>& vz, std::vector<double>& m, std::vector<std::string>& type);

void write_output(const char* filename, double min_dist, int hit_time_step, int gravity_device_id,
                  double missile_cost);

#endif  // UTILS_HPP