#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <limits>

void read_input(const char* filename, int& n, int& planet, int& asteroid, std::vector<double>& qx,
                std::vector<double>& qy, std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
                std::vector<double>& vz, std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step, int gravity_device_id,
                  double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}