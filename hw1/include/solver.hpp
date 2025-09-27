#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>

#include "game.hpp"

class Solver {
   public:
    Solver() {}

    vector<Direction> inner_path(const State& state, Position start,
                                 Position end);
    virtual vector<Direction> solve(const State& initial_state) = 0;
};

#endif  // SOLVER_HPP
