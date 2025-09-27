#ifndef A_STAR_SOLVER_HPP
#define A_STAR_SOLVER_HPP

#include <vector>

#include "solver.hpp"

class AStarSolver : public Solver {
   public:
    AStarSolver() {}
    vector<Direction> solve(const State& initial_state) override;
};

#endif  // A_STAR_SOLVER_HPP
