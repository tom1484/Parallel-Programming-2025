#ifndef A_STAR_SOLVER_HPP
#define A_STAR_SOLVER_HPP

#include <vector>

#include "solver.hpp"

class AStarSolver : public Solver {
   public:
    AStarSolver(const State& initial_state) : Solver(initial_state) {}
    vector<Direction> solve() override;
};

#endif  // A_STAR_SOLVER_HPP
