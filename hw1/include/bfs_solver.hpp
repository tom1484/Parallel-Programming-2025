#ifndef BFS_SOLVER_HPP
#define BFS_SOLVER_HPP

#include "solver.hpp"

class BFSSolver : public Solver {
   public:
    BFSSolver() {}
    vector<Direction> solve(const State& initial_state) override;
};

#endif  // BFS_SOLVER_HPP
