#ifndef BFS_SOLVER_HPP
#define BFS_SOLVER_HPP

#include <unordered_set>

#include "solver.hpp"

class BFSSolver : public Solver {
   private:
    bool normalize_and_check_invalid(State& state, unordered_set<uint64_t>& visited) const;

   public:
    BFSSolver() {}
    vector<Direction> solve(const State& initial_state) override;
};

#endif  // BFS_SOLVER_HPP
