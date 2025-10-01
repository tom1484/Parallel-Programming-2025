#ifndef BFS_SOLVER_HPP
#define BFS_SOLVER_HPP

#include <unordered_set>

#include "solver.hpp"

namespace BFS {

class BFSSolver : public BaseSolver {
   private:
    bool normalize_and_check(State& state, unordered_set<uint64_t>& visited) const;

   public:
    BFSSolver(const State& initial_state) : BaseSolver(initial_state) {}
    vector<Direction> solve() override;
};

}  // namespace BFS

#endif  // BFS_SOLVER_HPP
