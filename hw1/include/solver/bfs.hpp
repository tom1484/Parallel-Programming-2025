#ifndef BFS_SOLVER_HPP
#define BFS_SOLVER_HPP

#include <unordered_set>

#include "solver/base.hpp"

namespace BFS {

class Solver : public BaseSolver {
   private:
    bool solved = false;
    bool normalize_and_check(State& state, unordered_set<uint64_t>& visited) const;

   public:
    Solver(const State& initial_state) : BaseSolver(initial_state) {}
    vector<Direction> solve() override;
};

}  // namespace BFS

#endif  // BFS_SOLVER_HPP
