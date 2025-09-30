#ifndef BI_BFS_SOLVER_HPP
#define BI_BFS_SOLVER_HPP

#include <queue>
#include <unordered_set>
#include <vector>

#include "solver.hpp"

class BiBFSSolver : public Solver {
   private:
    bool normalize_and_check_invalid(State& state, StateMode mode, unordered_set<uint64_t>& visited) const;
    void forward_step(queue<pair<State, vector<Move>>>& q, unordered_set<uint64_t>& visited);
    void backward_step(queue<pair<State, vector<Move>>>& q, unordered_set<uint64_t>& visited);

   public:
    BiBFSSolver() {}
    vector<Direction> solve(const State& initial_state) override;
};

#endif  // BI_BFS_SOLVER_HPP
