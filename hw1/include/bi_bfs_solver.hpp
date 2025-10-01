#ifndef BI_BFS_SOLVER_HPP
#define BI_BFS_SOLVER_HPP

#include <queue>
#include <unordered_map>
#include <vector>
#include <optional>

#include "solver.hpp"

typedef vector<pair<Move, size_t>> History;       // ((current move), previous history index)
typedef pair<State, size_t> Node;                 // (state, history)
typedef unordered_map<uint64_t, size_t> Visited;  // (state hash, history index)

class BiBFSSolver : public Solver {
   private:
    queue<Node> forward_queue;
    queue<Node> backward_queue;
    // Hashmap of visited states and the histories index
    Visited forward_visited;
    Visited backward_visited;
    // Store the history of each visited state
    History forward_history;
    History backward_history;

    optional<size_t> normalize_and_insert_history(State& state, StateMode mode, const pair<Move, size_t>& new_op);
    void forward_step();
    void backward_step();

   public:
    BiBFSSolver(const State& initial_state) : Solver(initial_state) {}
    vector<Direction> solve() override;
#ifdef DEBUG
    vector<Direction> forward_solve();
    vector<Direction> backward_solve();
#endif
};

#endif  // BI_BFS_SOLVER_HPP
