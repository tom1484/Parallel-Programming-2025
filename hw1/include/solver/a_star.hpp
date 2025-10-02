#ifndef A_STAR_SOLVER_HPP
#define A_STAR_SOLVER_HPP

#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "solver/base.hpp"

namespace AStar {

typedef vector<pair<Move, size_t>> History;       // ((current move), previous history index)
typedef unordered_map<uint64_t, size_t> Visited;  // (state hash, history index)

typedef struct Node {
    uint32_t depth;
    uint32_t heuristic;
    State state;
    size_t history_idx;
} Node;

struct NodeCompare {
    bool operator()(const Node& a, const Node& b) {
        uint64_t a_combined = ((uint64_t)(a.depth) << 32) + a.heuristic;
        uint64_t b_combined = ((uint64_t)(b.depth) << 32) + b.heuristic;
        return a_combined > b_combined;
    }
};
typedef priority_queue<Node, vector<Node>, NodeCompare> PQueue;

uint32_t heuristic(const State& state, StateMode mode = StateMode::PUSH);

class Solver : public BaseSolver {
   private:
    PQueue forward_queue;
    PQueue backward_queue;
    // Hashmap of visited states and the histories index
    Visited forward_visited;
    Visited backward_visited;
    // Store the history of each visited state
    History forward_history;
    History backward_history;

    optional<pair<uint64_t, size_t>> normalize_and_insert_history(State& state, StateMode mode,
                                                                  const pair<Move, size_t>& new_op);
    void forward_step();
    void backward_step();
    void construct_solution(size_t forward_history_idx, size_t backward_history_idx);

   public:
    Solver(const State& initial_state) : BaseSolver(initial_state) {}
    vector<Direction> solve() override;
};

}  // namespace AStar

#endif  // A_STAR_SOLVER_HPP
