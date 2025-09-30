#include "bfs_solver.hpp"

#include <iostream>
#include <queue>
#include <unordered_set>

using namespace std;

extern Game game;

// Normalize the state and check if it's dead or visited
bool BFSSolver::normalize_and_check_invalid(State& state, unordered_set<uint64_t>& visited) const {
    state.normalize();
    if (state.dead) return true;  // Dead state

    uint64_t state_hash = state.hash();
    if (visited.count(state_hash)) return true;  // Visited state
    visited.insert(state_hash);
    return false;
}

vector<Direction> BFSSolver::solve(const State& initial_state) {
    if (game.targets == initial_state.boxes) return {};  // Already solved

    // State, (pushed box index, direction)
    queue<pair<State, vector<Move>>> q;
    unordered_set<uint64_t> visited;

    solved = false;
    solution.clear();

    State curr_state = initial_state;
    curr_state.normalize();

    q.push({curr_state, {}});
    while (!q.empty()) {
        auto [curr_state, history] = q.front();
        q.pop();

        // NOTE: This is tested to be slower for now
        // if (normalize_and_check_invalid(curr_state, visited)) continue;

        Move prev_move = history.empty() ? Move{} : history.back();
        for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
            Position box = curr_state.reachable_boxes[box_id];
            vector<Direction> pushes = curr_state.available_pushes(box_id);

            for (const Direction& dir : pushes) {
                if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid pushing back

                vector<Move> new_history = history;
                new_history.push_back({box, dir});

                State new_state = curr_state.push(box_id, dir);
                if (new_state.boxes == game.targets) {
                    solved = true;
                    solution = new_history;
                    break;
                }

                if (normalize_and_check_invalid(new_state, visited)) continue;
                q.push({new_state, new_history});
            }
            if (solved) break;
        }
        if (solved) break;

#ifdef DEBUG
        if (visited.size() % 10000 == 0) {
            cerr << "Explored: " << visited.size() << ", Queue size: " << q.size() << endl;
        }
#endif
    }

    // cout << "Solution length: " << solution.size() << endl;
    // for (const auto& [box, dir] : solution) {
    //     cout << "Push box " << (int)box.x << ", " << (int)box.y << " to direction " << dir << endl;
    // }
    return expand_solution(initial_state);
}
