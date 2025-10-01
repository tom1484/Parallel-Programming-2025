#include "bi_bfs_solver.hpp"

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>

using namespace std;

extern Game game;

// Normalize the state and check if it's dead or visited
bool BiBFSSolver::normalize_and_check(State& state, StateMode mode, unordered_set<uint64_t>& visited) const {
    state.normalize(mode);
    if (state.dead) return true;  // Dead state

    uint64_t state_hash = state.hash();
    if (visited.count(state_hash)) return true;  // Visited state
    visited.insert(state_hash);
    return false;
}

void BiBFSSolver::forward_step(queue<pair<State, vector<Move>>>& q, unordered_set<uint64_t>& visited) {
    auto [curr_state, history] = q.front();
    q.pop();

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
                return;
            }

            if (normalize_and_check(new_state, StateMode::PUSH, visited)) continue;
            q.push({new_state, new_history});
        }
    }
}

void BiBFSSolver::backward_step(queue<pair<State, vector<Move>>>& q, unordered_set<uint64_t>& visited) {
    auto [curr_state, history] = q.front();
    q.pop();

    Move prev_move = history.empty() ? Move{} : history.back();
    for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
        Position box = curr_state.reachable_boxes[box_id];
        vector<Direction> pulls = curr_state.available_pulls(box_id);

        for (const Direction& dir : pulls) {
            if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid pushing back

            vector<Move> new_history = history;
            new_history.push_back({box, dir});

            State new_state = curr_state.pull(box_id, dir);
            if (normalize_and_check(new_state, StateMode::PULL, visited)) continue;

            if (new_state.boxes == game.initial_boxes && new_state.reachable[initial_state.player.to_index()]) {
                solved = true;
                vector<Move> inv_history;
                for (const auto& [box, dir] : new_history) inv_history.push_back({box + dir, dir_inv(dir)});
                reverse(inv_history.begin(), inv_history.end());
                solution = inv_history;
                return;
            }
            q.push({new_state, new_history});
        }
    }
}

// NOTE: Below is a simple version of inverse BFS without meeting in the middle
vector<Direction> BiBFSSolver::solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved

    // State, (pushed box index, direction)
    queue<pair<State, vector<Move>>> q;
    unordered_set<uint64_t> visited;

    State curr_state = initial_state;
    curr_state.boxes = game.targets;  // Inverse the target and initial state
    normalize_and_check(curr_state, StateMode::PULL, visited);
    q.push({curr_state, {}});

    // Permute different end states
    {
        Map block = game.box_map | curr_state.boxes;
        for (uint8_t y = 0; y < game.height; ++y) {
            for (uint8_t x = 0; x < game.width; ++x) {
                Position pos(x, y);
                uint16_t idx = pos.to_index();
                if (block[idx] || curr_state.reachable[idx]) continue;

                State possible_state = curr_state;
                possible_state.reset();
                possible_state.player = pos;
                normalize_and_check(possible_state, StateMode::PULL, visited);
                q.push({possible_state, {}});

                curr_state.reachable |= possible_state.reachable;
            }
        }
    }

    while (!solved && !q.empty()) {
        backward_step(q, visited);
    }

    return expand_solution(initial_state);
}
