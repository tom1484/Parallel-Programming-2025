#include "bi_bfs_solver.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <queue>
#include <unordered_set>

using namespace std;

extern Game game;

// Normalize the state and check if it's dead or visited
optional<pair<uint64_t, size_t>> BiBFSSolver::normalize_and_insert_history(State& state, StateMode mode,
                                                                           const pair<Move, size_t>& new_op) {
    state.normalize(mode);
    if (state.dead) return nullopt;  // Dead state

    Visited& visited = (mode == StateMode::PUSH) ? forward_visited : backward_visited;
    History& history = (mode == StateMode::PUSH) ? forward_history : backward_history;

    uint64_t state_hash = state.hash();
    if (visited.count(state_hash)) return nullopt;  // Visited state
    size_t history_idx = history.size();
    visited[state_hash] = history_idx;
    history.push_back(new_op);

    // Hashing is expensive, return the hash value to avoid recomputation
    return {{state_hash, history_idx}};
}

void BiBFSSolver::forward_step() {
    queue<Node>& q = forward_queue;
    auto [curr_state, history_idx] = q.front();
    q.pop();

    const Move& prev_move = (history_idx != (size_t)(-1)) ? forward_history[history_idx].first : Move{};
    for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
        Position box = curr_state.reachable_boxes[box_id];
        vector<Direction> pushes = curr_state.available_pushes(box_id);

        for (const Direction& dir : pushes) {
            if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid pushing back

            State new_state = curr_state.push(box_id, dir);
            auto insert_result = normalize_and_insert_history(new_state, StateMode::PUSH, {{box, dir}, history_idx});
            if (!insert_result) continue;
            auto& [state_hash, new_history_idx] = insert_result.value();

            // Check if we meet the backward search
            if (backward_visited.count(state_hash)) {
                construct_solution(new_history_idx, backward_visited[state_hash]);
                return;
            }
            // Check if we reach the target
            if (new_state.boxes == game.targets) {
                construct_solution(new_history_idx, -1);
                return;
            }

            q.push({new_state, new_history_idx});
        }
    }
}

void BiBFSSolver::backward_step() {
    queue<Node>& q = backward_queue;
    auto [curr_state, history_idx] = q.front();
    q.pop();

    const Move& prev_move = (history_idx != (size_t)(-1)) ? backward_history[history_idx].first : Move{};
    for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
        Position box = curr_state.reachable_boxes[box_id];
        vector<Direction> pulls = curr_state.available_pulls(box_id);

        for (const Direction& dir : pulls) {
            if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid pushing back

            State new_state = curr_state.pull(box_id, dir);
            auto insert_result = normalize_and_insert_history(new_state, StateMode::PULL, {{box, dir}, history_idx});
            if (!insert_result) continue;
            auto& [state_hash, new_history_idx] = insert_result.value();

            // Check if we meet the forward search
            if (forward_visited.count(state_hash)) {
                construct_solution(forward_visited[state_hash], new_history_idx);
                return;
            }
            // Check if we meet the initial state
            if (new_state.boxes == game.initial_boxes && new_state.reachable[initial_state.player.to_index()]) {
                construct_solution(-1, new_history_idx);
                return;
            }

            q.push({new_state, new_history_idx});
        }
    }
}

void BiBFSSolver::construct_solution(size_t forward_history_idx, size_t backward_history_idx) {
    vector<Move> history;
    for (size_t i = forward_history_idx; i != (size_t)(-1); i = forward_history[i].second) {
        const auto& [move, prev_idx] = forward_history[i];
        history.push_back(move);
    }
    reverse(history.begin(), history.end());
    for (size_t i = backward_history_idx; i != (size_t)(-1); i = backward_history[i].second) {
        const auto& [move, prev_idx] = backward_history[i];
        const auto& [box, dir] = move;
        history.push_back({box + dir, dir_inv(dir)});
    }
    solution = history;
    solved = true;
}

#ifdef DEBUG
vector<Direction> BiBFSSolver::forward_solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved
    int last_print = 0;

    // State, (pushed box index, direction)
    State curr_state = initial_state;
    normalize_and_insert_history(curr_state, StateMode::PUSH, {Move(), -1});
    forward_queue.push({curr_state, -1});

    while (!solved && !forward_queue.empty()) {
        forward_step();
#ifdef DEBUG
        int total_explored = forward_visited.size();
        if (total_explored >= last_print + 10000) {
            cerr << "Explored: " << total_explored << endl;
            last_print += 10000;
        }
#endif
    }

    return expand_solution(initial_state);
}
#endif

#ifdef DEBUG
vector<Direction> BiBFSSolver::backward_solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved
    int last_print = 0;

    // State, (pushed box index, direction)
    State curr_state = initial_state;
    curr_state.boxes = game.targets;  // Inverse the target and initial state
    normalize_and_insert_history(curr_state, StateMode::PULL, {Move(), -1});
    backward_queue.push({curr_state, -1});

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
                normalize_and_insert_history(possible_state, StateMode::PULL, {Move(), -1});
                backward_queue.push({possible_state, -1});

                curr_state.reachable |= possible_state.reachable;
            }
        }
    }

    while (!solved && !backward_queue.empty()) {
        backward_step();
        int total_explored = backward_visited.size();
        if (total_explored >= last_print + 10000) {
            cerr << "Explored: " << total_explored << endl;
            last_print += 10000;
        }
    }

    return expand_solution(initial_state);
}
#endif

vector<Direction> BiBFSSolver::solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved
#ifdef DEBUG
    int last_print = 0;
#endif

    State curr_state = initial_state;
    // Initialize forward search
    normalize_and_insert_history(curr_state, StateMode::PUSH, {Move(), -1});
    forward_queue.push({curr_state, -1});
    // Initialize backward search
    curr_state.boxes = game.targets;  // Inverse the target and initial state
    normalize_and_insert_history(curr_state, StateMode::PULL, {Move(), -1});
    backward_queue.push({curr_state, -1});
    // Permute different end states for backward search
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
                normalize_and_insert_history(possible_state, StateMode::PULL, {Move(), -1});
                backward_queue.push({possible_state, -1});

                curr_state.reachable |= possible_state.reachable;
            }
        }
    }

    while (!solved && !forward_queue.empty() && !backward_queue.empty()) {
        if (forward_queue.size() <= backward_queue.size())
            forward_step();
        else
            backward_step();
#ifdef DEBUG
        int total_explored = forward_visited.size() + backward_visited.size();
        if (total_explored >= last_print + 10000) {
            cerr << "Explored: " << total_explored << endl;
            last_print += 10000;
        }
#endif
    }

    return expand_solution(initial_state);
}
