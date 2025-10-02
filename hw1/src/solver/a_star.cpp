#include "solver/a_star.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <queue>
#include <unordered_map>

using namespace std;
using namespace AStar;

extern Game game;

// Use the Manhattan distance between each box and its nearest target as the heuristic
uint32_t AStar::heuristic(const State& state, StateMode mode) {
    uint32_t h = 0;
    vector<Position>& target_list = mode == StateMode::PUSH ? game.target_list : game.initial_boxes_list;
    for (const Position& box : state.reachable_boxes) {
        uint32_t min_dist = UINT32_MAX;
        for (const Position& target : target_list) {
            uint32_t dist = abs((int)box.x - (int)target.x) + abs((int)box.y - (int)target.y);
            min_dist = min(min_dist, dist);
        }
        h += min_dist;
    }
    return h;
}

// Normalize the state and check if it's dead or visited
optional<pair<uint64_t, size_t>> AStar::Solver::normalize_and_insert_history(State& state, StateMode mode,
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

void AStar::Solver::forward_step() {
    PQueue& q = forward_queue;
    auto [curr_depth, _, curr_state, history_idx] = q.top();
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
            // NOTE: No need to check if we meet the target state, since the target state is already inserted in
            // backward search

            q.push({curr_depth + 1, heuristic(new_state), new_state, new_history_idx});
        }
    }
}

void AStar::Solver::backward_step() {
    PQueue& q = backward_queue;
    auto [curr_depth, _, curr_state, history_idx] = q.top();
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
            // NOTE: No need to check if we meet the initial state, since the initial state is already inserted in
            // forward search

            q.push({curr_depth + 1, heuristic(new_state), new_state, new_history_idx});
        }
    }
}

void AStar::Solver::construct_solution(size_t forward_history_idx, size_t backward_history_idx) {
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

vector<Direction> AStar::Solver::solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved
#ifdef DEBUG
    int last_print = 0;
#endif

    State curr_state = initial_state;
    // Initialize forward search
    normalize_and_insert_history(curr_state, StateMode::PUSH, {Move(), -1});
    forward_queue.push({0, heuristic(curr_state), curr_state, (size_t)(-1)});
    // Initialize backward search
    curr_state.boxes = game.targets;  // Inverse the target and initial state
    normalize_and_insert_history(curr_state, StateMode::PULL, {Move(), -1});
    backward_queue.push({0, heuristic(curr_state), curr_state, (size_t)(-1)});
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
                backward_queue.push({0, heuristic(possible_state), possible_state, (size_t)(-1)});

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
