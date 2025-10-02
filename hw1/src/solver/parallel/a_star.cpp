#include "solver/parallel/a_star.hpp"

#include <pthread.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace ParallelAStar;

extern Game game;

// Use the Manhattan distance between each box and its nearest target as the heuristic
uint32_t ParallelAStar::heuristic(const State& state, Mode mode) {
    uint32_t h = 0;
    vector<Position>& target_list = (mode == FORWARD) ? game.target_list : game.initial_boxes_list;
    // NOTE: Use all boxes instead of only reachable boxes?
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
optional<InsertResult> ParallelAStar::Solver::normalize_and_insert_history(size_t thread_id, State& state, Mode mode,
                                                                           const pair<Move, HistoryIndex>& new_op) {
    state.normalize(mode);
    if (state.dead) return nullopt;  // Dead state

    pthread_mutex_t& visited_mutex = visiteds_mutex[mode];
    pthread_mutex_t& sub_history_mutex = sub_history_mutexes[mode][thread_id];

    const Visited& visited = visiteds[mode];
    Visited& sub_visited = sub_visiteds[mode][thread_id];
    History& sub_history = sub_histories[mode][thread_id];

    uint64_t state_hash = state.hash();

    // Lock visited mutex to safely read the global visited map
    pthread_mutex_lock(&visited_mutex);
    bool in_visited = visited.count(state_hash);
    pthread_mutex_unlock(&visited_mutex);

    // Lock history mutex to safely access thread's sub_visited and sub_history
    if (in_visited || sub_visited.count(state_hash)) {
        // pthread_mutex_unlock(&sub_history_mutex);
        return nullopt;  // Visited state
    }
    pthread_mutex_lock(&sub_history_mutex);
    HistoryIndex history_idx = {thread_id, sub_history.size()};
    sub_history.push_back(new_op);
    pthread_mutex_unlock(&sub_history_mutex);

    sub_visited[state_hash] = history_idx;  // Only insert into sub visited

    // Hashing is expensive, return the hash value to avoid recomputation
    return {{state_hash, history_idx}};
}

pair<Move, HistoryIndex> ParallelAStar::Solver::get_history_at(Mode mode, const HistoryIndex& history_idx) const {
    pthread_mutex_t& sub_history_mutex = const_cast<pthread_mutex_t&>(sub_history_mutexes[mode][history_idx.first]);
    const History& sub_history = sub_histories[mode][history_idx.first];

    pthread_mutex_lock(&sub_history_mutex);
    pair<Move, HistoryIndex> result = sub_history[history_idx.second];
    pthread_mutex_unlock(&sub_history_mutex);

    return result;
}

void ParallelAStar::Solver::forward_step(size_t thread_id) {
    Queue& dist_queue = dist_queues[FORWARD][thread_id];
    Queue& sub_queue = sub_queues[FORWARD][thread_id];
    auto [curr_depth, _, curr_state, history_idx] = dist_queue.front();
    dist_queue.pop();

    const Move& prev_move = (history_idx.second != (size_t)(-1)) ? get_history_at(FORWARD, history_idx).first : Move{};
    for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
        Position box = curr_state.reachable_boxes[box_id];
        vector<Direction> pushes = curr_state.available_pushes(box_id);

        for (const Direction& dir : pushes) {
            if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid moving back

            State new_state = curr_state.push(box_id, dir);
            auto insert_result = normalize_and_insert_history(thread_id, new_state, FORWARD, {{box, dir}, history_idx});
            if (!insert_result) continue;
            auto& [state_hash, new_history_idx] = insert_result.value();

            // Check if we meet the backward search
            pthread_mutex_lock(&visiteds_mutex[BACKWARD]);
            bool in_backward_visited = visiteds[BACKWARD].count(state_hash);
            if (in_backward_visited) {
                set_solution_history_idx(new_history_idx, visiteds[BACKWARD][state_hash]);
                pthread_mutex_unlock(&visiteds_mutex[BACKWARD]);
                return;
            }
            pthread_mutex_unlock(&visiteds_mutex[BACKWARD]);

            // if (new_state.boxes == game.targets) {
            //     set_solution_history_idx(new_history_idx, {0, (size_t)(-1)});
            //     return;
            // }

            sub_queue.push({curr_depth + 1, heuristic(new_state, FORWARD), new_state, new_history_idx});
        }
    }
}

void ParallelAStar::Solver::backward_step(size_t thread_id) {
    Queue& dist_queue = dist_queues[BACKWARD][thread_id];
    Queue& sub_queue = sub_queues[BACKWARD][thread_id];
    auto [curr_depth, _, curr_state, history_idx] = dist_queue.front();
    dist_queue.pop();

    const Move& prev_move = (history_idx.second != (size_t)(-1)) ? get_history_at(BACKWARD, history_idx).first : Move{};
    for (size_t box_id = 0; box_id < curr_state.reachable_boxes.size(); ++box_id) {
        Position box = curr_state.reachable_boxes[box_id];
        vector<Direction> pulls = curr_state.available_pulls(box_id);

        for (const Direction& dir : pulls) {
            if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid moving back

            State new_state = curr_state.pull(box_id, dir);
            auto insert_result =
                normalize_and_insert_history(thread_id, new_state, BACKWARD, {{box, dir}, history_idx});
            if (!insert_result) continue;
            auto& [state_hash, new_history_idx] = insert_result.value();

            // Check if we meet the backward search
            pthread_mutex_lock(&visiteds_mutex[FORWARD]);
            bool in_forward_visited = visiteds[FORWARD].count(state_hash);
            if (in_forward_visited) {
                set_solution_history_idx(visiteds[FORWARD][state_hash], new_history_idx);
                pthread_mutex_unlock(&visiteds_mutex[FORWARD]);
                return;
            }
            pthread_mutex_unlock(&visiteds_mutex[FORWARD]);

            // if (new_state.boxes == game.initial_boxes && new_state.reachable[initial_state.player.to_index()]) {
            //     set_solution_history_idx({0, (size_t)(-1)}, new_history_idx);
            //     return;
            // }

            sub_queue.push({curr_depth + 1, heuristic(new_state, BACKWARD), new_state, new_history_idx});
        }
    }
}

void ParallelAStar::Solver::run_batch(size_t thread_id, Mode mode) {
    Queue& dist_queue = dist_queues[mode][thread_id];
    while (!solved && !dist_queue.empty()) {
        switch (mode) {
            case FORWARD:
                forward_step(thread_id);
                break;
            case BACKWARD:
                backward_step(thread_id);
                break;
        }
    }
}

void ParallelAStar::Solver::distribute_batch(size_t batch_size, Mode mode) {
    pthread_t(*threads)[MAX_THREADS] = &mode_threads[mode];

    PQueue& pqueue = pqueues[mode];
    Visited& visited = visiteds[mode];
    vector<Queue>& dist_queue = dist_queues[mode];
    vector<Queue>& sub_queue = sub_queues[mode];
    vector<Visited>& sub_visited = sub_visiteds[mode];

    bool empty = false;
    size_t depth = pqueue.top().depth;  // Should be safe as queue is not empty
    size_t distributed = 0;

    // Distribute nodes of the same depth to sub-queues
    for (size_t thread_id = 0; !empty && distributed < batch_size;) {
        Node node = pqueue.top();
        pqueue.pop();
        dist_queue[thread_id].push(node);  // Assuming single thread for now
        empty = pqueue.empty() || pqueue.top().depth != depth;
        thread_id = (thread_id + 1) % num_threads;
        distributed++;
    }
#ifdef DEBUG
    cerr << "Mode: " << mode << ", depth: " << depth << ", distributed: " << distributed << endl;
#endif

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        int create_result = pthread_create(
            &(*threads)[thread_id], nullptr,
            [](void* arg) -> void* {
                auto* params = static_cast<pair<ParallelAStar::Solver*, pair<size_t, Mode>*>*>(arg);
                auto& [solver, p] = *params;
                solver->run_batch(p->first, p->second);
                return nullptr;
            },
            new pair<ParallelAStar::Solver*, pair<size_t, Mode>*>(this, new pair<size_t, Mode>{thread_id, mode}));
        if (create_result) {
            cerr << "Error creating thread: " << create_result << endl;
            exit(-1);
        }
    }

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        pthread_join((*threads)[thread_id], nullptr);
    }
    if (solved) return;

    // Merge sub visited into main visited
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        visited.merge(sub_visited[thread_id]);
        sub_visited[thread_id].clear();
    }
    // Merge sub queues into main queue
    for (Queue& sub_q : sub_queue) {
        while (!sub_q.empty()) {
            pqueue.push(sub_q.front());
            sub_q.pop();
        }
    }
}

void ParallelAStar::Solver::set_solution_history_idx(HistoryIndex forward_history_idx,
                                                     HistoryIndex backward_history_idx) {
    pthread_mutex_lock(&solution_mutex);
    if (solved) {
        pthread_mutex_unlock(&solution_mutex);
        return;
    }
    solved = true;
    solution_history_idx = {forward_history_idx, backward_history_idx};
    pthread_mutex_unlock(&solution_mutex);
}

void ParallelAStar::Solver::construct_solution() {
    // Lock all sub-histories
    auto& [forward_history_idx, backward_history_idx] = solution_history_idx;

    vector<Move> history;
    for (HistoryIndex i = forward_history_idx; i.second != (size_t)(-1);) {
        const auto& [move, prev_history_idx] = get_history_at(FORWARD, i);
        history.push_back(move);
        i = prev_history_idx;
    }
    reverse(history.begin(), history.end());

    for (HistoryIndex i = backward_history_idx; i.second != (size_t)(-1);) {
        const auto& [move, prev_history_idx] = get_history_at(BACKWARD, i);
        const auto& [box, dir] = move;
        history.push_back({box + dir, dir_inv(dir)});
        i = prev_history_idx;
    }

    solution = history;
}

ParallelAStar::Solver::Solver(const State& initial_state) : BaseSolver(initial_state) {
    num_threads = min(thread::hardware_concurrency(), (unsigned int)MAX_THREADS);

    pthread_mutex_init(&solution_mutex, nullptr);
    for (size_t mode : {FORWARD, BACKWARD}) {
        dist_queues[mode].resize(num_threads);
        sub_queues[mode].resize(num_threads);
        sub_visiteds[mode].resize(num_threads);
        sub_histories[mode].resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            dist_queues[mode][i] = Queue();
            sub_queues[mode][i] = Queue();
            sub_visiteds[mode][i] = Visited();
            sub_histories[mode][i] = History();
        }
        // Initialize mutexes
        pthread_mutex_init(&visiteds_mutex[mode], nullptr);
        sub_history_mutexes[mode].resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            pthread_mutex_init(&sub_history_mutexes[mode][i], nullptr);
        }
    }
}

ParallelAStar::Solver::~Solver() {
    // Destroy mutexes
    pthread_mutex_destroy(&solution_mutex);
    for (size_t mode : {FORWARD, BACKWARD}) {
        pthread_mutex_destroy(&visiteds_mutex[mode]);
        for (size_t i = 0; i < num_threads; ++i) {
            pthread_mutex_destroy(&sub_history_mutexes[mode][i]);
        }
    }
}

vector<Direction> ParallelAStar::Solver::solve() {
    if (game.targets == initial_state.boxes) return {};  // Already solved

    State curr_state = initial_state;
    HistoryIndex dummy_history_idx = {0, (size_t)(-1)};
    pair<Move, HistoryIndex> dummy_op = {Move(), dummy_history_idx};

    PQueue& forward_pqueue = pqueues[FORWARD];
    PQueue& backward_pqueue = pqueues[BACKWARD];

    // Initialize forward search (insert into first sub-container)
    normalize_and_insert_history(0, curr_state, FORWARD, dummy_op);
    forward_pqueue.push({0, heuristic(curr_state, FORWARD), curr_state, dummy_history_idx});

    // Initialize backward search
    curr_state.boxes = game.targets;  // Inverse the target and initial state
    curr_state.reachable.reset();
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
                normalize_and_insert_history(0, possible_state, BACKWARD, dummy_op);
                backward_pqueue.push({0, heuristic(possible_state, BACKWARD), possible_state, dummy_history_idx});

                curr_state.reachable |= possible_state.reachable;
            }
        }
    }

    Visited& forward_visited = visiteds[FORWARD];
    Visited& backward_visited = visiteds[BACKWARD];

    while (!solved && !forward_pqueue.empty() && !backward_pqueue.empty()) {
        if (forward_visited.size() <= backward_visited.size())
            distribute_batch(BATCH_SIZE, FORWARD);
        else
            distribute_batch(BATCH_SIZE, BACKWARD);
    }
    if (solved) {
        construct_solution();
    }

    return expand_solution(initial_state);
}