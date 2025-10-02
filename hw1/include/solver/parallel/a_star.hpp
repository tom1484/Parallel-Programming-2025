#ifndef PARALLEL_A_STAR_SOLVER_HPP
#define PARALLEL_A_STAR_SOLVER_HPP

#include <atomic>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "solver/base.hpp"

#define MAX_THREADS 16
#define BATCH_SIZE 2000

namespace ParallelAStar {

typedef pair<size_t, size_t> HistoryIndex;              // (thread id, vector index)
typedef vector<pair<Move, HistoryIndex>> History;       // ((current move), previous history index)
typedef unordered_map<uint64_t, HistoryIndex> Visited;  // (state hash, history index)

typedef struct Node {
    uint32_t depth;
    uint32_t heuristic;
    State state;
    HistoryIndex history_idx;
} Node;

struct NodeCompare {
    bool operator()(const Node& a, const Node& b) {
        uint64_t a_combined = ((uint64_t)(a.depth) << 32) + a.heuristic;
        uint64_t b_combined = ((uint64_t)(b.depth) << 32) + b.heuristic;
        return a_combined > b_combined;
    }
};
typedef queue<Node> Queue;
typedef priority_queue<Node, vector<Node>, NodeCompare> PQueue;

typedef pair<uint64_t, HistoryIndex> InsertResult;

uint32_t heuristic(const State& state, Mode mode);

class Solver : public BaseSolver {
   private:
    atomic<bool> solved = false;

    PQueue pqueues[2];
    Visited visiteds[2];  // This may be accessed by multiple threads

    size_t num_threads;
    pthread_t mode_threads[2][MAX_THREADS];
    // Sub-containers for threads
    vector<Queue> dist_queues[2];
    vector<Queue> sub_queues[2];
    vector<Visited> sub_visiteds[2];
    vector<History> sub_histories[2];  // Only this sub-container may be accessed by multiple threads

    // Store solution history indexes
    pair<HistoryIndex, HistoryIndex> solution_history_idx;

    // Mutexes for thread safety
    pthread_mutex_t solution_mutex;
    pthread_mutex_t visiteds_mutex[2];
    vector<pthread_mutex_t> sub_history_mutexes[2];

    optional<InsertResult> normalize_and_insert_history(size_t thread_id, State& state, Mode mode,
                                                        const pair<Move, HistoryIndex>& new_op);
    pair<Move, HistoryIndex> get_history_at(Mode mode, const HistoryIndex& history_idx) const;

    void forward_step(size_t thread_id);
    void backward_step(size_t thread_id);
    void run_batch(size_t thread_id, Mode mode);
    void distribute_batch(size_t batch_size, Mode mode);

    void set_solution_history_idx(HistoryIndex forward_history_idx, HistoryIndex backward_history_idx);
    void construct_solution();

   public:
    Solver(const State& initial_state);
    ~Solver();
    vector<Direction> solve() override;
};

}  // namespace ParallelAStar

#endif  // PARALLEL_A_STAR_SOLVER_HPP
