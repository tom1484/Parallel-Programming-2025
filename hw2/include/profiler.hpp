#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

using namespace std;

// Enable/disable profiling globally
#define PROFILING_ENABLED

#ifdef PROFILING_ENABLED
#define PROFILE_FUNCTION() ScopedTimer _timer(__FUNCTION__)
#define PROFILE_SCOPE(name) ScopedTimer _timer(name)
#define PROFILE_MPI(name) ScopedTimer _mpi_timer(string("MPI_") + name)
#else
#define PROFILE_FUNCTION()
#define PROFILE_SCOPE(name)
#define PROFILE_MPI(name)
#endif

// Column widths
// #define

struct TimingData {
    double total_time_ms;
    int call_count;
    int depth;
    vector<string> children;  // Call paths of children
    double min_time_ms;
    double max_time_ms;
    bool is_mpi;          // Flag to identify MPI communication sections
    string display_name;  // Simple name for display (without path)
    string parent_path;   // Full path of parent for hierarchy
};

class Profiler {
   public:
    static Profiler& getInstance();

    void startSection(const string& name);
    void endSection(const string& name);
    void report() const;
    void reset();

    // MPI/OpenMP support
    void initializeMPI(int rank, int size);
    void gatherAndReport();  // Gather from all ranks and report on rank 0

    // Get total MPI communication time
    double getTotalMPITime() const;

   private:
    Profiler() : current_depth_(0), total_program_time_ms_(0.0), mpi_rank_(0), mpi_size_(1) {}
    ~Profiler() = default;

    // Prevent copying
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    struct SectionTimer {
        string name;       // Display name
        string call_path;  // Full call path for unique identification
        chrono::high_resolution_clock::time_point start_time;
        int depth;
        int thread_id;
    };

    map<string, TimingData> timings_;  // Key is call_path, not just name
    vector<SectionTimer> timer_stack_;
    int current_depth_;
    double total_program_time_ms_;
    string root_section_;  // Call path of root section

    // MPI info
    int mpi_rank_;
    int mpi_size_;

    // Thread safety
    mutex timings_mutex_;

    // Per-thread timer stacks (for OpenMP support)
    map<int, vector<SectionTimer>> thread_timer_stacks_;
    map<int, int> thread_depths_;

    void printAggregatedSection(const string& name, const map<string, TimingData>& aggregated_timings,
                                double parent_time_ms, string prefix, bool last, double total_time) const;
};

// RAII timer class for automatic scope-based timing
class ScopedTimer {
   public:
    explicit ScopedTimer(const string& name) : name_(name) { Profiler::getInstance().startSection(name_); }

    ~ScopedTimer() { Profiler::getInstance().endSection(name_); }

   private:
    string name_;
};

#endif  // PROFILER_HPP
