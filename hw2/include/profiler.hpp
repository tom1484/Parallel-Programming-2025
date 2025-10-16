#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <map>
#include <string>
#include <vector>

using namespace std;

// Enable/disable profiling globally
#define PROFILING_ENABLED

#ifdef PROFILING_ENABLED
#define PROFILE_FUNCTION() ScopedTimer _timer(__FUNCTION__)
#define PROFILE_SCOPE(name) ScopedTimer _timer(name)
#else
#define PROFILE_FUNCTION()
#define PROFILE_SCOPE(name)
#endif

// Column widths
// #define

struct TimingData {
    double total_time_ms;
    int call_count;
    int depth;
    vector<string> children;
};

class Profiler {
   public:
    static Profiler& getInstance();

    void startSection(const string& name);
    void endSection(const string& name);
    void report() const;
    void reset();

   private:
    Profiler() : current_depth_(0), total_program_time_ms_(0.0) {}
    ~Profiler() = default;

    // Prevent copying
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    struct SectionTimer {
        string name;
        chrono::high_resolution_clock::time_point start_time;
        int depth;
    };

    map<string, TimingData> timings_;
    vector<SectionTimer> timer_stack_;
    int current_depth_;
    double total_program_time_ms_;
    string root_section_;

    void printSection(const string& name, double parent_time_ms, string prefix, bool last) const;
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
