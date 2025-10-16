#include "profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace std;

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::startSection(const string& name) {
    SectionTimer timer;
    timer.name = name;
    timer.start_time = chrono::high_resolution_clock::now();
    timer.depth = current_depth_;

    // Initialize timing data if this is the first call
    if (timings_.find(name) == timings_.end()) {
        timings_[name] = {0.0, 0, current_depth_, {}};
    }

    // Track parent-child relationships
    if (!timer_stack_.empty()) {
        const string& parent_name = timer_stack_.back().name;
        auto& parent_children = timings_[parent_name].children;
        if (find(parent_children.begin(), parent_children.end(), name) == parent_children.end()) {
            parent_children.push_back(name);
        }
    } else {
        // This is a root section
        if (root_section_.empty()) {
            root_section_ = name;
        }
    }

    timer_stack_.push_back(timer);
    current_depth_++;
}

void Profiler::endSection(const string& name) {
    auto end_time = chrono::high_resolution_clock::now();

    if (timer_stack_.empty()) {
        cerr << "Profiler error: endSection called without matching startSection for " << name << "\n";
        return;
    }

    const SectionTimer& timer = timer_stack_.back();
    if (timer.name != name) {
        cerr << "Profiler error: endSection name mismatch. Expected " << timer.name << ", got " << name << "\n";
        return;
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - timer.start_time);
    double duration_ms = duration.count() / 1000.0;

    timings_[name].total_time_ms += duration_ms;
    timings_[name].call_count++;

    timer_stack_.pop_back();
    current_depth_--;

    // Track total program time from root sections
    if (timer_stack_.empty()) {
        total_program_time_ms_ += duration_ms;
    }
}

void Profiler::printSection(const string& name, double parent_time_ms, string prefix, bool last) const {
    auto it = timings_.find(name);
    if (it == timings_.end()) return;

    const TimingData& data = it->second;

    // Calculate percentages
    double percent_of_parent = parent_time_ms > 0 ? (data.total_time_ms / parent_time_ms * 100.0) : 100.0;
    double percent_of_total = total_program_time_ms_ > 0 ? (data.total_time_ms / total_program_time_ms_ * 100.0) : 0.0;
    double avg_time_ms = data.call_count > 0 ? (data.total_time_ms / data.call_count) : 0.0;

    // Print indentation based on depth
    string decorator = last ? "└─ " : "├─ ";

    // clang-format off
    cout << prefix << decorator << setw(42 - (data.depth + 1) * 3) << left << name
              << right
              << setw(10) << fixed << setprecision(2) << data.total_time_ms << " ms"
              << setw(8) << fixed << setprecision(1) << percent_of_parent << "%"
              << setw(8) << fixed << setprecision(1) << percent_of_total << "%"
              << setw(7) << data.call_count << " calls"
              << setw(10) << fixed << setprecision(2) << avg_time_ms << " ms/call"
              << "\n";
    // clang-format on

    // Recursively print children
    for (size_t i = 0; i < data.children.size(); ++i) {
        const string& child = data.children[i];
        bool last_child = (i == data.children.size() - 1);
        string child_prefix = prefix + (last ? "   " : "│  ");
        printSection(child, data.total_time_ms, child_prefix, last_child);
    }
}

void Profiler::report() const {
    if (timings_.empty()) {
        cout << "\n===== Profiling Report =====";
        cout << "\nNo profiling data collected.";
        cout << "\n============================\n";
        return;
    }

    // clang-format off
    cout << "\n";
    cout << "========================================================================================================\n";
    cout << "                                         PROFILING REPORT                                             \n";
    cout << "========================================================================================================\n";
    cout << setw(42) << left << "Section"
              << setw(13) << right << "Time"
              << setw(9) << "Parent%"
              << setw(9) << "Total%"
              << setw(13) << "Calls"
              << setw(18) << "Avg Time"
              << "\n";
    cout << "--------------------------------------------------------------------------------------------------------\n";
    // clang-format on

    // Print all top-level sections (depth 0)
    vector<string> root_sections;
    for (const auto& entry : timings_) {
        if (entry.second.depth == 0) {
            root_sections.push_back(entry.first);
        }
    }
    for (size_t i = 0; i < root_sections.size(); ++i) {
        bool last = (i == root_sections.size() - 1);
        printSection(root_sections[i], total_program_time_ms_, "", last);
    }

    // clang-format off
    cout << "--------------------------------------------------------------------------------------------------------\n";
    cout << "Total measured time: " << fixed << setprecision(2) << total_program_time_ms_ << " ms\n";
    cout << "========================================================================================================\n\n";
    // clang-format on
}

void Profiler::reset() {
    timings_.clear();
    timer_stack_.clear();
    current_depth_ = 0;
    total_program_time_ms_ = 0.0;
    root_section_.clear();
}
