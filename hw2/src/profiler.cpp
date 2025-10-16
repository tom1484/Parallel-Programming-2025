#include "profiler.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace std;

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::initializeMPI(int rank, int size) {
    mpi_rank_ = rank;
    mpi_size_ = size;
}

void Profiler::startSection(const string& name) {
    int thread_id = omp_get_thread_num();

    lock_guard<mutex> lock(timings_mutex_);

    // Get or initialize thread-specific data
    auto& timer_stack = thread_timer_stacks_[thread_id];
    int& current_depth = thread_depths_[thread_id];

    SectionTimer timer;
    timer.name = name;
    timer.start_time = chrono::high_resolution_clock::now();
    timer.depth = current_depth;
    timer.thread_id = thread_id;

    // Initialize timing data if this is the first call
    if (timings_.find(name) == timings_.end()) {
        timings_[name] = {0.0, 0, current_depth, {}, 1e9, 0.0};
    }

    // Track parent-child relationships
    if (!timer_stack.empty()) {
        const string& parent_name = timer_stack.back().name;
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

    timer_stack.push_back(timer);
    current_depth++;
}

void Profiler::endSection(const string& name) {
    auto end_time = chrono::high_resolution_clock::now();
    int thread_id = omp_get_thread_num();

    lock_guard<mutex> lock(timings_mutex_);

    auto& timer_stack = thread_timer_stacks_[thread_id];
    int& current_depth = thread_depths_[thread_id];

    if (timer_stack.empty()) {
        cerr << "Profiler error: endSection called without matching startSection for " << name << "\n";
        return;
    }

    const SectionTimer& timer = timer_stack.back();
    if (timer.name != name) {
        cerr << "Profiler error: endSection name mismatch. Expected " << timer.name << ", got " << name << "\n";
        return;
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - timer.start_time);
    double duration_ms = duration.count() / 1000.0;

    timings_[name].total_time_ms += duration_ms;
    timings_[name].call_count++;
    timings_[name].min_time_ms = min(timings_[name].min_time_ms, duration_ms);
    timings_[name].max_time_ms = max(timings_[name].max_time_ms, duration_ms);

    timer_stack.pop_back();
    current_depth--;

    // Track total program time from root sections
    if (timer_stack.empty()) {
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

void Profiler::printAggregatedSection(const string& name, const map<string, TimingData>& aggregated_timings,
                                      double parent_time_ms, string prefix, bool last, double total_time) const {
    auto it = aggregated_timings.find(name);
    if (it == aggregated_timings.end()) return;

    const TimingData& data = it->second;

    // Calculate percentages and averages
    double percent_of_total = total_time > 0 ? (data.total_time_ms / total_time * 100.0) : 0.0;
    double avg_time_ms = data.call_count > 0 ? (data.total_time_ms / data.call_count) : 0.0;

    // Print indentation based on depth
    string decorator = last ? "└─ " : "├─ ";
    // clang-format off
    cout << prefix << decorator << setw(40 - (data.depth + 1) * 3) << left << name
              << right
              << setw(12) << fixed << setprecision(2) << data.total_time_ms
              << setw(12) << fixed << setprecision(2) << data.min_time_ms
              << setw(12) << fixed << setprecision(2) << data.max_time_ms
              << setw(9) << fixed << setprecision(1) << percent_of_total << "%"
              << setw(10) << data.call_count
              << setw(14) << fixed << setprecision(2) << avg_time_ms
              << "\n";
    // clang-format on

    // Recursively print children
    for (size_t i = 0; i < data.children.size(); ++i) {
        const string& child = data.children[i];
        bool last_child = (i == data.children.size() - 1);
        string child_prefix = prefix + (last ? "   " : "│  ");
        printAggregatedSection(child, aggregated_timings, data.total_time_ms, child_prefix, last_child, total_time);
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
    lock_guard<mutex> lock(timings_mutex_);
    timings_.clear();
    thread_timer_stacks_.clear();
    thread_depths_.clear();
    current_depth_ = 0;
    total_program_time_ms_ = 0.0;
    root_section_.clear();
}

void Profiler::gatherAndReport() {
    // Serialize local timing data
    vector<string> section_names;
    vector<double> total_times;
    vector<int> call_counts;
    vector<int> depths;
    vector<double> min_times;
    vector<double> max_times;
    vector<int> children_counts;
    vector<string> all_children;

    for (const auto& entry : timings_) {
        section_names.push_back(entry.first);
        total_times.push_back(entry.second.total_time_ms);
        call_counts.push_back(entry.second.call_count);
        depths.push_back(entry.second.depth);
        min_times.push_back(entry.second.min_time_ms);
        max_times.push_back(entry.second.max_time_ms);
        children_counts.push_back(entry.second.children.size());
        for (const auto& child : entry.second.children) {
            all_children.push_back(child);
        }
    }

    int local_section_count = section_names.size();
    double local_total_time = total_program_time_ms_;
    string local_root = root_section_;

    // Gather section counts from all ranks
    vector<int> all_section_counts(mpi_size_);
    MPI_Gather(&local_section_count, 1, MPI_INT, all_section_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank_ == 0) {
        // Aggregate timings from all ranks
        map<string, TimingData> aggregated_timings;
        map<string, vector<double>> rank_times;  // For computing statistics
        double max_total_time = local_total_time;

        // Add rank 0's data
        for (size_t i = 0; i < section_names.size(); ++i) {
            const string& name = section_names[i];
            aggregated_timings[name] = {total_times[i], call_counts[i], depths[i], {}, min_times[i], max_times[i]};
            rank_times[name].push_back(total_times[i]);

            // Restore children
            int child_offset = 0;
            for (size_t j = 0; j < i; ++j) {
                child_offset += children_counts[j];
            }
            for (int j = 0; j < children_counts[i]; ++j) {
                aggregated_timings[name].children.push_back(all_children[child_offset + j]);
            }
        }

        // Receive data from other ranks
        for (int rank = 1; rank < mpi_size_; ++rank) {
            int remote_section_count = all_section_counts[rank];

            // Receive total program time
            double remote_total_time;
            MPI_Recv(&remote_total_time, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            max_total_time = max(max_total_time, remote_total_time);

            // Receive root section name length and name
            int root_len;
            MPI_Recv(&root_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (root_len > 0) {
                vector<char> root_buf(root_len);
                MPI_Recv(root_buf.data(), root_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            for (int i = 0; i < remote_section_count; ++i) {
                // Receive section name
                int name_len;
                MPI_Recv(&name_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<char> name_buf(name_len);
                MPI_Recv(name_buf.data(), name_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string name(name_buf.begin(), name_buf.end());

                // Receive timing data
                double total_time, min_time, max_time;
                int call_count, depth, children_count;
                MPI_Recv(&total_time, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&call_count, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&depth, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&min_time, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&max_time, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&children_count, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Receive children names
                vector<string> children;
                for (int j = 0; j < children_count; ++j) {
                    int child_len;
                    MPI_Recv(&child_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    vector<char> child_buf(child_len);
                    MPI_Recv(child_buf.data(), child_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    children.push_back(string(child_buf.begin(), child_buf.end()));
                }

                // Aggregate the data
                if (aggregated_timings.find(name) == aggregated_timings.end()) {
                    aggregated_timings[name] = {total_time, call_count, depth, children, min_time, max_time};
                } else {
                    aggregated_timings[name].total_time_ms += total_time;
                    aggregated_timings[name].call_count += call_count;
                    aggregated_timings[name].min_time_ms = min(aggregated_timings[name].min_time_ms, min_time);
                    aggregated_timings[name].max_time_ms = max(aggregated_timings[name].max_time_ms, max_time);

                    // Merge children lists
                    for (const auto& child : children) {
                        if (find(aggregated_timings[name].children.begin(), aggregated_timings[name].children.end(),
                                 child) == aggregated_timings[name].children.end()) {
                            aggregated_timings[name].children.push_back(child);
                        }
                    }
                }

                rank_times[name].push_back(total_time);
            }
        }

        // clang-format off
        // Print aggregated report
        cout << "\n";
        cout << "==============================================================================================================\n";
        cout << "                                        PARALLEL PROFILING REPORT\n";
        cout << "                                        MPI Ranks: " << mpi_size_ << " | OMP Threads: " 
             << omp_get_max_threads() << "\n";
        cout << "==============================================================================================================\n";
        cout << setw(40) << left << "Section"
             << setw(12) << right << "Total(ms)"
             << setw(12) << "Min(ms)"
             << setw(12) << "Max(ms)"
             << setw(10) << "Total%"
             << setw(10) << "Calls"
             << setw(14) << "Avg(ms/call)"
             << "\n";
        cout << "--------------------------------------------------------------------------------------------------------------\n";
        // clang-format on

        // Print all top-level sections
        vector<string> root_sections;
        for (const auto& entry : aggregated_timings) {
            if (entry.second.depth == 0) {
                root_sections.push_back(entry.first);
            }
        }

        for (size_t i = 0; i < root_sections.size(); ++i) {
            bool last = (i == root_sections.size() - 1);
            printAggregatedSection(root_sections[i], aggregated_timings, max_total_time, "", last, max_total_time);
        }

        cout << "------------------------------------------------------------------------------------------------------"
                "--------\n";
        cout << "Total measured time: " << fixed << setprecision(2) << max_total_time << " ms (max across all ranks)\n";
        cout << "======================================================================================================"
                "========\n\n";

    } else {
        // Send data to rank 0
        MPI_Send(&local_total_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        // Send root section name
        int root_len = local_root.size();
        MPI_Send(&root_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (root_len > 0) {
            MPI_Send(local_root.data(), root_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }

        for (size_t i = 0; i < section_names.size(); ++i) {
            const string& name = section_names[i];
            int name_len = name.size();
            MPI_Send(&name_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(name.data(), name_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

            MPI_Send(&total_times[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&call_counts[i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&depths[i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&min_times[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&max_times[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&children_counts[i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Send children names
            int child_offset = 0;
            for (size_t j = 0; j < i; ++j) {
                child_offset += children_counts[j];
            }
            for (int j = 0; j < children_counts[i]; ++j) {
                const string& child = all_children[child_offset + j];
                int child_len = child.size();
                MPI_Send(&child_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(child.data(), child_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
}
