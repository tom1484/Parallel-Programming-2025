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

    // Build call path: parent_path/name
    string call_path;
    string parent_path;
    if (!timer_stack.empty()) {
        parent_path = timer_stack.back().call_path;
        call_path = parent_path + "/" + name;
    } else {
        call_path = name;  // Root level
    }

    SectionTimer timer;
    timer.name = name;
    timer.call_path = call_path;
    timer.start_time = chrono::high_resolution_clock::now();
    timer.depth = current_depth;
    timer.thread_id = thread_id;

    // Initialize timing data if this is the first call to this path
    if (timings_.find(call_path) == timings_.end()) {
        bool is_mpi = (name.find("MPI_") == 0);  // Check if name starts with "MPI_"
        timings_[call_path] = {0.0, 0, current_depth, {}, 1e9, 0.0, is_mpi, name, parent_path};
    }

    // Track parent-child relationships using call paths
    if (!timer_stack.empty()) {
        const string& parent_call_path = timer_stack.back().call_path;
        auto& parent_children = timings_[parent_call_path].children;
        if (find(parent_children.begin(), parent_children.end(), call_path) == parent_children.end()) {
            parent_children.push_back(call_path);
        }
    } else {
        // This is a root section
        if (root_section_.empty()) {
            root_section_ = call_path;
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

    const string& call_path = timer.call_path;
    timings_[call_path].total_time_ms += duration_ms;
    timings_[call_path].call_count++;
    timings_[call_path].min_time_ms = min(timings_[call_path].min_time_ms, duration_ms);
    timings_[call_path].max_time_ms = max(timings_[call_path].max_time_ms, duration_ms);

    timer_stack.pop_back();
    current_depth--;

    // Track total program time from root sections
    if (timer_stack.empty()) {
        total_program_time_ms_ += duration_ms;
    }
}

void Profiler::printAggregatedSection(const string& call_path, const map<string, TimingData>& aggregated_timings,
                                      double parent_time_ms, string prefix, bool last, double total_time) const {
    auto it = aggregated_timings.find(call_path);
    if (it == aggregated_timings.end()) return;

    const TimingData& data = it->second;

    // Calculate percentages and averages
    double percent_of_total = total_time > 0 ? (data.total_time_ms / total_time * 100.0) : 0.0;
    double avg_time_ms = data.call_count > 0 ? (data.total_time_ms / data.call_count) : 0.0;

    // Print indentation based on depth
    string local_prefix = prefix + (last ? "└─ " : "├─ ");
    // clang-format off
    cout << left
         << local_prefix
         << setw(50 - (data.depth + 1) * 3) << data.display_name
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

void Profiler::reset() {
    lock_guard<mutex> lock(timings_mutex_);
    timings_.clear();
    thread_timer_stacks_.clear();
    thread_depths_.clear();
    current_depth_ = 0;
    total_program_time_ms_ = 0.0;
    root_section_.clear();
}

double Profiler::getTotalMPITime() const {
    double total_mpi_time = 0.0;
    for (const auto& entry : timings_) {
        if (entry.second.is_mpi) {
            total_mpi_time += entry.second.total_time_ms;
        }
    }
    return total_mpi_time;
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

    vector<string> display_names;
    vector<string> parent_paths;

    for (const auto& entry : timings_) {
        section_names.push_back(entry.first);  // call_path
        display_names.push_back(entry.second.display_name);
        parent_paths.push_back(entry.second.parent_path);
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
            const string& call_path = section_names[i];
            const string& display_name = display_names[i];
            const string& parent_path = parent_paths[i];
            bool is_mpi = (display_name.find("MPI_") == 0);
            aggregated_timings[call_path] = {total_times[i], call_counts[i], depths[i],    {},         min_times[i],
                                             max_times[i],   is_mpi,         display_name, parent_path};
            rank_times[call_path].push_back(total_times[i]);

            // Restore children
            int child_offset = 0;
            for (size_t j = 0; j < i; ++j) {
                child_offset += children_counts[j];
            }
            for (int j = 0; j < children_counts[i]; ++j) {
                aggregated_timings[call_path].children.push_back(all_children[child_offset + j]);
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
                // Receive call_path
                int name_len;
                MPI_Recv(&name_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<char> name_buf(name_len);
                MPI_Recv(name_buf.data(), name_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string call_path(name_buf.begin(), name_buf.end());

                // Receive display_name
                int display_len;
                MPI_Recv(&display_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<char> display_buf(display_len);
                MPI_Recv(display_buf.data(), display_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string display_name(display_buf.begin(), display_buf.end());

                // Receive parent_path
                int parent_len;
                MPI_Recv(&parent_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<char> parent_buf(parent_len);
                MPI_Recv(parent_buf.data(), parent_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string parent_path(parent_buf.begin(), parent_buf.end());

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
                bool is_mpi = (display_name.find("MPI_") == 0);
                if (aggregated_timings.find(call_path) == aggregated_timings.end()) {
                    aggregated_timings[call_path] = {total_time, call_count, depth,        children,   min_time,
                                                     max_time,   is_mpi,     display_name, parent_path};
                } else {
                    aggregated_timings[call_path].total_time_ms += total_time;
                    aggregated_timings[call_path].call_count += call_count;
                    aggregated_timings[call_path].min_time_ms =
                        min(aggregated_timings[call_path].min_time_ms, min_time);
                    aggregated_timings[call_path].max_time_ms =
                        max(aggregated_timings[call_path].max_time_ms, max_time);

                    // Merge children lists
                    for (const auto& child : children) {
                        if (find(aggregated_timings[call_path].children.begin(),
                                 aggregated_timings[call_path].children.end(),
                                 child) == aggregated_timings[call_path].children.end()) {
                            aggregated_timings[call_path].children.push_back(child);
                        }
                    }
                }

                rank_times[call_path].push_back(total_time);
            }
        }

        // clang-format off
        // Print aggregated report
        cout << "\n";
        cout << "========================================================================================================================\n";
        cout << "PARALLEL PROFILING REPORT\n";
        cout << "MPI Ranks: " << mpi_size_ << " | OMP Threads: " 
             << omp_get_max_threads() << "\n";
        cout << "========================================================================================================================\n";
        cout << left
             << setw(50) << "Section"
             << right
             << setw(12) << "Total(ms)"
             << setw(12) << "Min(ms)"
             << setw(12) << "Max(ms)"
             << setw(10) << "Total%"
             << setw(10) << "Calls"
             << setw(14) << "Avg(ms/call)"
             << "\n";
        cout << "------------------------------------------------------------------------------------------------------------------------\n";
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

        // Calculate total MPI communication time
        double total_mpi_time = 0.0;
        for (const auto& entry : aggregated_timings) {
            if (entry.second.is_mpi) {
                total_mpi_time += entry.second.total_time_ms;
            }
        }
        double mpi_percent = max_total_time > 0 ? (total_mpi_time / max_total_time * 100.0) : 0.0;

        // clang-format off
        cout << "------------------------------------------------------------------------------------------------------------------------\n";
        cout << "Total measured time: " << fixed << setprecision(2) << max_total_time << " ms (max across all ranks)\n";
        cout << "Total MPI communication time: " << fixed << setprecision(2) << total_mpi_time 
             << " ms (" << fixed << setprecision(1) << mpi_percent << "% of total)\n";
        cout << "========================================================================================================================\n\n";
        // clang-format on

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
            const string& call_path = section_names[i];
            const string& display_name = display_names[i];
            const string& parent_path = parent_paths[i];

            // Send call_path
            int name_len = call_path.size();
            MPI_Send(&name_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(call_path.data(), name_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

            // Send display_name
            int display_len = display_name.size();
            MPI_Send(&display_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(display_name.data(), display_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

            // Send parent_path
            int parent_len = parent_path.size();
            MPI_Send(&parent_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(parent_path.data(), parent_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

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
