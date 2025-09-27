#include "bfs_solver.hpp"

#include <iostream>
#include <queue>

using namespace std;

extern Game game;

vector<Direction> BFSSolver::solve(const State& initial_state) {
    // State, (pushed box index, direction)
    queue<pair<State, vector<Move>>> q;

    bool solved = false;
    vector<Move> solution;

    q.push({initial_state, {}});
    while (!q.empty()) {
        auto [curr_state, history] = q.front();
        q.pop();

        curr_state.normalize();
        if (curr_state.dead) continue;  // Skip dead states
        Move prev_move = history.empty() ? Move{Position(), Direction::LEFT} : history.back();

        for (int box_id = 0; box_id < curr_state.available_boxes.size(); ++box_id) {
            Position box = curr_state.available_boxes[box_id];
            vector<pair<Position, Direction>> pushes = curr_state.available_pushes(box_id);
            for (const auto& [player_pos, dir] : pushes) {
                if (box == prev_move.first && dir == dir_inv(prev_move.second)) continue;  // Avoid pushing back

                vector<Move> new_history = history;
                new_history.push_back({box, dir});

                State new_state = curr_state.push(box_id, dir);
                if (new_state.boxes == game.targets) {
                    solved = true;
                    solution = new_history;
                    break;
                }

                q.push({new_state, new_history});
            }

            if (solved) break;
        }

        if (solved) break;
    }

    // cout << "Solution length: " << solution.size() << endl;
    // for (const auto& [box, dir] : solution) {
    //     cout << "Push box " << (int)box.x << ", " << (int)box.y << " to direction " << dir << endl;
    // }
    return expand_solution(initial_state, solution);
}
