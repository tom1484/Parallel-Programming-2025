#include "solver.hpp"

#include <algorithm>
#include <queue>

extern Game game;

// Find a path from start to end within the player's connected component (simple BFS)
vector<Direction> Solver::inner_path(const Map &boxes, const Position &start, const Position &end) const {
    vector<Direction> path;

    Map visited = game.player_map | boxes;  // Prevent walking into walls or boxes
    Direction from[MAX_SIZE] = {};   // NOTE: This could be optimized (i.e. make it global for reuse)

    queue<Position> q;
    q.push(start);

    while (!q.empty()) {
        Position curr = q.front();
        q.pop();

        if (curr == end)  // Found the end position
        {
            // Backtrack to find the path
            Position p = end;
            while (!(p == start)) {
                Direction d = from[p.to_index()];
                path.push_back(d);
                p = p - d;  // Move backwards
            }
            break;
        }
        for (Direction d : DIRECTIONS) {
            Position next = curr + d;
            if (!game.pos_valid(next)) continue;

            if (!visited[next.to_index()]) {
                visited.set(next.to_index());
                from[next.to_index()] = d;
                q.push(next);
            }
        }
    }

    reverse(path.begin(), path.end());
    return path;
}

vector<Direction> Solver::expand_solution(const State &initial_state, const vector<Move> &moves) const {
    vector<Direction> full_path;

    State state = initial_state;
    for (const auto &[box, dir] : moves) {
        // Find path from current player position to the pushing position
        Position push_pos = box - dir;
        vector<Direction> path_to_push = inner_path(state.boxes, state.player, push_pos);
        full_path.insert(full_path.end(), path_to_push.begin(), path_to_push.end());
        full_path.push_back(dir);  // Add the pushing direction

        // Update the state
        Position new_box = box + dir;
        state.boxes.reset(box.to_index());
        state.boxes.set(new_box.to_index());
        state.player = box;
    }

    return full_path;
}