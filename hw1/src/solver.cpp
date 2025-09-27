#include "solver.hpp"

#include <algorithm>
#include <queue>

extern Game game;

// Find a path from start to end within the player's connected component (simple
// BFS)
vector<Direction> Solver::inner_path(const State& state, Position start,
                                     Position end) {
    vector<Direction> path;

    Map visited =
        game.map | state.boxes;  // Prevent walking into walls or boxes
    Direction from[MAX_SIZE] =
        {};  // NOTE: This could be optimized (i.e. make it global for reuse)

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
