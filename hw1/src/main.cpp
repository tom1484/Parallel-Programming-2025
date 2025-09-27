#include <iostream>

#include "bfs_solver.hpp"
#include "game.hpp"

extern Game game;

int main(int argc, char* argv[]) {
    State root = game.load(argv[1]);

    BFSSolver bfs_solver;
    vector<Direction> solution = bfs_solver.solve(root);

    for (Direction dir : solution) {
        cout << dir_to_str(dir) << endl;
    }

    return 0;
}
