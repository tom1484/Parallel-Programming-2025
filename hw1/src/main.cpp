#include <iostream>

#include "bfs_solver.hpp"
#include "game.hpp"

extern Game game;

int main(int argc, char* argv[]) {
    State root = game.load(argv[1]);

    BFSSolver bfs_solver;
    // auto path = bfs_solver.inner_path(root, Position(1, 1), Position(3, 2));

    return 0;
}
