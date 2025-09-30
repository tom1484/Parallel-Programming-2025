#include <iostream>

#include "bfs_solver.hpp"
#include "bi_bfs_solver.hpp"
#include "game.hpp"

extern Game game;

int main(int /*argc*/, char* argv[]) {
    State root = game.load(argv[1]);
    game.mark_virtual_fragile_tiles();

    // BFSSolver solver;
    BiBFSSolver solver;
    vector<Direction> solution = solver.solve(root);

    string output = "";
    for (Direction dir : solution) output += dir_to_str(dir);
    cout << output << endl;

    return 0;
}
