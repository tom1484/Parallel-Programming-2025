#include <iostream>

#include "solver/a_star.hpp"
#include "game.hpp"

extern Game game;

int main(int /*argc*/, char* argv[]) {
    State initial_state = game.load(argv[1]);
    game.mark_virtual_fragile_tiles();

    AStar::Solver solver(initial_state);
    vector<Direction> solution = solver.solve();

    string output = "";
    for (Direction dir : solution) output += dir_to_str(dir);
    cout << output << endl;

    return 0;
}
