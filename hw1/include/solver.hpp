#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>

#include "game.hpp"

typedef pair<Position, Direction> Move;  // (pushed box index, direction)

class Solver {
   protected:
    bool solved = false;
    vector<Move> solution;
    vector<Direction> inner_path(const Map &boxes, const Position &start, const Position &end) const;
    vector<Direction> expand_solution(const State &initial_state) const;

   public:
    Solver() {}
    virtual vector<Direction> solve(const State &initial_state) = 0;
};

#endif  // SOLVER_HPP
