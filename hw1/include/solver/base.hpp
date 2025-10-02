#ifndef BASE_SOLVER_HPP
#define BASE_SOLVER_HPP

#include <vector>

#include "game.hpp"

typedef pair<Position, Direction> Move;  // (pushed box index, direction)

class BaseSolver {
   protected:
    const State initial_state;
    bool solved = false;
    vector<Move> solution;

    vector<Direction> inner_path(const Map &boxes, const Position &start, const Position &end) const;
    vector<Direction> expand_solution(const State &initial_state) const;

   public:
    BaseSolver(const State& initial_state) : initial_state(initial_state) {}
    virtual vector<Direction> solve() = 0;
};

#endif  // SOLVER_HPP
