#include "game.hpp"

#include <fstream>
#include <queue>
#include <string>
#include <vector>

using namespace std;

// Global variables

Game game;

// class Position implementation

Position::Position() : x(0), y(0) {}

Position::Position(uint8_t x, uint8_t y) : x(x), y(y) {}

uint16_t Position::to_index() const { return (y << EDGE_BITS) + x; }

// WARNING: Not tested
bool Position::is_dead_pos(Map block, bool advanced) const {
    static const Direction corner_dirs[4][2] = {
        {Direction::LEFT, Direction::UP},
        {Direction::UP, Direction::RIGHT},
        {Direction::RIGHT, Direction::DOWN},
        {Direction::DOWN, Direction::LEFT},
    };
    static const Direction wall_dirs[4][3] = {
        {Direction::LEFT, Direction::UP, Direction::DOWN},
        {Direction::UP, Direction::LEFT, Direction::RIGHT},
        {Direction::RIGHT, Direction::UP, Direction::DOWN},
        {Direction::DOWN, Direction::LEFT, Direction::RIGHT},
    };

    for (const Direction* comb : corner_dirs) {
        Position neighbor0 = *this + comb[0];
        Position neighbor1 = *this + comb[1];
        if (block[neighbor0.to_index()] && block[neighbor1.to_index()] && !game.targets[this->to_index()]) return true;
    }

    if (advanced) {
        for (const Direction* comb : wall_dirs) {
            Position wall_pivot = *this + comb[0];
            if (!block[wall_pivot.to_index()]) continue;

            for (int i = 1; i < 3; i++) {
                Direction dir = comb[i];
                Position self = *this + dir;
                Position wall = wall_pivot + dir;
                bool escaped = false;
                do {
                    if ((!block[wall.to_index()] && !block[self.to_index()]) || game.targets[self.to_index()]) {
                        escaped = true;
                        break;
                    }
                    self = self + dir;
                    wall = wall + dir;
                } while (game.pos_valid(self) && game.pos_valid(wall));

                if (!escaped) return true;
            }
        }
    }

    return false;
}

Position Position::operator+(const Direction& dir) const {
    DirectionDelta delta = dir_to_delta(dir);
    return Position(static_cast<unsigned char>(x + delta.dx), static_cast<unsigned char>(y + delta.dy));
}

Position Position::operator-(const Direction& dir) const {
    DirectionDelta delta = dir_to_delta(dir);
    return Position(static_cast<unsigned char>(x - delta.dx), static_cast<unsigned char>(y - delta.dy));
}

bool Position::operator==(const Position& other) const { return x == other.x && y == other.y; }

bool Position::operator<(const Position& other) const { return (y < other.y) || (y == other.y && x < other.x); }

inline void set_pos(Map& bset, const Position& pos) { bset.set(pos.to_index()); }

inline void reset_pos(Map& bset, const Position& pos) { bset.reset(pos.to_index()); }

// class State implementation

State::State() : player(0, 0), boxes(0), normalized(false), dead(false) {}

State::State(Position init_player, Map boxes) : player(init_player), boxes(boxes), normalized(false), dead(false) {}

// Move the box and update the current connected component
State State::push(int box_id, const Direction& dir) const {
    const Position& box = available_boxes[box_id];
    Position new_box = box + dir;
    Map new_boxes = boxes;

    reset_pos(new_boxes, box);
    set_pos(new_boxes, new_box);

    Position new_player = box;
    return State(new_player, new_boxes);
}

// NOTE: The player position may be removed from the returned values
vector<pair<Position, Direction>> State::available_pushes(int box_id) const {
    Map block = game.map | boxes;
    vector<pair<Position, Direction>> pushes;

    for (Direction dir : DIRECTIONS) {
        const Position& box = available_boxes[box_id];
        Position player_pos = box - dir;
        Position new_box = box + dir;

        if (!game.pos_valid(new_box) || !game.pos_valid(player_pos)) continue;
        if (!reachable[player_pos.to_index()] || block[new_box.to_index()]) continue;

        pushes.push_back({player_pos, dir});
    }

    return pushes;
}

void State::normalize() {
    Map block = game.map | boxes;
    Map visited;

    queue<Position> q;
    q.push(player);

    while (!q.empty()) {
        Position curr = q.front();
        q.pop();

        visited.set(curr.to_index());
        reachable.set(curr.to_index());

        // Update the new player position to be the smallest position in
        // the component
        if (curr < player) player = curr;

        for (Direction d : DIRECTIONS) {
            Position next = curr + d;
            if (!game.pos_valid(next)) continue;

            uint16_t idx = next.to_index();
            if (!visited[idx]) {
                if (!block[idx])  // Empty space
                    q.push(next);
                if (boxes[idx]) {
                    if (next.is_dead_pos(block)) {
                        normalized = false;
                        dead = true;
                        return;
                    }
                    available_boxes.push_back(next);  // A reachable box
                    visited.set(idx);
                }
            }
        }
    }

    normalized = true;
    dead = false;
}

// class Game implementation

State Game::load(const char* sample_filepath) {
    ifstream file(sample_filepath);
    if (!file.is_open()) throw runtime_error("Failed to open file");

    string line;
    vector<string> lines;
    while (getline(file, line)) lines.push_back(line);
    file.close();

    width = lines[0].size();
    height = lines.size();

    Position player;
    Map boxes;
    boxes.reset();

    map.reset();
    targets.reset();

    Position pos;
    for (pos.y = 0; pos.y < height; pos.y++) {
        string& row = lines[pos.y];
        if (row.size() != width) throw runtime_error("Inconsistent row width");

        for (pos.x = 0; pos.x < width; pos.x++) {
            char cell = row[pos.x];
            if (cell == '#')
                map.set(pos.to_index());
            else if (cell == '.')
                targets.set(pos.to_index());
            else if (cell == 'x')
                boxes.set(pos.to_index());
            else if (cell == 'X') {
                boxes.set(pos.to_index());
                targets.set(pos.to_index());
            }
            else if (cell == 'o')
                player = pos;
            else if (cell == 'O') {
                player = pos;
                targets.set(pos.to_index());
            }
        }
    }

    return State(player, boxes);
}

Game::Game() {}
