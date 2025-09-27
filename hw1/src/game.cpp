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

uint16_t Position::to_index() { return (y << EDGE_BITS) + x; }

Position Position::operator+(Direction dir) {
    DirectionDelta delta = to_delta(dir);
    return Position(static_cast<unsigned char>(x + delta.dx),
                    static_cast<unsigned char>(y + delta.dy));
}

Position Position::operator-(Direction dir) {
    DirectionDelta delta = to_delta(dir);
    return Position(static_cast<unsigned char>(x - delta.dx),
                    static_cast<unsigned char>(y - delta.dy));
}

bool Position::operator==(const Position& other) {
    return x == other.x && y == other.y;
}

bool Position::operator<(const Position& other) {
    return (y < other.y) || (y == other.y && x < other.x);
}

inline void set_pos(Map& bset, Position pos) { bset.set(pos.to_index()); }

inline void reset_pos(Map& bset, Position pos) { bset.reset(pos.to_index()); }

// class State implementation

State::State() : player(0, 0), boxes(0) {}

State::State(Position init_player, Map boxes) : boxes(boxes) {
    Map visited = game.map | boxes;
    player = init_player;

    queue<Position> q;
    q.push(player);

    while (!q.empty()) {
        Position curr = q.front();
        q.pop();

        // Update the new player position to be the smallest position in
        // the component
        if (curr < player) player = curr;

        for (Direction d : DIRECTIONS) {
            Position next = curr + d;
            if (!game.pos_valid(next)) continue;

            if (!visited[next.to_index()]) {
                visited.set(next.to_index());
                q.push(next);
            } else if (boxes[next.to_index()])
                available_boxes.push_back(next);  // A reachable box
        }
    }
}

// Move the box and update the current connected component
State State::push(int box_id, Direction dir) {
    Position& box = available_boxes[box_id];
    Position new_box = box + dir;
    Map new_boxes = boxes;

    reset_pos(new_boxes, box);
    set_pos(new_boxes, new_box);

    Position new_player = box;
    return State(new_player, new_boxes);
}

Direction State::available_directions(int box_id) {
    // Check the four directions and return the first available one
    return LEFT;  // Placeholder
}

// class Game implementation

State Game::load(char* sample_filepath) {
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
            else if (cell == 'x' || cell == 'X')
                boxes.set(pos.to_index());
            else if (cell == 'o' || cell == 'O')
                player = pos;
        }
    }

    return State(player, boxes);
}

Game::Game() {}
