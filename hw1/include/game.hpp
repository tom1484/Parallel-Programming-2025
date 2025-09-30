#ifndef GAME_HPP
#define GAME_HPP

#include <bitset>
#include <cstdint>
#include <vector>

#define MAX_SIZE 256
// WARNING: Not sure if max width/height don't exceed 16
#define EDGE_BITS 4

using namespace std;

typedef bitset<MAX_SIZE> Map;

class Position;
class State;
class Game;

// enum Direction

enum Direction {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
};

const Direction DIRECTIONS[4] = {Direction::LEFT, Direction::RIGHT, Direction::UP, Direction::DOWN};

typedef struct DirectionDelta {
    int8_t dx, dy;
} DirectionDelta;

const DirectionDelta DIRECTION_DELTAS[4] = {
    {-1, 0},  // Left
    {1, 0},   // Right
    {0, -1},  // Up
    {0, 1},   // Down
};

inline Direction dir_inv(const Direction& dir) { return static_cast<Direction>(dir ^ 1); }

inline DirectionDelta dir_to_delta(const Direction& dir) { return DIRECTION_DELTAS[dir]; }

inline char dir_to_str(const Direction& dir) {
    switch (dir) {
        case Direction::LEFT:
            return 'A';
        case Direction::RIGHT:
            return 'D';
        case Direction::UP:
            return 'W';
        case Direction::DOWN:
            return 'S';
        default:
            return 'X';
    }
}

// class Position

class Position {
   public:
    uint8_t x, y;

    Position();
    Position(uint8_t x, uint8_t y);

    uint16_t to_index() const;
    bool is_dead_corner(const Map& boxes) const;
    bool is_dead_wall(const Map& boxes) const;
    bool is_dead_pos(const Map& boxes, bool advanced = false) const;

    Position operator+(const Direction& dir) const;
    Position operator-(const Direction& dir) const;
    bool operator==(const Position& other) const;
    bool operator<(const Position& other) const;
};

inline void set_pos(Map& bset, const Position& pos);
inline void reset_pos(Map& bset, const Position& pos);

// class State

enum StateMode {
    PUSH = 0,
    PULL = 1,
};

class State {
   public:
    Position player;  // The first grid of connected component
    Map reachable;
    Map boxes;
    vector<Position> reachable_boxes;

    bool normalized;
    bool dead;

    State();
    State(Position init_player, Map boxes);

    State push(size_t box_id, const Direction& dir) const;
    vector<Direction> available_pushes(size_t box_id) const;

    State pull(size_t box_id, const Direction& dir) const;
    vector<Direction> available_pulls(size_t box_id) const;

    void normalize(StateMode mode = StateMode::PUSH);
    uint64_t hash() const;

#ifdef DEBUG
    vector<string> map_vis;
    void calculate_map_vis();
#endif
};

// class Game

class Game {
   public:
    size_t width, height;
    Map player_map, box_map;
    Map targets;
    Map initial_boxes;  // For bi-directional BFS

    Game();

    State load(const char* sample_filepath);
    void mark_virtual_fragile_tiles();
    inline bool pos_valid(const Position& pos) const { return pos.x < width && pos.y < height; }
};

#endif  // GAME_HPP
