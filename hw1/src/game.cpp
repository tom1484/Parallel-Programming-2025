#include "game.hpp"

#include <fstream>
#include <functional>
#include <queue>
#include <string>
#include <vector>

using namespace std;

// Global variables

Game game;

// class Position implementation

Position::Position() : x(0), y(0) {}

Position::Position(uint8_t x, uint8_t y) : x(x), y(y) {}

// NOTE: If max width/height don't exceed 16, we can use EDGE_BITS to optimize
uint16_t Position::to_index() const { return (y * game.width) + x; }

bool Position::is_dead_corner(const Map& boxes) const {
    static const Direction corner_dirs[4][2] = {
        {LEFT, UP},
        {UP, RIGHT},
        {RIGHT, DOWN},
        {DOWN, LEFT},
    };

    if (game.targets[to_index()]) return false;  // Target is never dead

    for (const Direction* comb : corner_dirs) {
        Position neighbor0 = *this + comb[0];
        Position neighbor1 = *this + comb[1];
        Position corner = *this + comb[0] + comb[1];
        if (game.player_map[neighbor0.to_index()] && game.player_map[neighbor1.to_index()]) return true;
        if (boxes[neighbor0.to_index()] && boxes[neighbor1.to_index()] && game.player_map[corner.to_index()])
            return true;
    }

    return false;
}

bool Position::is_dead_wall() const {
    static const Direction wall_dirs[4][3] = {
        {LEFT, UP, DOWN},
        {UP, LEFT, RIGHT},
        {RIGHT, UP, DOWN},
        {DOWN, LEFT, RIGHT},
    };

    for (const Direction* comb : wall_dirs) {
        Position wall_pivot = *this + comb[0];
        if (!game.box_map[wall_pivot.to_index()]) continue;

        bool escaped = false;
        for (int i = 1; i < 3; i++) {
            Direction dir = comb[i];
            Position self = *this;
            Position wall = wall_pivot;
            do {
                // Case 1: # x   #
                //          ####
                // Case 2: # x   #
                //          ####@
                // Case 3: # x . #
                //          #####
                if (!game.box_map[wall.to_index()] || !game.player_map[wall.to_index()] ||
                    game.targets[self.to_index()]) {
                    escaped = true;
                    break;
                }
                self = self + dir;
                wall = wall + dir;
            } while (game.pos_valid(self) && game.pos_valid(wall) && !game.box_map[self.to_index()]);

            if (escaped) break;
        }
        if (!escaped) return true;
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

#ifdef DEBUG
void State::calculate_map_vis() {
    map_vis.resize(game.height, string(game.width, ' '));
    for (size_t y = 0; y < game.height; ++y) {
        for (size_t x = 0; x < game.width; ++x) {
            Position pos(x, y);
            uint16_t idx = pos.to_index();
            if (game.player_map[idx] && game.box_map[idx]) {
                map_vis[y][x] = '#';
            } else if (!game.player_map[idx] && game.box_map[idx]) {
                map_vis[y][x] = '@';
            } else if (boxes[idx]) {
                if (game.targets[idx])
                    map_vis[y][x] = 'X';
                else
                    map_vis[y][x] = 'x';
            } else if (game.targets[idx]) {
                map_vis[y][x] = '.';
            } else if (reachable[idx]) {
                map_vis[y][x] = '-';
            } 
            if (pos == player && !game.player_map[idx]) {
                if (game.targets[idx])
                    map_vis[y][x] = 'O';
                else
                    map_vis[y][x] = 'o';
            }
        }
    }
}
#endif

State State::push(size_t box_id, const Direction& dir) const {
    const Position& box = reachable_boxes[box_id];
    Position new_box = box + dir;
    Map new_boxes = boxes;

    reset_pos(new_boxes, box);
    set_pos(new_boxes, new_box);

    Position new_player = box;
    return State(new_player, new_boxes);
}

vector<Direction> State::available_pushes(size_t box_id) const {
    Map box_block = game.box_map | boxes;
    vector<Direction> pushes;

    for (Direction dir : DIRECTIONS) {  // The push direction
        const Position& box = reachable_boxes[box_id];
        Position player_pos = box - dir;
        Position new_box = box + dir;

        if (!game.pos_valid(new_box) || !game.pos_valid(player_pos)) continue;
        if (!reachable[player_pos.to_index()] || box_block[new_box.to_index()]) continue;

        pushes.push_back(dir);
    }

    return pushes;
}

State State::pull(size_t box_id, const Direction& dir) const {
    const Position& box = reachable_boxes[box_id];
    Position new_box = box + dir;
    Map new_boxes = boxes;

    reset_pos(new_boxes, box);
    set_pos(new_boxes, new_box);

    Position new_player = new_box + dir;
    return State(new_player, new_boxes);
}

vector<Direction> State::available_pulls(size_t box_id) const {
    Map player_block = game.player_map | boxes;
    Map box_block = game.box_map | boxes;
    vector<Direction> pulls;

    for (Direction dir : DIRECTIONS) {  // The pull direction
        const Position& box = reachable_boxes[box_id];
        Position player_pos = box + dir;
        Position new_box = player_pos;
        Position new_player = player_pos + dir;

        if (!game.pos_valid(new_box) || !game.pos_valid(new_player) || !game.pos_valid(player_pos)) continue;
        if (!reachable[player_pos.to_index()] || box_block[new_box.to_index()] || player_block[new_player.to_index()])
            continue;

        pulls.push_back(dir);
    }

    return pulls;
}

void State::reset() {
    reachable.reset();
    reachable_boxes.clear();
    normalized = false;
    dead = false;
#ifdef DEBUG
    map_vis.clear();
#endif
}

void State::normalize(Mode mode) {
    Map player_block = game.player_map | boxes;
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
                if (!player_block[idx])  // Empty space
                    q.push(next);
                if (boxes[idx]) {
                    // The dead case of pulling is nearly impossible, so we only check the dead case of pushing here
                    if (mode == FORWARD && next.is_dead_corner(player_block)) {
                        normalized = false;
                        dead = true;
                        return;
                    }
                    reachable_boxes.push_back(next);  // A reachable box
                    visited.set(idx);
                }
            }
        }
    }

    normalized = true;
    dead = false;

#ifdef DEBUG
    calculate_map_vis();
#endif
}

uint64_t State::hash() const {
    // Combine player position into a single value
    uint64_t player_hash = static_cast<uint64_t>(player.to_index());

    // Hash the boxes map by converting to string and using std::hash
    string boxes_str = boxes.to_string();
    uint64_t boxes_hash = std::hash<string>{}(boxes_str);

    // Combine the two hashes using bitwise operations
    // Use different bit positions to avoid collisions
    // TODO: Update the shifting value
    return (player_hash << 16) ^ boxes_hash;
}

// class Game implementation

Game::Game() {}

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

    player_map.reset();
    box_map.reset();
    targets.reset();
    initial_boxes.reset();

    target_list.clear();
    initial_boxes_list.clear();

    Position pos;
    for (pos.y = 0; pos.y < height; pos.y++) {
        string& row = lines[pos.y];
        if (row.size() != width) throw runtime_error("Inconsistent row width");

        for (pos.x = 0; pos.x < width; pos.x++) {
            char cell = row[pos.x];
            if (cell == '#') {
                player_map.set(pos.to_index());
                box_map.set(pos.to_index());
            } else if (cell == '@') {  // Fragile tile: only boxes will be blocked
                box_map.set(pos.to_index());
            } else if (cell == '!') {  // Player stepping on a fragile tile
                player = pos;
                box_map.set(pos.to_index());
            } else if (cell == '.') {  // Target position
                targets.set(pos.to_index());
                target_list.push_back(pos);
            } else if (cell == 'x') {  // Box
                initial_boxes.set(pos.to_index());
                initial_boxes_list.push_back(pos);
            } else if (cell == 'X') {  // Box on target
                initial_boxes.set(pos.to_index());
                initial_boxes_list.push_back(pos);
                targets.set(pos.to_index());
                target_list.push_back(pos);
            } else if (cell == 'o') {  // Player
                player = pos;
            } else if (cell == 'O') {  // Player on target
                player = pos;
                targets.set(pos.to_index());
                target_list.push_back(pos);
            }
        }
    }

    return State(player, initial_boxes);
}

void Game::mark_virtual_fragile_tiles() {
    Map new_box_map = box_map;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            Position pos(x, y);
            uint16_t idx = pos.to_index();
            if (player_map[idx] || box_map[idx]) continue;
            if (pos.is_dead_wall() || pos.is_dead_corner()) new_box_map.set(idx);
        }
    }

    box_map = new_box_map;
}
