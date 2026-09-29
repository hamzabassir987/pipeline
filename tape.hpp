#ifndef TAPE_HPP
#define TAPE_HPP
// ===========================================================================
// TAPES  --  recorded Kaggle players as training opponents
// ===========================================================================
// A tape is ONE SIDE of a Kaggle episode JSON, replayed open-loop into the
// opponent's seat of a live training game:
//
//   * its unit and market actions are applied turn by turn, exactly as
//     recorded;
//   * it sits in a GHOST seat (simulation::ghost): buys, hires and land never
//     fail for money, so the recorded plan cannot be derailed by the prices
//     our policy's selling produced. Money may go negative, and the money it
//     ends with is what the recorded plan earned IN OUR MARKET -- which is
//     what check_winner compares;
//   * its weeds and the town's shop unlocks are adopted from the recording
//     every turn (the same sync replay.cpp does), so a weed the recorded
//     player dug exists here too and the town drains what it drained there.
//     The live player's weeds stay random.
//
// Episodes are parsed once and cached in a compact form (8 bytes a move, one
// 100-bit weed mask per step). The directory is re-listed every iteration, so
// JSONs dropped in while training runs are picked up.
//
// ALIGNMENT. Kaggle stores steps[t].action either as the action that produced
// steps[t].observation (POST, kaggle_environments' convention) or the one
// taken from it (PRE). replay.cpp detects it by trial; so does this, on the
// first PROBE_STEPS steps, by replaying BOTH recorded sides with no ghosts and
// counting money / crew disagreements.
//
//   sim turn T applies  steps[T+1].action  (POST)   steps[T].action  (PRE)
//   and syncs from      steps[T].observation         (both)
// ===========================================================================
#include "game.hpp"
#include "json.hpp"

#include <array>
#include <bitset>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>

namespace tape {

using kjson::JValue;

// ---------------------------------------------------------------------------
// Decoding (the same tables as replay.cpp, which cannot be linked: it owns a
// main()).
// ---------------------------------------------------------------------------
namespace detail {

inline const char* const* item_names() {
    static const char* N[NUM_ITEMS] = {
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP" };
    return N;
}
inline int item_id(const std::string& s) {
    const char* const* N = item_names();
    for (int i = 0; i < NUM_ITEMS; ++i) if (s == N[i]) return i;
    return -1;
}
inline int crop_id(const std::string& s) {
    const int i = item_id(s);
    return (i >= 0 && i < NUM_CROPS) ? i : -1;
}
inline int animal_id(const std::string& s) {
    const int i = item_id(s);
    return (i >= GOOSE && i < GOOSE + NUM_ANIMALS) ? i - GOOSE : -1;
}

inline bool decode_unit(const JValue& a, int unit, move& out) {
    if (!a.is_arr() || a.size() == 0) return false;
    const std::string op  = a[0].as_str();
    const std::string arg = a.size() > 1 ? a[1].as_str() : "";
    const int n = a.size() > 2 ? a[2].as_int(1) : 1;

    if (op == "NORTH") { out = act(unit, A_MOVE_NORTH); return true; }
    if (op == "SOUTH") { out = act(unit, A_MOVE_SOUTH); return true; }
    if (op == "EAST")  { out = act(unit, A_MOVE_EAST);  return true; }
    if (op == "WEST")  { out = act(unit, A_MOVE_WEST);  return true; }
    if (op == "PASS")  { out = act(unit, A_PASS);       return true; }
    if (op == "WATER") { out = act(unit, A_WATER);      return true; }
    if (op == "HARVEST")   { out = act(unit, A_HARVEST);   return true; }
    if (op == "FERTILIZE") { out = act(unit, A_FERTILIZE); return true; }
    if (op == "DIG")   { out = act(unit, A_DIG);   return true; }
    if (op == "FEED")  { out = act(unit, A_FEED);  return true; }
    if (op == "CARE")  { out = act(unit, A_CARE);  return true; }
    if (op == "COLLECT_FERTILIZER") { out = act(unit, A_COLLECT_FERTILIZER); return true; }
    if (op == "BUILD_COOP")    { out = act(unit, A_BUILD_COOP);    return true; }
    if (op == "BUILD_PASTURE") { out = act(unit, A_BUILD_PASTURE); return true; }
    if (op == "DROP")   { out = act_drop_all(unit); return true; }
    if (op == "PICKUP") { out = act_pickup(unit, item_id(arg), n); return true; }
    if (op == "PLANT")  { out = act_plant(unit, crop_id(arg)); return true; }
    if (op == "PLACE")  { out = act_place_item(unit, item_id(arg)); out.count = n; return true; }
    return false;
}

inline bool decode_market(const JValue& a, move& out) {
    if (!a.is_arr() || a.size() == 0) return false;
    const std::string op  = a[0].as_str();
    const std::string arg = a.size() > 1 ? a[1].as_str() : "";
    const int n = a.size() > 2 ? a[2].as_int(1) : 1;

    if (op == "HIRE")        { out = order_hire();     return true; }
    if (op == "BUY_LAND")    { out = order_buy_land(); return true; }
    if (op == "BUY_SEED")    { out = order_buy_seed(crop_id(arg), n); return true; }
    if (op == "BUY_ANIMAL")  { out = order_buy_animal(animal_id(arg), n); return true; }
    if (op == "BUY_PRODUCT") { out = order_buy_product(item_id(arg), n); return true; }
    if (op == "SELL")        { out = order_sell(item_id(arg), n); return true; }
    return false;
}

inline bool is_weed(const JValue& t) {
    if (!t.is_obj()) return false;
    const JValue* k = t.find("kind");
    return k && k->as_str() == "WEED";
}

}  // namespace detail

// ---------------------------------------------------------------------------
// The compact episode
// ---------------------------------------------------------------------------
struct TMove {                     // 8 bytes
    int8_t  idx;                   // unit index, or MARKET_ACTOR
    int8_t  type;
    int8_t  item;
    int32_t count;
};

struct Episode {
    std::string path;
    bool post = true;              // alignment, see the banner
    int  n_steps = 0;

    // Per recorded side: step s's moves are mv[side][off[side][s] .. off[side][s+1]).
    std::vector<uint32_t> off[NUM_PLAYERS];
    std::vector<TMove>    mv[NUM_PLAYERS];

    // Per step: that side's recorded weeds, and the town's shops.
    std::vector<std::bitset<CELLS>> weeds[NUM_PLAYERS];
    std::vector<uint8_t> has_farms;          // was a farms block recorded at s
    std::vector<std::array<int8_t, MAX_SHOP_INSTANCES>> shops;
    std::vector<int8_t>  n_shops;            // -1 == not recorded at s

    bool   ok_side[NUM_PLAYERS] = { false, false };
    double final_money[NUM_PLAYERS] = { 0.0, 0.0 };
    int    winner = -1;                      // by recorded final money, -1 on a tie

    int action_step(int turn) const { return post ? turn + 1 : turn; }
};

// Moves the recorded `side` makes on sim turn `turn`, re-addressed to seat
// `seat` of our sim (only the market actor id is seat-independent; unit
// indices are the side's own, which is what our seat's units are too).
inline void moves_for(const Episode& e, int side, int turn, std::vector<move>& out) {
    const int s = e.action_step(turn);
    if (s < 0 || s >= e.n_steps) return;
    const std::vector<uint32_t>& off = e.off[side];
    for (uint32_t k = off[s]; k < off[s + 1]; ++k) {
        const TMove& t = e.mv[side][k];
        out.push_back({ (int)t.idx, (int)t.type, (int)t.item, (int)t.count });
    }
}

// Adopt the recording's weeds on `seat`'s board and the town's shops, for the
// state at sim turn `turn`. Call BEFORE apply_turn.
inline void sync(const Episode& e, int side, int turn, simulation& sim, int seat) {
    if (turn < 0 || turn >= e.n_steps) return;
    if (e.n_shops[turn] >= 0) {
        sim.unlocked_shops.clear();
        for (int k = 0; k < e.n_shops[turn]; ++k)
            sim.unlocked_shops.push_back(e.shops[turn][k]);
    }
    if (!e.has_farms[turn]) return;
    const std::bitset<CELLS>& w = e.weeds[side][turn];
    player_state& ps = sim.players[seat];
    for (int c = 0; c < CELLS; ++c) {
        tile& t = ps.board[c / BOARD_SIZE][c % BOARD_SIZE];
        if (w[c]) {
            if (t.type == T_EMPTY && t.bought) t = tile{ T_WEED, 1 };
        } else if (t.type == T_WEED) {
            t = tile{ T_EMPTY, 1 };
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
namespace detail {

struct Recorded {                      // probe-only, discarded after load
    std::vector<double> money[NUM_PLAYERS];
    std::vector<int>    hands[NUM_PLAYERS];
};

// Replay BOTH sides under one alignment for `k` turns and count how often the
// sim disagrees with the recorded money / crew size.
inline long probe(Episode& e, const Recorded& r, bool post, int k) {
    e.post = post;
    const Episode& alt = e;
    simulation sim;                // no log
    sim.weed_chance = 0.0;
    sim.auto_shops  = false;
    sim.auto_log    = false;
    long bad = 0;
    for (int T = 0; T < k && T + 1 < e.n_steps; ++T) {
        std::vector<move> mv[NUM_PLAYERS];
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            sync(alt, p, T, sim, p);
            moves_for(alt, p, T, mv[p]);
        }
        sim.apply_turn(mv);
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            const double m = r.money[p][T + 1];
            if (!std::isnan(m) && (long)m != (long)sim.players[p].money) ++bad;
            const int h = r.hands[p][T + 1];
            if (h >= 0 && h != (int)sim.players[p].farmers.size() - 1) ++bad;
        }
    }
    return bad;
}

}  // namespace detail

constexpr int PROBE_STEPS = 24;

// Parse an episode from JSON TEXT already in memory. `name` is only a label.
inline std::shared_ptr<Episode> parse_episode(const std::string& text,
                                              const std::string& name, std::string& err) {
    auto e = std::make_shared<Episode>();
    e->path = name;
    detail::Recorded rec;
    std::string last_status[NUM_PLAYERS];
    long n_actions[NUM_PLAYERS] = { 0, 0 };

    for (int p = 0; p < NUM_PLAYERS; ++p) e->off[p].push_back(0);

    try {
        kjson::stream_steps(text, [&](int, const JValue& step) {
            const int ns = step.is_arr() ? (int)step.size() : 0;

            // ---- actions -------------------------------------------------
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                if (p < ns) {
                    if (const JValue* st = step[p].find("status"))
                        last_status[p] = st->as_str();
                    const JValue* a = step[p].find("action");
                    if (a && a->is_obj()) {
                        auto push = [&](const move& m) {
                            e->mv[p].push_back({ (int8_t)m.farmer_idx, (int8_t)m.type,
                                                 (int8_t)m.item, (int32_t)m.count });
                        };
                        if (const JValue* f = a->find("farmer")) {
                            move m;
                            if (detail::decode_unit(*f, 0, m)) push(m);
                        }
                        if (const JValue* hs = a->find("hands"))
                            for (size_t h = 0; h < hs->size(); ++h) {
                                move m;
                                if (detail::decode_unit((*hs)[h], (int)h + 1, m)) push(m);
                            }
                        if (const JValue* mk = a->find("market"))
                            for (size_t o = 0; o < mk->size(); ++o) {
                                move m;
                                if (detail::decode_market((*mk)[o], m)) push(m);
                            }
                        ++n_actions[p];
                    }
                }
                e->off[p].push_back((uint32_t)e->mv[p].size());
            }

            // ---- observation: weeds, shops, money, crew -----------------
            const JValue* shared = nullptr;
            for (int p = 0; p < ns && !shared; ++p) {
                const JValue* o = step[p].find("observation");
                if (o && o->is_obj() && o->find("farms")) shared = o;
            }
            std::array<int8_t, MAX_SHOP_INSTANCES> sh{};
            int8_t n_sh = -1;
            uint8_t has_f = 0;
            std::bitset<CELLS> w[NUM_PLAYERS];
            double money[NUM_PLAYERS] = { NAN, NAN };
            int hands[NUM_PLAYERS] = { -1, -1 };

            if (shared) {
                if (const JValue* town = shared->find("town"))
                    if (const JValue* us = town->find("unlocked_shops")) {
                        static const char* SHOP_KEY[NUM_SHOPS] = {
                            "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
                            "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE" };
                        n_sh = 0;
                        for (size_t i = 0; i < us->size() && n_sh < MAX_SHOP_INSTANCES; ++i) {
                            const std::string nm = (*us)[i].as_str();
                            for (int k = 0; k < NUM_SHOPS; ++k)
                                if (nm == SHOP_KEY[k]) { sh[n_sh++] = (int8_t)k; break; }
                        }
                    }
                const JValue* farms = shared->find("farms");
                if (farms && farms->is_arr()) {
                    has_f = 1;
                    for (int p = 0; p < (int)farms->size() && p < NUM_PLAYERS; ++p) {
                        const JValue& f = (*farms)[p];
                        if (const JValue* m = f.find("money")) money[p] = m->as_num();
                        if (const JValue* h = f.find("hands")) hands[p] = (int)h->size();
                        if (const JValue* rows = f.find("tiles"))
                            for (int y = 0; y < (int)rows->size() && y < BOARD_SIZE; ++y)
                                for (int x = 0; x < (int)(*rows)[y].size() && x < BOARD_SIZE; ++x)
                                    if (detail::is_weed((*rows)[y][x]))
                                        w[p].set(y * BOARD_SIZE + x);
                    }
                }
            }
            e->shops.push_back(sh);
            e->n_shops.push_back(n_sh);
            e->has_farms.push_back(has_f);
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                e->weeds[p].push_back(w[p]);
                rec.money[p].push_back(money[p]);
                rec.hands[p].push_back(hands[p]);
            }
            ++e->n_steps;
        });
    } catch (const std::exception& ex) {
        err = ex.what();
        return nullptr;
    }

    if (e->n_steps < 2) { err = "fewer than two steps"; return nullptr; }

    // ---- which side is usable ---------------------------------------------
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const std::string& st = last_status[p];
        const bool crashed = (st == "ERROR" || st == "TIMEOUT" || st == "INVALID");
        e->ok_side[p] = !crashed && n_actions[p] > e->n_steps / 2 && !e->mv[p].empty();
        for (int s = e->n_steps - 1; s >= 0; --s)
            if (!std::isnan(rec.money[p][s])) { e->final_money[p] = rec.money[p][s]; break; }
    }
    if (e->final_money[0] > e->final_money[1]) e->winner = 0;
    else if (e->final_money[1] > e->final_money[0]) e->winner = 1;

    // ---- alignment ---------------------------------------------------------
    const long post = detail::probe(*e, rec, true,  PROBE_STEPS);
    const long pre  = detail::probe(*e, rec, false, PROBE_STEPS);
    e->post = (post <= pre);

    if (!e->ok_side[0] && !e->ok_side[1]) { err = "no usable side"; return nullptr; }
    for (int p = 0; p < NUM_PLAYERS; ++p) e->mv[p].shrink_to_fit();
    return e;
}

inline std::shared_ptr<Episode> load_json(const std::string& path, std::string& err) {
    std::string text;
    try { text = kjson::read_file(path); }
    catch (const std::exception& ex) { err = ex.what(); return nullptr; }
    return parse_episode(text, path, err);
}

// ---------------------------------------------------------------------------
// THE .tape FORMAT  --  the compact episode, on disk
// ---------------------------------------------------------------------------
// Written once by tapeconv from the 32 MB JSONs; ~100-200 KB each and loads in
// well under a millisecond. Native endianness: convert and train on the same
// kind of machine. Bump TAPE_VERSION whenever Episode changes shape, and old
// files are refused (and reported) rather than misread.
//
//   "TAPE" u32 version
//   u32 name_len, name bytes          (source zip:entry, for reports)
//   u8 post, i32 n_steps, u8 ok_side[2], f64 final_money[2], i32 winner
//   per side:  u32 n_off, u32 off[n_off], u32 n_mv, n_mv x {i8 idx,i8 type,i8 item,i32 count}
//   per side:  n_steps x 2 u64         (the weed bitsets)
//   u8 has_farms[n_steps], i8 n_shops[n_steps], i8 shops[n_steps][MAX_SHOP_INSTANCES]
// ---------------------------------------------------------------------------
constexpr uint32_t TAPE_VERSION = 1;

namespace detail {
struct Out {
    std::string buf;
    template <class T> void put(const T& v) {
        buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    void raw(const void* p, size_t n) { buf.append(static_cast<const char*>(p), n); }
};
struct In {
    const char* p; const char* end;
    template <class T> T get() {
        if ((size_t)(end - p) < sizeof(T)) throw std::runtime_error("truncated .tape");
        T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    void raw(void* dst, size_t n) {
        if ((size_t)(end - p) < n) throw std::runtime_error("truncated .tape");
        std::memcpy(dst, p, n); p += n;
    }
};
inline void put_bits(Out& o, const std::bitset<CELLS>& b) {
    uint64_t w[2] = { 0, 0 };
    for (int c = 0; c < CELLS; ++c) if (b[c]) w[c / 64] |= (uint64_t)1 << (c % 64);
    o.put(w[0]); o.put(w[1]);
}
inline std::bitset<CELLS> get_bits(In& in) {
    const uint64_t w0 = in.get<uint64_t>(), w1 = in.get<uint64_t>();
    std::bitset<CELLS> b;
    for (int c = 0; c < CELLS; ++c)
        if (((c < 64 ? w0 : w1) >> (c % 64)) & 1) b.set(c);
    return b;
}
}  // namespace detail

inline std::string serialize(const Episode& e) {
    detail::Out o;
    o.raw("TAPE", 4);
    o.put(TAPE_VERSION);
    o.put((uint32_t)e.path.size()); o.raw(e.path.data(), e.path.size());
    o.put((uint8_t)e.post); o.put((int32_t)e.n_steps);
    for (int p = 0; p < NUM_PLAYERS; ++p) o.put((uint8_t)e.ok_side[p]);
    for (int p = 0; p < NUM_PLAYERS; ++p) o.put(e.final_money[p]);
    o.put((int32_t)e.winner);
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        o.put((uint32_t)e.off[p].size());
        o.raw(e.off[p].data(), e.off[p].size() * sizeof(uint32_t));
        o.put((uint32_t)e.mv[p].size());
        for (const TMove& m : e.mv[p]) { o.put(m.idx); o.put(m.type); o.put(m.item); o.put(m.count); }
    }
    for (int p = 0; p < NUM_PLAYERS; ++p)
        for (int s = 0; s < e.n_steps; ++s) detail::put_bits(o, e.weeds[p][s]);
    o.raw(e.has_farms.data(), e.n_steps);
    o.raw(e.n_shops.data(), e.n_steps);
    for (int s = 0; s < e.n_steps; ++s) o.raw(e.shops[s].data(), MAX_SHOP_INSTANCES);
    return std::move(o.buf);
}

inline std::shared_ptr<Episode> load_tape(const std::string& path, std::string& err) {
    std::string data;
    try { data = kjson::read_file(path); }
    catch (const std::exception& ex) { err = ex.what(); return nullptr; }
    try {
        detail::In in{ data.data(), data.data() + data.size() };
        char magic[4];
        in.raw(magic, 4);
        if (std::memcmp(magic, "TAPE", 4) != 0) { err = "not a .tape file"; return nullptr; }
        const uint32_t ver = in.get<uint32_t>();
        if (ver != TAPE_VERSION) {
            err = "tape version " + std::to_string(ver) + ", expected "
                + std::to_string(TAPE_VERSION) + " -- re-run tapeconv";
            return nullptr;
        }
        auto e = std::make_shared<Episode>();
        std::string src(in.get<uint32_t>(), '\0');
        in.raw(&src[0], src.size());
        e->path = path;                 // the .tape path: that is what the library keys on
        e->post = in.get<uint8_t>() != 0;
        e->n_steps = in.get<int32_t>();
        if (e->n_steps < 2 || e->n_steps > 100000) { err = "bad step count"; return nullptr; }
        for (int p = 0; p < NUM_PLAYERS; ++p) e->ok_side[p] = in.get<uint8_t>() != 0;
        for (int p = 0; p < NUM_PLAYERS; ++p) e->final_money[p] = in.get<double>();
        e->winner = in.get<int32_t>();
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            e->off[p].resize(in.get<uint32_t>());
            if ((int)e->off[p].size() != e->n_steps + 1) { err = "bad offsets"; return nullptr; }
            in.raw(e->off[p].data(), e->off[p].size() * sizeof(uint32_t));
            e->mv[p].resize(in.get<uint32_t>());
            for (TMove& m : e->mv[p]) {
                m.idx = in.get<int8_t>(); m.type = in.get<int8_t>();
                m.item = in.get<int8_t>(); m.count = in.get<int32_t>();
            }
            if (e->off[p].back() != e->mv[p].size()) { err = "offsets past moves"; return nullptr; }
        }
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            e->weeds[p].resize(e->n_steps);
            for (int s = 0; s < e->n_steps; ++s) e->weeds[p][s] = detail::get_bits(in);
        }
        e->has_farms.resize(e->n_steps);
        in.raw(e->has_farms.data(), e->n_steps);
        e->n_shops.resize(e->n_steps);
        in.raw(e->n_shops.data(), e->n_steps);
        e->shops.resize(e->n_steps);
        for (int s = 0; s < e->n_steps; ++s) in.raw(e->shops[s].data(), MAX_SHOP_INSTANCES);
        return e;
    } catch (const std::exception& ex) {
        err = ex.what();
        return nullptr;
    }
}

// Either format, by extension.
inline std::shared_ptr<Episode> load_episode(const std::string& path, std::string& err) {
    const std::filesystem::path fp(path);
    if (fp.extension() == ".tape") return load_tape(path, err);
    return load_json(path, err);
}

// ---------------------------------------------------------------------------
// The library: directory listing, cache, sampling
// ---------------------------------------------------------------------------
struct Pick {
    std::shared_ptr<const Episode> ep;
    int side = 0;
};

// ---- a tape in a game, always in seat 1 -----------------------------------
// seat()      once, right after the simulation is constructed
// moves()     every turn, before apply_turn
// after()     every turn, after apply_turn: adopts the recording's weeds and
//             shops for the NEW turn, so the next observation already has them
constexpr int SEAT = 1;
inline void seat(simulation& s, const Pick& tp) {
    s.ghost[SEAT]  = true;        // recorded buys never fail for money
    s.auto_shops   = false;       // the town follows the recording
    sync(*tp.ep, tp.side, s.turn, s, SEAT);
}
inline void moves(const simulation& s, const Pick& tp, std::vector<move>& out) {
    moves_for(*tp.ep, tp.side, s.turn, out);
}
inline void after(simulation& s, const Pick& tp) {
    sync(*tp.ep, tp.side, s.turn, s, SEAT);
}

struct Library {
    std::string dir;
    size_t max_cache       = 4000;     // episodes kept parsed in memory
    bool   winners_only    = false;    // only the side that won its recording
    double min_final_money = -1e18;    // skip sides that finished below this

    std::vector<std::string> files;
    std::unordered_map<std::string, std::shared_ptr<const Episode>> cache;
    std::unordered_set<std::string> bad;
    std::unordered_set<std::string> exclude;   // held out: never drawn by sample()

    explicit Library(std::string d = "./tapes") : dir(std::move(d)) {}

    // Re-list the directory. Cheap; done every iteration.
    void rescan() {
        files.clear();
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) return;
        for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
            if (!de.is_regular_file()) continue;
            // .tape (from tapeconv) or raw .json. Files still being written
            // by tapeconv end in .tmp and are skipped.
            const auto ext = de.path().extension();
            if (ext != ".tape" && ext != ".json") continue;
            const std::string p = de.path().string();
            if (!bad.count(p) && !exclude.count(p)) files.push_back(p);
        }
        std::sort(files.begin(), files.end());   // deterministic before shuffling
    }

    bool side_usable(const Episode& e, int p) const {
        if (!e.ok_side[p]) return false;
        if (winners_only && e.winner != p) return false;
        return e.final_money[p] >= min_final_money;
    }

    // `n` tapes, distinct episodes while the directory has enough, then with
    // replacement. Parses what is not cached yet, in parallel.
    std::vector<Pick> sample(int n, std::mt19937& rng) {
        std::vector<Pick> out;
        rescan();
        if (n <= 0 || files.empty()) return out;

        std::vector<std::string> chosen = files;
        std::shuffle(chosen.begin(), chosen.end(), rng);
        if ((int)chosen.size() > n) chosen.resize(n);

        std::vector<std::string> todo;
        for (const auto& p : chosen) if (!cache.count(p)) todo.push_back(p);
        if (!todo.empty()) {
            std::vector<std::shared_ptr<Episode>> got(todo.size());
            std::vector<std::string> why(todo.size());
            #pragma omp parallel for schedule(dynamic, 1)
            for (int k = 0; k < (int)todo.size(); ++k)
                got[k] = load_episode(todo[k], why[k]);
            int n_bad = 0;
            for (size_t k = 0; k < todo.size(); ++k) {
                if (got[k]) cache[todo[k]] = got[k];
                else {
                    bad.insert(todo[k]);
                    if (++n_bad <= 3)
                        std::cerr << "tape: skipping " << todo[k] << ": " << why[k] << "\n";
                }
            }
            if (n_bad > 3) std::cerr << "tape: ... " << n_bad << " files skipped\n";
        }

        std::vector<Pick> pool;
        for (const auto& p : chosen) {
            auto it = cache.find(p);
            if (it == cache.end()) continue;
            const Episode& e = *it->second;
            int sides[NUM_PLAYERS], ns = 0;
            for (int s = 0; s < NUM_PLAYERS; ++s) if (side_usable(e, s)) sides[ns++] = s;
            if (ns == 0) continue;
            pool.push_back({ it->second, sides[std::uniform_int_distribution<int>(0, ns - 1)(rng)] });
        }
        if (pool.empty()) return out;

        out = pool;
        std::uniform_int_distribution<int> any(0, (int)pool.size() - 1);
        while ((int)out.size() < n) out.push_back(pool[any(rng)]);
        if ((int)out.size() > n) out.resize(n);

        evict(chosen);
        return out;
    }

private:
    void evict(const std::vector<std::string>& keep_list) {
        if (cache.size() <= max_cache) return;
        std::unordered_set<std::string> keep(keep_list.begin(), keep_list.end());
        for (auto it = cache.begin(); it != cache.end() && cache.size() > max_cache; ) {
            if (keep.count(it->first)) { ++it; continue; }
            it = cache.erase(it);      // a Pick still holding it keeps it alive
        }
    }
};

}  // namespace tape

#endif  // TAPE_HPP