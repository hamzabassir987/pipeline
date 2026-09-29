#ifndef FORECAST_HPP
#define FORECAST_HPP
// ===========================================================================
// THE BOARD, AS AN MLP CAN READ IT  --  policy-visible features + melon mask
// ===========================================================================
// The global heads and the critic lost their conv trunks, so whatever they
// need to know about the boards has to arrive as numbers. This file computes
// those numbers, and the melon planting mask the filter uses, off PUBLIC tile
// state only, by the same arithmetic the referee and projection.hpp run.
//
// HARVEST HORIZONS. For each product (9: five crops, egg, milk, wool,
// fertilizer), how many units a player CAN bank:
//
//   h0  today
//   h1  today + tomorrow
//   h2  today + the next 3 days
//   h3  the lifetime: for a crop, until the plant is done (or the game ends);
//       for an animal product or fertilizer, the next 10 days
//
// "Can" is read per crop family:
//   non-ongoing  (wheat, carrot, melon) ONE harvest wipes the tile, so the
//                horizon value is the BEST single harvest inside the horizon:
//                the standing yield plus every yield water still to come up to
//                that day, counted only once the crop is ripe. A melon is
//                harvested the first day it is ripe (see melon_harvest_now),
//                so its curve stops there.
//   ongoing      (tomato, strawberry) and animals: every production night's
//                units can be banked the next morning, so the horizon value
//                is the running SUM. Taken straight from projection.hpp's
//                canonical run-off (run_plant / run_animal).
//
// Both run the canonical care policy (the must tier's waters, fed and cared
// every day, the canonical fertilize schedule where it pays), exactly like the
// critic projection, so the two never disagree about what a tile will do.
//
// MELON HAS NO BUYER. No town shop consumes it; only the town centre drains
// one unit a day. So every melon sold walks the price down for every later
// one, and the question "is one more melon worth 80 coins" depends on how
// many are already coming. melon_profit answers it: every melon standing on
// BOTH boards (the opponent's board is public) plus the own shed stock is
// sold, in harvest order, on one market with the daily drain in between;
// then new melons planted today (harvested straight away at age 10, sold the
// next morning) are appended one at a time, and each is kept only while its
// walked proceeds beat its seed (zero for a seed already owned) plus
// MELON_MIN_MARGIN. The count that survives is the filter's hard cap on the
// melon count head, and 0 masks melon out of the type slots entirely.
// ===========================================================================
#include "game.hpp"
#include "projection.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace fcast {

constexpr int N_HORIZ          = 4;
constexpr int HORIZ_DAYS[3]    = { 0, 1, 3 };  // h0..h2; h3 is the lifetime
constexpr int ANIMAL_LIFETIME  = 10;           // h3 for animal products / fertilizer
constexpr double MELON_MIN_MARGIN = 20.0;      // coins a new melon must clear above its seed

constexpr int PER_PLAYER      = NUM_PRODUCTS * (N_HORIZ + 1);            // 45
constexpr int FREE_KINDS      = 4;     // see FREE LAND below
constexpr int FREE_BANDS      = 3;     // near / mid / far, as the open-ground bands
constexpr int OWN_EXTRA       = 3 + 2 + NUM_COUNT_TYPES * 2
                              + FREE_KINDS * FREE_BANDS;                 // 33
constexpr int FCAST_FEATURES  = 2 * PER_PLAYER + OWN_EXTRA;              // 123
static_assert(FCAST_FEATURES == FCAST_DIM,
              "game.hpp's FCAST_DIM must match the forecast feature count");

// Is this product grown on a crop (lifetime = the plant's life) or produced by
// an animal (lifetime = ANIMAL_LIFETIME days)?
inline bool crop_product(int k) { return k >= WHEAT && k < WHEAT + NUM_CROPS; }

// ---------------------------------------------------------------------------
// NON-ONGOING CROPS: units a harvest ON day d would bank, for every d.
// ---------------------------------------------------------------------------
// out[d] = 0 when the crop is not ripe (or no longer exists) that day. Mirrors
// run_plant's non-ongoing branch, but does NOT stop when the crop fills up --
// a full crop can still be harvested on any later day of its window -- except
// for melon, which is harvested the first ripe day.
inline void nonongoing_curve(tile t, int day, int turn0, double out[NUM_DAYS]) {
    for (int d = 0; d < NUM_DAYS; ++d) out[d] = 0.0;
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return;
    const CropDef& cd = CROPS[t.crop];
    if (cd.ongoing) return;

    for (int d = day; d <= proj::LAST; ++d) {
        const int age = d - t.phase;
        if (d == day) {
            const int lost = proj::decay_before(t, turn0, proj::ARRIVE_HOUR);
            if (lost > 0) {
                t.yield_units -= lost;
                if (t.yield_units <= 0) return;
            }
        }
        if (fert_day_for(t.crop, t, d) && fertilize_pays(t, d))
            t.fertilized = std::max(t.fertilized, d + 2);

        bool water;
        if (d == proj::LAST)
            water = age >= cd.first_yield_day && water_yields_now(t, d);
        else
            water = plant_has_future(t, d) || water_yields_now(t, d);
        if (water && !t.watered_today) {
            t.watered_today = 1;
            if (age >= (cd.max_yield_day + 1) / 2 && age <= cd.max_yield_day) {
                const int bonus = (t.fertilized >= d) ? 2 : 1;
                t.yield_units = std::min(cd.max_yield, t.yield_units + bonus);
            }
        }
        if (age >= cd.first_yield_day) {
            out[d] = t.yield_units;
            if (t.crop == MELON) return;            // harvested straight away
        }
        if (age >= cd.max_yield_day || d == proj::LAST) return;

        const bool was = t.watered_today != 0;
        t.consecutive_unwatered = was ? 0 : t.consecutive_unwatered + 1;
        t.watered_today = 0;
        if (t.consecutive_unwatered >= 2) return;   // weeds
    }
}

// A harvest the run-off makes on day h lands in slot h+1 (LATE on the last day).
inline double stream_on_day(const proj::Stream& s, int k, int h) {
    if (h < 0 || h > proj::LAST) return 0.0;
    return s.arrive[h < proj::LAST ? h + 1 : proj::LATE][k];
}

// Per-tile curves. Non-ongoing tiles are kept separately so the horizon can
// take each tile's BEST day (a max per tile, then the sum over tiles).
struct Board {
    std::vector<std::array<double, NUM_DAYS>> non_tiles[NUM_CROPS];
    proj::Stream run;
};

inline void build_board(const player_state& ps, int day, int turn0, Board& b) {
    for (int c = 0; c < NUM_CROPS; ++c) b.non_tiles[c].clear();
    b.run = proj::Stream{};
    if (day < 0 || day > proj::LAST) return;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const tile& t = ps.board[y][x];
            if (!t.bought) continue;
            if (t.type == T_PLANT && t.crop >= 0 && t.crop < NUM_CROPS) {
                if (CROPS[t.crop].ongoing) {
                    b.run.w = 1.0;
                    proj::run_plant(t, day, turn0, b.run);
                } else {
                    std::array<double, NUM_DAYS> a{};
                    nonongoing_curve(t, day, turn0, a.data());
                    b.non_tiles[t.crop].push_back(a);
                }
            } else if (t.animal >= 0) {
                b.run.w = 1.0;
                proj::run_animal(t, day, b.run);
            }
        }
}

// Units of product k bankable over [day, last].
inline double units_by(const Board& b, int k, int day, int last) {
    last = std::min(last, proj::LAST);
    if (last < day) return 0.0;
    double u = 0.0;
    if (crop_product(k) && !CROPS[k].ongoing) {
        for (const auto& a : b.non_tiles[k]) {
            double best = 0.0;
            for (int d = day; d <= last; ++d) best = std::max(best, a[d]);
            u += best;
        }
        return u;
    }
    for (int d = day; d <= last; ++d) u += stream_on_day(b.run, k, d);
    return u;
}

inline int horizon_last(int k, int h, int day) {
    if (h < 3) return day + HORIZ_DAYS[h];
    return crop_product(k) ? proj::LAST : day + ANIMAL_LIFETIME;
}

// ---------------------------------------------------------------------------
// ONE NEW UNIT OF `type`, PUT DOWN TODAY: lifetime units of its product.
// ---------------------------------------------------------------------------
// The fresh tile is exactly what the filter projects after A_PLANT / A_PLACE
// (the planting-day water rides with every seed). This is what makes the
// end-of-game cutoff visible to the count heads: a strawberry on day 20 or a
// cow on day 21 simply yields less.
inline double new_unit_units(int type, int day, int turn0) {
    if (day < 0 || day > proj::LAST) return 0.0;
    tile t{};
    t.bought = 1;
    t.phase  = day;
    if (type < NUM_CROPS) {
        const int c = type;
        if (day + CROPS[c].first_yield_day > PLANT_HORIZON_DAY - 1) return 0.0;
        t.type = T_PLANT;
        t.crop = c;
        t.consecutive_unwatered = 1;
        t.watered_today = 1;
        t.yield_units = CROPS[c].ongoing ? 0 : 1;
        t.max_lifespan_step = CROPS[c].ongoing
            ? -1 : (day + CROPS[c].max_yield_day + 1) * TURNS_PER_DAY;
        if (!CROPS[c].ongoing) {
            double a[NUM_DAYS];
            nonongoing_curve(t, day, turn0, a);
            double best = 0.0;
            for (int d = day; d <= proj::LAST; ++d) best = std::max(best, a[d]);
            return best;
        }
        proj::Stream s;
        proj::run_plant(t, day, turn0, s);
        double u = 0.0;
        for (int d = day; d <= proj::LAST; ++d) u += stream_on_day(s, c, d);
        return u;
    }
    const int a = type - NUM_CROPS;
    if (!animal_in_horizon(a, day)) return 0.0;
    t.type   = ANIMALS[a].structure;
    t.animal = a;
    proj::Stream s;
    proj::run_animal(t, day, s);
    double u = 0.0;
    for (int d = day; d <= proj::LAST; ++d) u += stream_on_day(s, ANIMALS[a].product, d);
    return u;
}

inline int type_product(int type) {
    return type < NUM_CROPS ? type : ANIMALS[type - NUM_CROPS].product;
}
inline double type_cost(int type) {
    return type < NUM_CROPS ? (double)CROPS[type].seed_cost
                            : (double)ANIMALS[type - NUM_CROPS].cost;
}

// ===========================================================================
// THE MELON MASK
// ===========================================================================
struct MelonMarket {
    int    max_new = 0;         // how many melons planted today still pay
    double next_margin = 0.0;   // proceeds - cost of the FIRST new melon (<0: none pay)
    double units_each = 0.0;    // what one melon planted today yields
};

inline MelonMarket melon_profit(const player_state players[NUM_PLAYERS], int me,
                                const market_state& mk,
                                const std::vector<int>& shops,
                                int day, int turn0, int limit) {
    MelonMarket r;
    if (day < 0 || day > proj::LAST) return r;

    // ---- what one new melon does, and when it sells ----------------------
    int new_harvest = -1;
    {
        if (day + CROPS[MELON].first_yield_day > PLANT_HORIZON_DAY - 1) return r;
        tile t{};
        t.bought = 1; t.type = T_PLANT; t.crop = MELON; t.phase = day;
        t.consecutive_unwatered = 1; t.watered_today = 1; t.yield_units = 1;
        t.max_lifespan_step = (day + CROPS[MELON].max_yield_day + 1) * TURNS_PER_DAY;
        double a[NUM_DAYS];
        nonongoing_curve(t, day, turn0, a);
        for (int d = day; d <= proj::LAST; ++d)
            if (a[d] > 0.0) { new_harvest = d; r.units_each = a[d]; break; }
    }
    if (new_harvest < 0 || r.units_each <= 0.0) return r;

    // Sale day of a harvest on day h: the next morning, or the liquidation.
    // Index by slot so a LAST-day harvest (sold mid-day) comes after LAST's
    // morning.
    auto sale_slot = [](int h) { return h < proj::LAST ? h + 1 : proj::LATE; };

    // ---- the melons already coming, both boards -------------------------
    double arrive[proj::SLOTS] = {};
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const player_state& ps = players[p];
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x) {
                const tile& t = ps.board[y][x];
                if (!t.bought || t.type != T_PLANT || t.crop != MELON) continue;
                double a[NUM_DAYS];
                nonongoing_curve(t, day, turn0, a);
                for (int d = day; d <= proj::LAST; ++d)
                    if (a[d] > 0.0) { arrive[sale_slot(d)] += a[d]; break; }
            }
    }
    // Own stock (the opponent's shed is not public): sold this morning.
    {
        int stock = players[me].shed[MELON];
        for (const unit_inv& u : players[me].carried) stock += u[MELON];
        arrive[day] += stock;
    }

    // ---- the daily drain on melon ----------------------------------------
    double drain_day = 1.0;                         // the town centre
    for (int sh : shops) {
        const double mult = (SHOP_PRODUCTS[sh].size() == 1) ? 2.0 : 1.0;
        for (int item : SHOP_PRODUCTS[sh])
            if (item == MELON) drain_day += mult * (TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL);
    }

    double inv = mk.inventory[MELON];
    auto sell = [&](double n) -> double {
        double got = 0.0;
        const int whole = (int)std::lround(n);
        for (int j = 0; j < whole; ++j) {
            const int price = market_price(MELON, (int)std::lround(inv));
            got += price;
            if (price > PRICE_FLOOR) inv += 1.0;
        }
        return got;
    };

    const int new_slot = sale_slot(new_harvest);
    for (int D = day; D <= proj::LAST; ++D) {
        sell(arrive[D]);
        if (D == new_slot) break;
        inv -= drain_day;                           // the rest of D, overnight
    }
    if (new_slot == proj::LATE) sell(arrive[proj::LATE]);

    // ---- append new melons while each one pays ----------------------------
    const int seeds_owned = players[me].seeds[MELON];
    for (int i = 0; i < limit; ++i) {
        const double cost = (i < seeds_owned ? 0.0 : (double)CROPS[MELON].seed_cost)
                          + MELON_MIN_MARGIN;
        const double got = sell(r.units_each);
        if (i == 0) r.next_margin = got - cost;
        if (got < cost) break;
        ++r.max_new;
    }
    return r;
}

// ===========================================================================
// FREE LAND  --  what comes free today and tomorrow, beyond the open ground
// ===========================================================================
// The open-ground bands count ground a seed could go in at hour 0. The type
// slots and count heads also need the ground the day ITSELF frees, and they
// read the board only through this block. Four kinds, on the own board:
//
//   FK_FORCED_TODAY  a plant with a FORCED harvest today (must_harvest). Every
//                    such plant is either a wiping crop (wheat / carrot /
//                    melon: A_HARVEST empties the tile) or doomed (an ongoing
//                    crop rotting from tomorrow: HARVEST, DIG, PLANT). Either
//                    way the filter may replant it THE SAME DAY.
//   FK_OPTIONAL_TODAY a ripe wiping crop whose harvest is NOT forced: freed
//                    today only if the harvest head takes it.
//   FK_WEED_BY_MORNING plant_is_doomed: a weed by morning whatever the plan
//                    does. Overlaps FK_FORCED_TODAY (a doomed plant still
//                    holding units) and the open band (one holding nothing).
//   FK_FORCED_TOMORROW not forced today, but forced tomorrow: a wiping crop
//                    whose harvest window closes tomorrow (or a melon ripe
//                    tomorrow), or an ongoing crop whose LAST production night
//                    is tonight. Assumes the canonical care keeps it alive.
//
// Bands by walking distance to the shed block, exactly as the open ground.
enum FreeKind { FK_FORCED_TODAY = 0, FK_OPTIONAL_TODAY, FK_WEED_BY_MORNING,
                FK_FORCED_TOMORROW };

inline int walk_band(int x, int y) {
    const int h = BOARD_SIZE / 2;
    const int dx = (x < h - 1) ? (h - 1 - x) : (x > h ? x - h : 0);
    const int dy = (y < h - 1) ? (h - 1 - y) : (y > h ? y - h : 0);
    const int walk = dx + dy;
    return walk <= 2 ? 0 : walk <= 4 ? 1 : 2;
}

// Is this plant's harvest forced TOMORROW (and not today)?
inline bool forced_tomorrow_plant(const tile& t, int day, int turn0) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    if (must_harvest(t, day, turn0) || plant_is_doomed(t, day, turn0)) return false;
    if (day + 1 > NUM_DAYS - 1) return false;
    const CropDef& cd = CROPS[t.crop];
    const int age1 = day + 1 - t.phase;                 // tomorrow's age
    if (!cd.ongoing) {
        if (age1 < cd.first_yield_day) return false;    // not ripe tomorrow
        // A melon is forced the first ripe day; wheat / carrot the day
        // max_lifespan_step comes within a day, i.e. tomorrow's age reaches
        // max_yield_day.
        return t.crop == MELON || age1 >= cd.max_yield_day;
    }
    // Ongoing: tonight is the last production night, so tomorrow it is doomed
    // and still holds that night's unit.
    if (t.max_lifespan_step >= 0 || !plant_produces_tonight(t, day)) return false;
    const int since = (day + 1) - t.phase - cd.first_yield_day;
    return since / cd.interval + 1 == cd.max_yield;
}

inline void free_land(const player_state& ps, int day, int turn0,
                      float out[FREE_KINDS][FREE_BANDS]) {
    for (int k = 0; k < FREE_KINDS; ++k)
        for (int b = 0; b < FREE_BANDS; ++b) out[k][b] = 0.0f;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const tile& t = ps.board[y][x];
            if (!t.bought || t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) continue;
            const int b = walk_band(x, y);
            const bool forced = must_harvest(t, day, turn0);
            if (forced) out[FK_FORCED_TODAY][b] += 1.0f;
            else if (!CROPS[t.crop].ongoing && harvest_ready(t, day))
                out[FK_OPTIONAL_TODAY][b] += 1.0f;
            if (plant_is_doomed(t, day, turn0)) out[FK_WEED_BY_MORNING][b] += 1.0f;
            if (forced_tomorrow_plant(t, day, turn0)) out[FK_FORCED_TOMORROW][b] += 1.0f;
        }
}

// ===========================================================================
// FEATURES  --  FCAST_FEATURES floats, own player first
// ===========================================================================
// Split in two so cell_to_input can share work between the two viewers of one
// game. Shared holds everything that does not depend on who is looking: each
// board's 45-float block (it is the same numbers whether that board is "own"
// or "opponent") and the new-unit yields. write_features_shared then writes
// the viewer's order plus the viewer-only parts. The floats are produced by
// exactly the expressions the one-shot version used, so the output is
// bit-identical.
struct Shared {
    float  side[NUM_PLAYERS][PER_PLAYER];
    double units[NUM_COUNT_TYPES];
};

inline void build_shared(const player_state players[NUM_PLAYERS],
                         const market_state& mk, int day, int turn0, Shared& sh) {
    Board b;
    // ---- per player: 9 products x (4 horizons + lifetime value) ----------
    for (int side = 0; side < NUM_PLAYERS; ++side) {
        build_board(players[side], day, turn0, b);
        int w = 0;
        for (int k = 0; k < NUM_PRODUCTS; ++k) {
            for (int h = 0; h < N_HORIZ; ++h) {
                const double u = units_by(b, k, day, horizon_last(k, h, day));
                sh.side[side][w++] = (float)(std::log1p(u) / 5.0);
            }
            const double life = units_by(b, k, day, horizon_last(k, 3, day));
            sh.side[side][w++] = (float)std::clamp(life * mk.prices[k] / 10000.0, 0.0, 4.0);
        }
    }
    for (int k = 0; k < NUM_COUNT_TYPES; ++k) sh.units[k] = new_unit_units(k, day, turn0);
}

template <class WriteFn>
inline void write_features_shared(const Shared& sh,
                                  const player_state players[NUM_PLAYERS], int me,
                                  const market_state& mk, const std::vector<int>& shops,
                                  int day, int turn0, WriteFn&& W) {
    const int op = 1 - me;

    for (int side : { me, op })
        for (int i = 0; i < PER_PLAYER; ++i) W(sh.side[side][i]);

    // ---- own: open ground by walking distance to the shed ----------------
    // Ground a seed could go in today: owned, and empty, a weed, a bare
    // structure, or a doomed plant with nothing left on it.
    {
        const player_state& ps = players[me];
        float band[3] = { 0, 0, 0 };
        const int h = BOARD_SIZE / 2;
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x) {
                const tile& t = ps.board[y][x];
                if (!t.bought || t.animal >= 0) continue;
                const bool open = t.type == T_EMPTY || t.type == T_WEED ||
                                  t.type == T_COOP || t.type == T_PASTURE ||
                                  (plant_is_doomed(t, day, turn0) && !harvest_ready(t, day));
                if (!open) continue;
                const int dx = (x < h - 1) ? (h - 1 - x) : (x > h ? x - h : 0);
                const int dy = (y < h - 1) ? (h - 1 - y) : (y > h ? y - h : 0);
                const int walk = dx + dy;
                band[walk <= 2 ? 0 : walk <= 4 ? 1 : 2] += 1.0f;
            }
        for (float v : band) W(v / 25.0f);
    }

    // ---- own: the melon market --------------------------------------------
    {
        const MelonMarket mm = melon_profit(players, me, mk, shops, day, turn0,
                                            MACRO_PLANT_MAX);
        W(mm.max_new / (float)MACRO_PLANT_MAX);
        W((float)std::clamp(mm.next_margin / 1500.0, -1.0, 1.0));
    }

    // ---- own: what ONE unit of each type, put down today, still yields ----
    for (int k = 0; k < NUM_COUNT_TYPES; ++k)
        W((float)(std::log1p(sh.units[k]) / 5.0));
    for (int k = 0; k < NUM_COUNT_TYPES; ++k) {
        const double net = sh.units[k] * mk.prices[type_product(k)] - type_cost(k);
        W((float)std::clamp(net / 2000.0, -1.0, 4.0));
    }

    // ---- own: land coming free (APPENDED LAST, so older checkpoints keep
    //      every earlier column where it was) ------------------------------
    {
        float fl[FREE_KINDS][FREE_BANDS];
        free_land(players[me], day, turn0, fl);
        for (int k = 0; k < FREE_KINDS; ++k)
            for (int b = 0; b < FREE_BANDS; ++b) W(fl[k][b] / 25.0f);
    }
}

// The one-shot form, for a single viewer.
template <class WriteFn>
inline void write_features(const player_state players[NUM_PLAYERS], int me,
                           const market_state& mk, const std::vector<int>& shops,
                           int day, int turn0, WriteFn&& W) {
    Shared sh;
    build_shared(players, mk, day, turn0, sh);
    write_features_shared(sh, players, me, mk, shops, day, turn0, W);
}

}  // namespace fcast

#endif  // FORECAST_HPP