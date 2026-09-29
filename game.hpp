#ifndef GAME_HPP
#define GAME_HPP

#include <torch/torch.h>
#include <torch/cuda.h>
#include <iomanip>
#include <vector>
#include <fstream>
#include <random>
#include <iostream>
#include <queue>
#include <functional>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <array>

extern std::ofstream dbg_file;

// ===========================================================================
// THE MACRO GAME
// ===========================================================================
// One mode. Two players, 30 days of 24 hours, and the side with more money on
// the last night wins.
//
// The network decides ONCE PER DAY. At hour 0 it emits an objective grid plus
// its global choices; the filter in policy.cpp turns that into a legal
// cell_goal grid and a market order list, and MacroPlanner routes the day's
// hands through it. Hours 1..23 involve no network at all.
//
// ---------------------------------------------------------------------------
// WATER IS NOT A DECISION. FERTILIZE IS, AND IT HAS ITS OWN CHANNEL.
// ---------------------------------------------------------------------------
// There is no water head. Enumerate every case where a water is worth an hour:
//
//   1. life support        must_water        -- a second dry day is weeds,
//                                              AND the plant can still gain
//                                              something after tonight (see
//                                              plant_has_future)
//   2. yield, non-ongoing  must_yield_water  -- A_WATER *is* the unit
//   3. banking a fertilize must_bank_water   -- both payout sites read the
//                                              fertilizer only on a WATERED
//                                              day, so the coins are wasted
//                                              without it. The fertilize pass
//                                              drags the same water in on the
//                                              day it spends the fertilizer.
//   4. the planting day    the primary pass  -- consecutive_unwatered == 1 on
//                                              a fresh plant, so an unwatered
//                                              first night is weeds
//
// Every one of those is decided by arithmetic the environment already
// performs, so no head chooses a water.
//
// THE FERTILIZE HEAD. The primary slot names a crop and nothing else -- the
// old "fertilizer path" slots (WHEAT_0 / WHEAT_1 / ... / STRAW_2) and the
// tile's fert_plan counter are GONE. In their place is one [No_Fert, Fert]
// pair per cell, drawn every day by its own pass (H_FERTILIZE), which runs
// AFTER the primary pass so a cell planted this morning can be fertilized on
// the same visit.
//
// The pair is only ever drawn where a fertilizer PAYS (fertilize_pays): the
// cell is a plant, not a melon, and the days the new cover adds contain a
// payout night with headroom for the extra unit. Everywhere else the cell is
// never offered, so the head gets no log-prob and no entropy from a choice
// that had one right answer.
//
// ---------------------------------------------------------------------------
// PLANTING IS FORCED ONCE DRAWN
// ---------------------------------------------------------------------------
// Every unit the count heads ask for is carried out if it can be: if the
// routes cannot absorb it, the filter HIRES hands mid-plan until they can (up
// to the hard crew cap and whatever the day can pay for). A unit no
// affordable hire makes fit is skipped, and the next cell in the type's
// distance order is tried.
//
// ---------------------------------------------------------------------------
// THE DEAD-PLANT REWORK  --  a cell that is a weed by morning
// ---------------------------------------------------------------------------
// A plant with nothing left to gain is NOT life support. The ONGOING crops are
// where this bit: max_lifespan_step is written on the last production night,
// so from the next morning a tomato or a strawberry is decaying scenery that
// still reads T_PLANT. The old must_water fired on it anyway, so a hand walked
// out to water a crop decay_plants turned into T_WEED the same day, and the
// cell then sat unplantable until someone dug the weed. Two predicates fix it:
//
//   plant_has_future  can a LATER day still add a unit to this plant? It is
//                     the guard on must_water, and nothing else changed about
//                     the other three reasons a water is worth an hour.
//   plant_is_doomed   the tile is a weed by morning whatever the plan does.
//                     cell_musts spends nothing on such a cell beyond the
//                     harvest that banks what is standing, and the filter
//                     treats it as CLEARABLE GROUND, so the primary pass may
//                     dig it and plant on the same day rather than losing one.
//
// HARVEST, DIG, PLANT -- ONE VISIT. The morning after an ongoing crop's last
// production night it is doomed (it starts to rot tomorrow) AND it still holds
// that night's units, so must_harvest fires on it. The harvest used to lock
// the cell for the day. Now the primary pass may replant it: the goal carries
// want_pre_harvest, and build_task emits A_HARVEST, A_DIG, A_PLANT in that
// order, so the units are banked before the dig clears the tile. The same
// applies to a doomed ongoing crop whose harvest was drawn rather than forced.
//
// Neither predicate changes a channel index or an output width: they are read
// through must_water / cell_musts / tile_pending_ops, which the observation
// already routes through channels 37, 42 and 44.
//
// ---------------------------------------------------------------------------
// THE MUST TIER
// ---------------------------------------------------------------------------
// Nothing in this tier is a decision. Getting any of it wrong costs a whole
// cell's investment, or throws away something already paid for, and the sign
// is never ambiguous, so the filter FORCES it -- no candidate set, no
// log-prob, no entropy -- and masks the surviving heads off exactly those
// cells. See CellMust / cell_musts below.
//
//   harvest    ripe and gone by morning
//   water      life support on a plant with a future, or the yield water, or
//              banking a fertilizer
//   feed       the animal escapes tonight, or a care bonus would be wiped
//   care       rides on every feed, unconditionally
//   collect    any animal with fertilizer waiting: one hour for ~100 coins
//   prewater   a wiping harvest is watered first when the window is open
//
// Fertilize is NOT in this tier any more: it is the fertilize head's call.
//
// ---------------------------------------------------------------------------
// BREAKING CHANGES IN THIS REVISION  --  no hire head, a richer sell-time head
// ---------------------------------------------------------------------------
//   * THE HIRE HEAD IS GONE. Only hand 0 survives the night, so every day
//     starts with one hand and EVERY other hand is hired on demand by the
//     pass that needs it, in priority order: the must tier, harvest, plant,
//     fertilize, feed, then the mid-day drops of the Sell harvests. Each pass
//     hires only when the routes cannot absorb what it drew, up to the crew
//     cap (MACRO_HIRE_HARD_MAX, 16 hands in all) and what the day can pay
//     for. The hire floor is gone with it (the forced pass hires for itself).
//     glob_hire and its 8 outputs are deleted; H_HIRE stays in HeadId as a
//     dead head, like H_WATER.
//   * THE SELL-TIME HEAD HAS 14 BINS: the 12 hours, then ST_NO_HARVEST and
//     ST_KEEP. See the sell-time banner above MACRO_SELL_TIME_BINS for when
//     each is offered.
//   * `spatial` is 40 -> 42 channels and MACRO_DIM 4167 -> 4359. Every global
//     head moved (the hire bins went, the sell-time block grew).
//     load_compatible remaps the old `spatial` channels into the new layout,
//     so only the two new channels start fresh; glob_hire is ignored.
//   * THE FORECAST BLOCK GREW BY 12 (FCAST_DIM 111 -> 123): the land that
//     comes free today or tomorrow, by distance band (forecast.hpp). That
//     pushes the scalar features to 503, past five planes, so there are SIX
//     scalar planes now (NUM_INPUT_CH 93 -> 94). The new features are
//     appended at the END of the forecast block, so load_compatible copies
//     the old columns of in_fc / value_in_fc and zeroes the new ones.
//
// ---------------------------------------------------------------------------
// EARLIER  --  mid-day selling
// ---------------------------------------------------------------------------
//   * THE HARVEST HEAD HAS THREE BINS: [None, Keep, Sell] per type (the
//     triples are still H_HARVEST). Sell drops the units at the shed mid-day
//     and sells them at an hour drawn by the new per-cell SELL-TIME head
//     (H_SELL_TIME, bins = hours 5 9 13 17 21 23). A forced harvest draws
//     [Keep, Sell] -- or is forced to Sell when the night shed is already
//     over capacity. `spatial` is 20 -> 34 channels and MACRO_DIM
//     2167 -> 4167; load_compatible re-initialises `spatial` only.
//   * THE FOUR TYPE SLOTS ARE FOUR HEADS (H_PLANT_TYPE_0..3), each with its
//     own entropy target (plant_type_1..4 in entropy_file; a bare plant_type
//     still sets all four). Which model a head reads is head_is_global().
//   * MacroPlanner inserts the mid-day drops itself (drop-all when the hand
//     holds only the batch, drop-item otherwise), searches harder before
//     refusing (ejection, re-optimisation), and the filter hires up to the
//     crew cap for any harvest bin that does not fit.
//   * GB_FERTILIZED is satisfied at fertilized >= day+2, not >= day. A
//     re-application fertilize_pays offered used to be BOUGHT and then never
//     emitted -- the head was credited for a no-op and the unused fertilizer
//     overflowed the night ledger.
//
// ---------------------------------------------------------------------------
// EARLIER BREAKING CHANGES
// ---------------------------------------------------------------------------
//   * THE FERTILIZE HEAD IS BACK and the path slots are gone. The primary head
//     is 15 -> 9 channels (NONE, one slot per crop, one per animal) and a new
//     [No_Fert, Fert] pair sits between primary and feed, so `spatial` is
//     33 -> 29 channels and MACRO_DIM 3442 -> 3042. Every global head moved
//     400 floats down. load_compatible re-initialises `spatial` and keeps
//     everything else.
//   * H_FERTILIZE draws again. H_WATER stays dead.
//   * `tile` lost fert_plan, so write_player_block is back to 15 fields per
//     tile. THE VIEWER'S PARSER MOVES BACK.
//   * A_PLANT ignores `count` again; act_plant takes the crop only.
//   * Observation channels keep their indices but 56 and 58 change meaning:
//     56 is FERTILIZE PAYS (the fertilize head's mask on the hour-0 board) and
//     58 is a live fertilizer READ tonight. Channel 21 still reports the
//     canonical schedule's remaining applications.
//   * All hires go out as ONE A_HIRE order carrying the count, so a hire made
//     mid-plan for a forced plant never moves the order hours.
//   * The critic projection no longer infers the opponent's path: both sides
//     run the same canonical fertilize schedule (fert_day_for + fertilize_pays).
// ===========================================================================

// ---------------------------------------------------------------------------
// Items
// ---------------------------------------------------------------------------
// Products 0..8 are tradable. Livestock 9..11 are carryable and shed-storable
// but not tradable as products.
enum Item {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON,
    EGG, MILK, WOOL, FERTILIZER,
    GOOSE, COW, SHEEP,
    NUM_ITEMS
};

constexpr int NUM_PRODUCTS = 9;   // WHEAT..FERTILIZER, the market-traded set
constexpr int NUM_CROPS    = 5;   // WHEAT..MELON
constexpr int NUM_ANIMALS  = 3;   // GOOSE..SHEEP

inline int animal_item(int a)    { return GOOSE + a; }
inline int animal_product(int a) { return EGG + a; }

// ---------------------------------------------------------------------------
// Board / episode geometry
// ---------------------------------------------------------------------------
constexpr int BOARD_SIZE    = 10;
constexpr int TURNS_PER_DAY = 24;
constexpr int NUM_DAYS      = 30;
constexpr int MAX_TURNS     = NUM_DAYS * TURNS_PER_DAY;   // 720
constexpr int NUM_PLAYERS   = 2;
constexpr int CELLS         = BOARD_SIZE * BOARD_SIZE;    // 100
// ---------------------------------------------------------------------------
// THE LAST DAY IS NOT A DECISION
// ---------------------------------------------------------------------------
// check_winner reads `money` and nothing else, and no plant, animal, seed or
// hand outlives the night of day NUM_DAYS-1. So on that day there is exactly
// one right answer -- bank every unit that exists and sell it -- and nothing
// to trade off, nothing to explore, nothing to learn. It is run by
// macro_liquidate_day, which draws nothing and reads no logits.
//
// The episode is therefore NUM_DAYS-1 decisions long. The terminal +-1 is
// credited to the LAST DECISION (day NUM_DAYS-2), not to the liquidation, or
// the critic would be regressing a state whose value is already determined by
// arithmetic the policy had no say in.
constexpr int MACRO_DECISION_DAYS = NUM_DAYS - 1;         // 29
inline bool macro_is_last_day(int day) { return day >= NUM_DAYS - 1; }

// The last day a crop may still be planted on and reach its FIRST yield, as
// `day + first_yield_day <= PLANT_HORIZON_DAY - 1`. A curriculum knob, not a
// derived quantity: shortening it is how you stop the planner sinking seeds
// into melons on day 26.
//
// NUM_DAYS, not NUM_DAYS-1. The gate asks whether the first yield ARRIVES in
// time, and the liquidation harvests on day NUM_DAYS-1, so a first yield on
// day 29 is bankable. At 29 this was one day tight and masked off wheat that
// would have sold -- the last day used to be a normal planning day that might
// or might not have got round to it, and now it is a routine that always does.
constexpr int PLANT_HORIZON_DAY = NUM_DAYS;

constexpr double STARTING_MONEY     = 3000.0;
constexpr int    SHED_CAPACITY      = 100;
constexpr double WEED_SPAWN_CHANCE  = 0.005;
constexpr int    MAX_MARKET_ORDERS_PER_TURN = 10;
constexpr int    FARM_HAND_COST_MULT = 1;
constexpr int    MAX_UNITS = 16;

// Quadrants. NW is owned from turn 0; the rest unlock in LAND_ORDER.
enum Quadrant { Q_NW = 0, Q_NE, Q_SW, Q_SE, NUM_QUADRANTS };
constexpr int LAND_ORDER[3]  = { Q_NE, Q_SW, Q_SE };
constexpr int LAND_PRICES[3] = { 1000, 2000, 4000 };

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------
enum TileType { T_EMPTY = 0, T_WEED, T_PLANT, T_COOP, T_PASTURE };

struct tile {
    int type   = T_EMPTY;
    int bought = 0;          // 0 == LOCKED. Units may stand on it; tile ops no-op.
    int has_farmer = 0;      // recomputed each turn, for the encoder

    // T_PLANT
    int crop        = -1;
    int phase       = 0;     // planted_day / placed_day
    int watered_today = 0;
    int consecutive_unwatered = 0;
    int fertilized  = -1;    // fertilized_until_day, -1 == never
    int max_lifespan_step = -1;

    // T_COOP / T_PASTURE
    int animal      = -1;    // -1 == empty structure
    int fed_today   = 0;
    int cared_today = 0;
    int consecutive_unfed = 0;
    int fertilizer_available = 0;
    int pending_care_bonus = 0;

    // both
    int yield_units = 0;
};

// ---------------------------------------------------------------------------
// A unit's carried inventory
// ---------------------------------------------------------------------------
// `keys` mirrors Python dict insertion order, which is what decides who wins
// the last of the shed capacity in the two overflow-sensitive paths (DROP-all
// and the end-of-day sweep).
struct unit_inv {
    std::array<int, NUM_ITEMS> n{};
    std::vector<int> keys;

    int operator[](int item) const { return n[item]; }

    void add(int item, int k = 1) {
        if (item < 0 || item >= NUM_ITEMS || k <= 0) return;
        if (n[item] == 0) keys.push_back(item);
        n[item] += k;
    }
    bool empty() const { return keys.empty(); }
    // All-or-nothing, and the key vanishes at zero.
    bool take(int item, int k = 1) {
        if (item < 0 || item >= NUM_ITEMS || k <= 0) return false;
        if (n[item] < k) return false;
        n[item] -= k;
        if (n[item] == 0) drop_key(item);
        return true;
    }
    void sub_to(int item, int k) {
        if (item < 0 || item >= NUM_ITEMS || k <= 0) return;
        n[item] -= k;
        if (n[item] <= 0) { n[item] = 0; drop_key(item); }
    }
    void drop_key(int item) {
        keys.erase(std::remove(keys.begin(), keys.end(), item), keys.end());
    }
    void clear() { n.fill(0); keys.clear(); }
    int  total() const { return std::accumulate(n.begin(), n.end(), 0); }
};

// ---------------------------------------------------------------------------
// Static game data
// ---------------------------------------------------------------------------
struct CropDef {
    int  seed_cost;
    int  first_yield_day;
    int  max_yield_day;
    int  interval;          // ongoing crops only
    int  max_yield;
    bool ongoing;
};

struct AnimalDef {
    int cost;
    int structure;          // T_COOP / T_PASTURE
    int first_yield_day;
    int interval;
    int max_held;
    int product;            // Item
};

extern const CropDef   CROPS[NUM_CROPS];
extern const AnimalDef ANIMALS[NUM_ANIMALS];

// ---------------------------------------------------------------------------
// THE ANIMAL HORIZON  --  the mirror of the crop one, and it was missing
// ---------------------------------------------------------------------------
// A_PLACE writes phase = day. daily_refresh_animals produces when
// `next - phase - first_yield_day` hits 0 with next == day+1, so the first
// unit lands at the end of day phase + first_yield_day - 1 and is harvestable
// from day phase + first_yield_day. The liquidation harvests on day
// NUM_DAYS-1, so anything placed later than that produces NOTHING a hand can
// ever pick up:
//
//   goose  first_yield 4, 300 coins -> placeable through day 25
//   sheep  first_yield 6, 500 coins -> through day 23
//   cow    first_yield 8, 400 coins -> through day 21
//
// Without this the count head could spend 400 coins on a cow on day 25 and
// then pay wheat into forced feeds for the rest of the episode to keep an
// animal that cannot produce. Crops were gated by PLANT_HORIZON_DAY from the
// start; animals simply were not.
inline int animal_place_horizon_day(int a) {
    return (NUM_DAYS - 1) - ANIMALS[a].first_yield_day;
}
inline bool animal_in_horizon(int a, int day) {
    return a >= 0 && a < NUM_ANIMALS && day <= animal_place_horizon_day(a);
}

// Market pricing:  price(inv) = base +- amp * f(|inv - I0|), floored at 1.
enum Shape { SH_LINEAR = 0, SH_SQ, SH_SQRT, SH_LOG, SH_LOG10, SH_HINGE };

struct MarketDef {
    double base;
    double I0;
    double T;
    Shape  below;  double below_target;
    Shape  above;  double above_target;
};

extern const MarketDef MARKET[NUM_PRODUCTS];

constexpr double HINGE_GAIN  = 8.0;
constexpr int    PRICE_FLOOR = 1;
constexpr double MARKET_I0   = 10000.0;

// Town
enum Shop {
    SHOP_BAKERY = 0, SHOP_BRUNCH_SPOT, SHOP_FARMERS_MARKET, SHOP_ICE_CREAM_SHOP,
    SHOP_PET_CAFE, SHOP_PIZZA_SHOP, SHOP_SMOOTHIE_SHOP, SHOP_YARN_STORE,
    NUM_SHOPS
};
extern const std::vector<int> SHOP_PRODUCTS[NUM_SHOPS];

constexpr int MAX_SHOP_INSTANCES        = 8;
constexpr int TOWN_SHOP_SELL_INTERVAL   = 4;
constexpr int TOWN_CENTER_SELL_INTERVAL = 24;
constexpr int TOWN_SHOP_UNLOCK_INTERVAL = 3;

// ===========================================================================
// THE OBJECTIVE GRID
// ===========================================================================
// One goal per cell, saying what the cell should look like by the end of the
// day. In the macro rework this is an ACTION, not an environment sample: the
// network emits it, the filter makes it legal, and MacroPlanner executes it.
//
// There is no `then_plant` field. `target_type` alone carries the post-harvest
// state -- a cell holding ripe wheat with target TOMATO means harvest the
// wheat and plant a tomato -- and the PATH is inferred by build_task rather
// than stored, so a goal and a tile can never disagree about what was asked.
//
//   harvest vs dig: if harvesting would clear the tile (a non-ongoing crop --
//   A_HARVEST does t = tile{T_EMPTY,1}) and the target differs, the path is a
//   harvest. Otherwise (ongoing crop, weed, empty structure) it is a dig. Note
//   A_DIG does not touch a live animal, which is why such cells are frozen.
//
// `want_harvest` is only ever APPLICABLE for ongoing crops (tomato /
// strawberry) and for animals -- the two cases where harvesting does not
// change the type, so the type field cannot carry the information. For a
// non-ongoing crop the transition already implies it, and the tile is wiped on
// harvest anyway, so no per-cell "was harvested" flag could survive.
//
// APPLICABILITY IS RE-DERIVED from target_type and never trusted from the
// struct, so a filter bug cannot leave a requirement in the plan that nothing
// will ever act on.
//
// GB_FERTILIZED is set by the fertilize pass, one [No_Fert, Fert] draw per
// cell where a fertilizer pays. GB_CARED has no head: the filter sets it
// whenever it sets GB_FED, and build_task emits A_FEED before A_CARE.
//
// want_pre_water is the one BOOLEAN that is NOT a day-end predicate, which is
// why it is not a GoalBool. Every GoalBool describes what the cell should LOOK
// LIKE tonight, and a water spent on a plant that is harvested away the same
// day leaves nothing on the tile to read -- the same reason there is no "was
// harvested" flag. It means "water what is STANDING here, before the type fix
// clears it", build_task emits it ahead of the clearing harvest, and it exists
// because A_WATER inside the non-ongoing yield window is what adds the unit
// that harvest then banks.
//
// want_pre_harvest is the same kind of boolean for the ONGOING crops: "bank
// what is STANDING here, before the type fix digs it". It is only set when
// the primary pass replants a doomed ongoing crop that still holds units --
// the morning after its last production night -- and build_task emits it
// after any pre-water and before the A_DIG. Without it the harvest would be
// ordered after the A_PLANT and would land on the new, empty plant.
// ===========================================================================
enum GoalBool {
    GB_WATERED = 0,      // target is a plant
    GB_FERTILIZED,       // target is a plant
    GB_FED,              // target is an animal-holding structure
    GB_CARED,            // target is an animal-holding structure
    GB_COLLECT,          // target is an animal-holding structure
    GB_HARVEST,          // ongoing crop / animal only  (see above)
    NUM_GOAL_BOOLS
};

struct cell_goal {
    int target_type   = T_EMPTY;
    int target_crop   = -1;
    int target_animal = -1;

    int want_watered  = 0;
    int want_fertilized = 0;
    int want_fed = 0;
    int want_cared = 0;
    int want_collect_fertilizer = 0;
    int want_harvest = 0;

    // NOT a GoalBool: it describes an op on the cell's OLD contents, which do
    // not survive the day. See the note above.
    int want_pre_water = 0;

    // NOT a GoalBool either: an in-place (ongoing) harvest of the OLD plant,
    // ahead of the dig that clears it for a replant. See the note above.
    int want_pre_harvest = 0;

    // NOT a predicate: set only by the primary pass. A plant goal with this
    // set is a NEW planting, so whatever stands on the cell at hour 0 --
    // even the same crop -- is harvested (if ripe) or dug, then replanted.
    // Without it a plant goal means "keep the plant that is there", which is
    // what every must-tier goal on a standing plant is.
    int replant = 0;

    // NOT a GoalBool: HOW this cell's harvest leaves the hand, not what the
    // cell looks like tonight. -1 (the default) is the old behaviour -- the
    // units ride in hand to the nightly sweep. An hour >= 0 is the SELL bin
    // of the harvest head: the hand DROPS the units at the shed no later than
    // this hour (a deadline -- earlier is fine) and a market sell of exactly
    // those units goes out AT this hour (not before, not after). Only read
    // when the task actually contains an A_HARVEST; see MacroPlanner.
    int sell_hour = -1;

    // Set the moment the filter accepts anything for this cell. A cell with
    // this clear was never asked for anything and build_task refuses it, which
    // is what keeps "leave it exactly as it is" free.
    int in_denominator = 0;
};

inline bool goal_is_plant(const cell_goal& g) {
    return g.target_type == T_PLANT && g.target_crop >= 0 && g.target_crop < NUM_CROPS;
}
inline bool goal_is_structure(const cell_goal& g) {
    return g.target_type == T_COOP || g.target_type == T_PASTURE;
}
// "animal-holding structure" -- the only case the three animal-care booleans
// and want_harvest can apply to.
inline bool goal_is_animal(const cell_goal& g) {
    return goal_is_structure(g) &&
           g.target_animal >= 0 && g.target_animal < NUM_ANIMALS;
}

inline int goal_wants(const cell_goal& g, int b) {
    switch (b) {
        case GB_WATERED:    return g.want_watered;
        case GB_FERTILIZED: return g.want_fertilized;
        case GB_FED:        return g.want_fed;
        case GB_CARED:      return g.want_cared;
        case GB_COLLECT:    return g.want_collect_fertilizer;
        case GB_HARVEST:    return g.want_harvest;
    }
    return 0;
}

inline bool goal_bool_applicable(const cell_goal& g, int b) {
    if (goal_is_plant(g)) {
        if (b == GB_WATERED || b == GB_FERTILIZED) return true;
        if (b == GB_HARVEST) return CROPS[g.target_crop].ongoing;
        return false;
    }
    if (goal_is_animal(g))
        return b == GB_FED || b == GB_CARED || b == GB_COLLECT || b == GB_HARVEST;
    return false;   // empty target, or a bare structure: nothing applies
}

inline int goal_asked_count(const cell_goal& g) {
    int n = 0;
    for (int b = 0; b < NUM_GOAL_BOOLS; ++b)
        if (goal_bool_applicable(g, b) && goal_wants(g, b)) ++n;
    return n;
}
// Would A_HARVEST do anything on this tile, this turn? The harvest head's mask
// and the observation channel share it.
inline bool harvest_ready(const tile& t, int day) {
    if (!t.bought || t.yield_units <= 0) return false;
    if (t.type == T_PLANT)
        return t.crop >= 0 && day - t.phase >= CROPS[t.crop].first_yield_day;
    return t.animal >= 0;
}

// Does the cell hold the right THING? Crop death and animal escape need no
// special case: a dead crop is T_WEED and a fled animal leaves an empty
// structure, so both simply stop matching their target.
inline bool goal_gate_passed(const cell_goal& g, const tile& t, int day) {
    if (goal_is_plant(g)) {
        // A new planting never already stands at hour 0: clear and replant,
        // even over the same crop (ripe -> HARVEST, otherwise DIG).
        if (g.replant) return false;
        return t.type == T_PLANT && t.crop == g.target_crop;
    }
    if (goal_is_structure(g)) {
        if (t.type != g.target_type) return false;
        if (g.target_animal >= 0) return t.animal == g.target_animal;
        return true;                       // bare structure: type is enough
    }
    return t.type == T_EMPTY;
}

// How each boolean is READ off a tile.
//   GB_COLLECT  A_COLLECT_FERTILIZER clears fertilizer_available, so the
//               cleared flag IS the record of the collection.
//   GB_HARVEST  A_HARVEST zeroes yield_units, and for the ongoing crops and
//               animals this applies to, both refill paths live in end_of_day,
//               so within a day yield_units == 0 is the record.
inline bool goal_bool_satisfied(int b, const tile& t, int day) {
    switch (b) {
        case GB_WATERED:    return t.watered_today != 0;
        // A_FERTILIZE sets max(fertilized, day+2), and fertilize_pays offers a
        // re-application while the live cover ends before day+2 -- so the goal
        // is only met once the cover reaches day+2. (It used to read
        // `>= day`: the re-application was paid for, then never emitted.)
        case GB_FERTILIZED: return t.fertilized >= day + 2;
        case GB_FED:        return t.fed_today != 0;
        case GB_CARED:      return t.cared_today != 0;
        case GB_COLLECT:    return t.fertilizer_available == 0;
        case GB_HARVEST:    return t.yield_units == 0;
    }
    return false;
}

// ===========================================================================
// DOES A FERTILIZER PAY?  --  the fertilize head's mask
// ===========================================================================
// THE ONE PLACE THE TWO PAYOUT SITES ARE WRITTEN DOWN, by the calendar alone.
//
// A_FERTILIZE sets the deadline to max(fertilized, day+2), so an application
// today covers [day, day+2]. The bonus is only read on a covered day that ALSO
// produces:
//   non-ongoing: read in A_WATER, ages [(max_yield_day+1)/2 .. max_yield_day]
//   ongoing:     read in daily_refresh_plants, on the derived production nights
// Both read it only on a day the plant is WATERED, which is why every
// fertilize drags a water with it on a payout day (the fertilize pass) and on
// every later covered payout day (must_bank_water).
//
// The last day is the same statement about the clock. A unit produced on an
// ongoing crop's night d is harvested on day d+1 at the earliest, so the
// night must be <= NUM_DAYS-2. A non-ongoing unit is harvested the day the
// water adds it, so any day up to NUM_DAYS-1 banks. And the application itself
// is never made after MACRO_FERT_LAST_DAY: the coins are the one number the
// game scores.
constexpr int MACRO_FERT_LAST_DAY = NUM_DAYS - 2;      // 28

inline bool fert_payout_day(const tile& t, int d) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    const CropDef& cd = CROPS[t.crop];
    if (!cd.ongoing) {
        if (d > NUM_DAYS - 1) return false;
        const int age = d - t.phase;
        return age >= (cd.max_yield_day + 1) / 2 && age <= cd.max_yield_day;
    }
    if (d > NUM_DAYS - 2 || cd.interval <= 0) return false;
    const int since = d + 1 - t.phase - cd.first_yield_day;
    if (since < 0 || since % cd.interval != 0) return false;
    return since / cd.interval + 1 <= cd.max_yield;
}

// Is spending ONE FERTILIZER on this tile today going to add a unit?
//
//   * never on a melon (the referee reads it, but the head was told never to
//     spend on one), never after MACRO_FERT_LAST_DAY;
//   * only the days the NEW cover adds count -- a cell still covered through
//     tomorrow only gains day+2 -- so a second application on top of a live
//     one is offered only when that extra day is a payout day;
//   * the extra unit needs headroom: both payout sites clamp with
//     min(max_yield, y + bonus), so the bonus is worth nothing at
//     yield_units + 1 >= max_yield. The standing yield is walked forward
//     across the window (no harvest assumed in between, which is the
//     conservative direction), with any EXISTING cover adding its 2.
//
// A pure function of public tile state, so it is safe over the opponent's
// board as well (tile_pending_ops).
inline bool fertilize_pays(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    if (t.crop == MELON) return false;
    if (day > MACRO_FERT_LAST_DAY) return false;
    const CropDef& cd = CROPS[t.crop];
    const int lo = std::max(day, t.fertilized + 1);   // first day the new cover adds
    const int hi = day + 2;
    if (lo > hi) return false;
    int y = t.yield_units;
    for (int d = day; d <= hi; ++d) {
        if (!fert_payout_day(t, d)) continue;
        if (d >= lo && y + 1 < cd.max_yield) return true;
        y = std::min(cd.max_yield, y + ((t.fertilized >= d) ? 2 : 1));
    }
    return false;
}

// ===========================================================================
// DERIVED TILE PREDICATES  --  the observation's engineered channels
// ===========================================================================
// Every one of these is a fact the environment already computes somewhere in
// game.cpp and that the network would otherwise have to rediscover from an age
// channel, a crop one-hot and a modulus it cannot represent. They live here,
// next to fertilize_pays and harvest_ready, for the same reason those do:
// ONE definition, read by the encoder and available to the filter, so the
// observation and the referee can never drift apart.
//
// They are all pure functions of (tile, day) or of a static table, so none of
// them leaks anything the player does not already see.
// ===========================================================================

// What this cell would put in a hand if harvested, or -1 for nothing.
inline int tile_product(const tile& t) {
    if (t.type == T_PLANT && t.crop >= 0) return t.crop;   // crop id == product id
    if (t.animal >= 0) return ANIMALS[t.animal].product;
    return -1;
}
// WHICH HARVEST CHANNEL PAIR THIS CELL DRAWS FROM, in the same 0..7 type space
// the count ceilings use: 0..NUM_CROPS-1 is a crop, NUM_CROPS.. an animal.
// -1 means the cell has nothing that could ever be harvested off it, which
// harvest_ready already excludes -- the guard is kept so a caller that has not
// checked cannot index a channel that does not exist.
inline int tile_harvest_type(const tile& t) {
    if (t.type == T_PLANT && t.crop >= 0 && t.crop < NUM_CROPS) return t.crop;
    if (t.animal >= 0 && t.animal < NUM_ANIMALS) return NUM_CROPS + t.animal;
    return -1;
}
inline int tile_max_yield(const tile& t) {
    if (t.type == T_PLANT && t.crop >= 0) return CROPS[t.crop].max_yield;
    if (t.animal >= 0) return ANIMALS[t.animal].max_held;
    return 0;
}
inline int tile_first_yield_day(const tile& t) {
    if (t.type == T_PLANT && t.crop >= 0) return CROPS[t.crop].first_yield_day;
    if (t.animal >= 0) return ANIMALS[t.animal].first_yield_day;
    return -1;
}
inline int tile_interval(const tile& t) {
    if (t.type == T_PLANT && t.crop >= 0) return CROPS[t.crop].interval;
    if (t.animal >= 0) return ANIMALS[t.animal].interval;
    return 0;
}

// daily_refresh_plants turns a plant that ends a SECOND dry day into weeds,
// and a freshly planted one already carries consecutive_unwatered == 1. This
// is the single most expensive thing the planner can get wrong, and it is one
// AND away from being invisible in the raw channels.
inline bool plant_dies_tonight(const tile& t, int day) {
    (void)day;
    return t.type == T_PLANT && !t.watered_today && t.consecutive_unwatered >= 1;
}
// The animal mirror: daily_refresh_animals evicts on the second unfed night.
inline bool animal_escapes_tonight(const tile& t, int day) {
    (void)day;
    return t.animal >= 0 && !t.fed_today && t.consecutive_unfed >= 1;
}

// ---------------------------------------------------------------------------
// CAN THIS PLANT STILL GAIN A UNIT AFTER TONIGHT?
// ---------------------------------------------------------------------------
// THE GUARD ON LIFE SUPPORT. Keeping a plant alive is only worth an hour if a
// later day can still add something to it:
//
//   ongoing      max_lifespan_step is written on the LAST production night
//                (daily_refresh_plants, production_count == max_yield), so the
//                moment it is set the crop will never produce again. From the
//                next morning it is decaying scenery that still reads T_PLANT,
//                and every water spent on it is an hour thrown away.
//   non-ongoing  A_WATER inside [(max_yield_day+1)/2 .. max_yield_day] is the
//                ONLY way a unit ever arrives, so once day reaches
//                phase + max_yield_day no later day can add one. Today's
//                water may still pay -- that is must_yield_water, which is a
//                separate must and is NOT gated on this.
//
// The last day is the same statement about the clock: nothing happens after the
// night of day NUM_DAYS-1, so surviving it buys nothing.
inline bool plant_has_future(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    if (day + 1 > NUM_DAYS - 1) return false;        // no tomorrow to gain in
    const CropDef& cd = CROPS[t.crop];
    if (cd.ongoing) return t.max_lifespan_step < 0;
    return day < t.phase + cd.max_yield_day;
}

// GONE BY MORNING, whatever the plan does today. Two ways in:
//
//   * decay_plants is already eating it, or starts tonight. THIS IS THE
//     ONGOING CASE, and it is the one that used to cost hours AND cells: the
//     day after a tomato's or a strawberry's last production night the tile
//     still reads T_PLANT, so must_water fired, a hand walked out to water a
//     crop that turned into T_WEED the same day, and the cell then sat
//     unplantable until someone dug the weed.
//   * it ends a second dry day and life support was not worth the hour
//     (plant_has_future said no), so nothing is going to water it.
//
// Read in two places: cell_musts spends nothing on such a cell beyond the
// harvest that banks what is standing, and the filter's `clearable` treats it
// as ground, so the primary pass may dig it and plant on the same day.
inline bool plant_is_doomed(const tile& t, int day, int turn) {
    if (t.type != T_PLANT || t.crop < 0) return false;
    if (t.max_lifespan_step >= 0 && turn + TURNS_PER_DAY >= t.max_lifespan_step)
        return true;
    return plant_dies_tonight(t, day) && !plant_has_future(t, day);
}

// Will end_of_day add a unit to this ongoing crop tonight? Same arithmetic as
// daily_refresh_plants, including the production_count cap and the fact that
// the death check runs first and `continue`s past production.
inline bool plant_produces_tonight(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    const CropDef& cd = CROPS[t.crop];
    if (!cd.ongoing || cd.interval <= 0) return false;
    if (plant_dies_tonight(t, day)) return false;
    if (t.yield_units >= cd.max_yield) return false;
    const int since = (day + 1) - t.phase - cd.first_yield_day;
    if (since < 0 || since % cd.interval != 0) return false;
    return since / cd.interval + 1 <= cd.max_yield;
}
// The animal version, out of daily_refresh_animals. NOTE it does not read
// fed_today: an animal produces whether or not it ate. Feeding buys two
// things only -- not losing the animal (a must) and a fed night for the care
// bonus to bank on (which is why every feed drags a care with it).
inline bool animal_produces_tonight(const tile& t, int day) {
    if (t.animal < 0) return false;
    const AnimalDef& a = ANIMALS[t.animal];
    if (a.interval <= 0) return false;
    if (animal_escapes_tonight(t, day)) return false;
    if (t.yield_units >= a.max_held) return false;
    const int since = (day + 1) - t.phase - a.first_yield_day;
    return since >= 0 && since % a.interval == 0;
}

// Is a care bonus BANKED tonight?
//
// Read daily_refresh_animals in order: tonight's production is resolved and
// pending_care_bonus consumed FIRST, and only then does `cared_today &&
// fed_today` add one to the bank. So a care tonight can never pay tonight --
// it pays on a LATER production night that also fed.
inline bool care_banks_tonight(const tile& t, int day) {
    (void)day;
    return t.animal >= 0 && t.fed_today && t.cared_today;
}

// Given that this animal IS being fed today, the care is forced as well.
// Unconditional: the hand is already on the cell and the feed supplies the
// fed night the bonus banks on. No head, no draw.
inline bool must_care_after_feed(const tile& t, int day) {
    (void)day;
    return t.animal >= 0 && !t.cared_today;
}

// NON-ongoing crops do not produce at night at all: A_WATER is what adds the
// unit, inside [(max_yield_day+1)/2 .. max_yield_day]. So for wheat, carrots
// and melons "watering pays right now" is a completely different predicate
// from "watering keeps it alive", and both are musts.
inline bool water_yields_now(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    const CropDef& cd = CROPS[t.crop];
    if (cd.ongoing || t.watered_today) return false;
    if (t.yield_units >= cd.max_yield) return false;
    const int age = day - t.phase;
    return age >= (cd.max_yield_day + 1) / 2 && age <= cd.max_yield_day;
}

// ===========================================================================
// THE CANONICAL FERTILIZER SCHEDULE  --  a reference, not a rule
// ===========================================================================
// Nothing in the filter reads this any more: when to fertilize is the
// fertilize head's call, gated only by fertilize_pays. The schedule survives
// as the CANONICAL POLICY the critic projection runs for BOTH players
// (projection.hpp) and as the reference behind observation channel 21. It
// names the days a sensible player would spend on:
//
//   wheat, carrot   one payout window, opening within two days of planting.
//   tomato      first_yield 8, interval 1, max_yield 4
//               nights at ages 7,8,9,10 and no more.
//               age 7  covers 7-9   -> three nights
//               age 10 covers 10-12 -> the last one
//   strawberry  first_yield 10, interval 2, max_yield 4
//               nights at ages 9,11,13,15. INTERVAL 2 INSIDE A 3-DAY WINDOW
//               MEANS AN ODD AGE CATCHES TWO:
//               age 9  covers 9-11  -> nights 9 and 11
//               age 13 covers 13-15 -> nights 13 and 15
//   melon       never.
//
// EVERY caller ANDs this with fertilize_pays, which is the authority on
// whether the unit has anywhere to land and whether the window really contains
// a payout. This function only says "the canonical schedule names today".
// ===========================================================================
inline bool fert_day_for(int crop, const tile& t, int day) {
    if (crop < 0 || crop >= NUM_CROPS) return false;
    const int age = day - t.phase;

    if (crop == MELON) return false;
    if (crop == WHEAT)  return age == 2;
    if (crop == CARROT) return age == 1;
    if (crop == TOMATO)     return age == 7 || age == 10;
    if (crop == STRAWBERRY) return age == 9 || age == 13;
    return false;
}

// Does a live fertilizer get READ tonight? The two payout sites, as a
// predicate: the ongoing crops read it in daily_refresh_plants on a production
// night, the non-ongoing ones read it in A_WATER inside the yield window.
inline bool fert_read_tonight(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return false;
    if (CROPS[t.crop].ongoing) return plant_produces_tonight(t, day);
    return water_yields_now(t, day);
}

// ===========================================================================
// THE MUST SET
// ===========================================================================
// Nothing here is a decision. Getting any of it wrong costs a whole cell's
// investment, or throws away something already paid for, and the sign is never
// ambiguous, so the filter FORCES it: no candidate set, no log-prob, no
// entropy. The surviving heads are then masked off exactly these cells, so
// every draw is a draw where both answers were defensible.
// ---------------------------------------------------------------------------

// The standing yield is gone by morning. For a PLANT that means decay, or the
// thirst must_water declines to pay for. For an animal it means escape, which
// is a real loss because must_feed can legitimately be dropped for want of
// wheat.
inline bool yield_lost_tonight(const tile& t, int day, int turn) {
    if (t.type == T_PLANT) {
        // NOT plant_dies_tonight on its own, though it was until the water
        // head was deleted. must_water covers the thirst case for any plant
        // that can still gain something -- and treating that as lost makes
        // must_harvest fire on a plant mid-yield-window, where `wiped`
        // suppresses the very water that would have saved it and A_HARVEST
        // clears the cell. Measured against the referee's own arithmetic, that
        // costs wheat 2 units of 4 (it banks at age 2 instead of running to
        // age 4) and carrot 1 of 3. Melon is unaffected: its window opens
        // before first_yield_day, so must_yield_water waters it daily and it
        // never goes dry.
        //
        // >= and not >: decay_plants starts eating at max_lifespan_step, so the
        // harvest has to be forced on the last day BEFORE that, or a hand
        // arriving mid-day finds half the yield already gone.
        if (t.max_lifespan_step >= 0 &&
            turn + TURNS_PER_DAY >= t.max_lifespan_step) return true;
        // The other half: a plant with NO future is not watered (see
        // plant_has_future), so it really does turn to weeds tonight and
        // whatever is standing on it has to be banked today.
        return plant_dies_tonight(t, day) && !plant_has_future(t, day);
    }
    if (t.animal >= 0) return animal_escapes_tonight(t, day);
    return false;
}

// ---------------------------------------------------------------------------
// MELONS ARE HARVESTED THE MOMENT THEY CAN BE
// ---------------------------------------------------------------------------
// Melon's yield window is ages 6..12 and first_yield_day is 10, so with the
// forced daily yield water it reaches max_yield (1 at planting + ages 6..10)
// on exactly the first day it is harvestable. Waiting past that buys nothing
// and risks the decay at age 13, so there is no decision to make: the harvest
// is FORCED (with its pre-water, via the `wiped` branch of cell_musts) and the
// melon harvest pair is never drawn. The planting mask in the filter
// (fcast::melon_profit) assumes exactly this timing.
inline bool melon_harvest_now(const tile& t, int day) {
    return t.type == T_PLANT && t.crop == MELON && harvest_ready(t, day);
}

// Ripe AND about to be lost: harvest it or throw it away. Or a ripe melon.
inline bool must_harvest(const tile& t, int day, int turn) {
    return t.bought && harvest_ready(t, day) &&
           (yield_lost_tonight(t, day, turn) || melon_harvest_now(t, day));
}
// LIFE SUPPORT, and only where the plant can still gain something after
// tonight. An ongoing crop past its last production night, or any plant past
// its yield window, is decaying scenery: watering it keeps a T_PLANT tile
// alive for nothing and blocks the cell from being replanted.
inline bool must_water(const tile& t, int day) {
    return t.bought && plant_dies_tonight(t, day) && plant_has_future(t, day);
}
inline bool must_feed(const tile& t, int day) {
    if (!t.bought) return false;
    if (animal_escapes_tonight(t, day)) return true;
    // Protect uncashed care bonuses from being wiped out on a production night
    if (t.pending_care_bonus > 0 && animal_produces_tonight(t, day)) return true;
    return false;
}

// Fertilizer is waiting on a stocked structure. Always worth the hour.
inline bool must_collect(const tile& t) {
    return t.bought && t.animal >= 0 && t.fertilizer_available != 0;
}

// On wheat, carrots and melons A_WATER inside the yield window IS the unit.
// One hand-hour for a unit is never a loss, so it is forced.
constexpr bool MACRO_FORCE_YIELD_WATER = true;
inline bool must_yield_water(const tile& t, int day) {
    return MACRO_FORCE_YIELD_WATER && t.bought && water_yields_now(t, day);
}

// A live fertilizer is READ tonight, and both payout sites read it only on a
// WATERED day. The coins are already spent, so the water is not a decision --
// this is the third of the four reasons a water is ever worth an hour. (The
// day the fertilizer is spent, the fertilize pass drags the same water in
// itself: at hour 0 the cover is not on the tile yet.)
inline bool must_bank_water(const tile& t, int day) {
    if (t.type != T_PLANT || t.crop < 0 || t.watered_today) return false;
    return (t.fertilized >= day) && fert_read_tonight(t, day);
}

// A forced harvest that WIPES the tile takes the standing yield with it -- and
// if the water would have landed inside the yield window, one more hour on a
// cell a hand is already visiting turns into one or two more units in the same
// harvest. The hand is routed there either way, so the marginal cost is the op
// and nothing else.
constexpr bool MACRO_FORCE_PREWATER_ON_MUST_HARVEST = true;

// PER-CELL PRECEDENCE. A must-harvest on a NON-ongoing crop wipes the tile
// (A_HARVEST does t = tile{T_EMPTY,1}), so watering it FOR LIFE SUPPORT is an
// hour spent on a plant that will not exist. An
// ongoing crop or an animal survives its harvest, so it keeps everything.
//
// A DOOMED PLANT IS THE SAME SITUATION ARRIVED AT FROM THE OTHER SIDE: the
// tile is a weed by morning whatever the plan does, so it gets the harvest (and
// the prewater, if the window is open) and nothing else. That is what stops a
// hand being sent out to water a tomato on the morning after its last
// production night.
//
// `prewater` is the other half of that: the wiped cell gets no life-support
// water, but it does get a YIELD water, ahead of the harvest, whenever
// water_yields_now says the window is open.
//
// `care` is only ever set together with `feed`. `collect` is independent.
// There is no `fertilize`: spending a fertilizer is the fertilize head's call.
struct CellMust {
    bool harvest = false, water = false, feed = false, prewater = false;
    bool care = false, collect = false;
    bool any() const {
        return harvest || water || feed || prewater || care || collect;
    }
    int  ops() const {
        return (int)harvest + (int)water + (int)feed + (int)prewater
             + (int)care + (int)collect;
    }
};

inline CellMust cell_musts(const tile& t, int day, int turn) {
    CellMust m;
    if (!t.bought) return m;
    m.harvest = must_harvest(t, day, turn);
    const bool wiped = m.harvest && t.type == T_PLANT && t.crop >= 0 &&
                       !CROPS[t.crop].ongoing;
    // A weed by morning either way: nothing but the harvest is worth an hour
    // here.
    const bool doomed = plant_is_doomed(t, day, turn);
    if (!wiped && !doomed) {
        // The three reasons a standing cell needs water. The third banks a
        // fertilizer spent on an earlier day.
        m.water = must_water(t, day) || must_yield_water(t, day)
               || must_bank_water(t, day);
        m.feed  = must_feed(t, day);
        m.care  = m.feed && must_care_after_feed(t, day);
        m.collect = must_collect(t);
    } else if (MACRO_FORCE_PREWATER_ON_MUST_HARVEST && water_yields_now(t, day)) {
        m.prewater = true;
    }
    return m;
}

// A structure standing empty is one A_PLACE away from producing, which is a
// different situation from bare ground and from a stocked structure.
inline bool structure_place_ready(const tile& t) {
    return (t.type == T_COOP || t.type == T_PASTURE) && t.animal < 0;
}
// What A_DIG would accept. It never evicts a live animal.
inline bool tile_diggable(const tile& t) {
    return t.bought && t.type != T_EMPTY && t.animal < 0;
}

// A crude hour count for "what does this cell still want today", used as a
// per-cell channel and summed into a farm-wide work-demand scalar. It is NOT
// build_task -- there is no goal here yet -- it is the upper bound on the
// basic-care ops the filter could still accept on this tile.
//
// THIS MUST STAY A FUNCTION OF PUBLIC TILE STATE. farm_agg runs it over the
// OPPONENT's board as well, and the result lands in SC_OPP_FARM, so anything
// it reads is revealed. fertilize_pays reads crop, phase, yield_units and
// fertilized, all in the raw 0..27, so the fertilize term is public.
// plant_has_future reads crop, phase and max_lifespan_step, all of which are in
// the raw 0..27, so the water term stays public too.
inline int tile_pending_ops(const tile& t, int day) {
    if (!t.bought) return 0;
    int n = 0;
    if (harvest_ready(t, day)) ++n;
    // A water only counts if it keeps something worth keeping or pays today.
    if (t.type == T_PLANT && !t.watered_today &&
        (plant_has_future(t, day) || water_yields_now(t, day))) ++n;
    if (fertilize_pays(t, day)) ++n;
    if (t.type == T_WEED) ++n;                            // a dig
    if (t.animal >= 0) {
        if (!t.fed_today)   ++n;
        if (!t.cared_today) ++n;
        if (t.fertilizer_available) ++n;
    }
    if (structure_place_ready(t)) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
enum ActionType {
    A_MOVE_NORTH = 0, A_MOVE_SOUTH, A_MOVE_EAST, A_MOVE_WEST,
    A_PASS,
    A_WATER, A_HARVEST, A_FERTILIZE, A_DIG,
    A_BUILD_COOP, A_BUILD_PASTURE,
    A_FEED, A_COLLECT_FERTILIZER, A_CARE,
    A_DROP,      // item < 0 dumps everything; else `count` of `item`
    A_PICKUP,    // `count` of `item` out of the shed
    A_PLANT,     // `item` is a crop id
    A_PLACE,     // `item` is an ITEM id (GOOSE..SHEEP place, else shed deposit)
    A_BUY_SEED,    // `item` crop id,    `count` units
    A_BUY_ANIMAL,  // `item` animal id,  `count` units
    A_BUY_PRODUCT, // `item` WHEAT or FERTILIZER only
    A_SELL,        // `item` product id, `count` units
    A_HIRE,
    A_BUY_LAND,
    NUM_ACTION_TYPES
};

constexpr int MARKET_ACTOR = -1;
constexpr int NO_ITEM = -1;

inline bool is_market_action(int type) { return type >= A_BUY_SEED; }
inline bool is_move_action(int type)   { return type <= A_MOVE_WEST; }

struct move {
    int farmer_idx = 0;    // >= 0 hand index, or MARKET_ACTOR
    int type       = A_PASS;
    int item       = NO_ITEM;
    int count      = 1;
};

inline move act(int idx, int type)              { return { idx, type, NO_ITEM, 1 }; }
inline move act_plant(int idx, int crop)        { return { idx, A_PLANT, crop, 1 }; }
inline move act_place_item(int idx, int item)   { return { idx, A_PLACE, item, 1 }; }
inline move act_pickup(int idx, int item, int n = 1) { return { idx, A_PICKUP, item, n }; }
inline move act_drop_all(int idx)               { return { idx, A_DROP, NO_ITEM, 0 }; }
inline move act_drop(int idx, int item, int n)  { return { idx, A_DROP, item, n }; }

inline move order_buy_seed(int crop, int n = 1)   { return { MARKET_ACTOR, A_BUY_SEED,    crop, n }; }
inline move order_buy_animal(int a, int n = 1)    { return { MARKET_ACTOR, A_BUY_ANIMAL,  a,    n }; }
inline move order_buy_product(int item, int n = 1){ return { MARKET_ACTOR, A_BUY_PRODUCT, item, n }; }
inline move order_sell(int item, int n = 1)       { return { MARKET_ACTOR, A_SELL,        item, n }; }
inline move order_hire()                          { return { MARKET_ACTOR, A_HIRE,     NO_ITEM, 1 }; }
inline move order_buy_land()                      { return { MARKET_ACTOR, A_BUY_LAND, NO_ITEM, 1 }; }

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
struct ipos {
    int x = 0, y = 0;
    bool operator==(const ipos& o) const { return x == o.x && y == o.y; }
};

int  quadrant_of(int x, int y);
bool is_shed_adjacent(const ipos& p);
const std::array<ipos, 4>& shed_access_tiles();   // NW, NE, SW, SE order
ipos default_spawn();

// Hands may roam up to MAX_SHED_DIST tiles (Chebyshev) from the 2x2 shed block
// on OWNED land. Locked tiles are walkable only within LOCKED_STEP_DIST, so a
// hand standing on a locked shed-access tile is never stranded.
constexpr int MAX_SHED_DIST    = 6;
constexpr int LOCKED_STEP_DIST = 1;

int shed_distance(int x, int y);

// ---------------------------------------------------------------------------
// Per-player state
// ---------------------------------------------------------------------------
struct player_state {
    double money = STARTING_MONEY;
    tile   board[BOARD_SIZE][BOARD_SIZE];

    std::vector<ipos>     farmers;             // [0] == the hand that survives the night
    std::vector<unit_inv> carried;             // parallel to farmers

    std::array<int, NUM_ITEMS> shed{};
    std::array<int, NUM_CROPS> seeds{};

    std::vector<int> unlocked_quadrants;       // starts { Q_NW }
    int hires_today = 0;

    // Logging only. Nothing scores on these any more.
    int stat_harvests = 0;
    int stat_water_feed = 0;
    int stat_plants = 0;
    int stat_fertilized = 0;

    int  shed_total() const;
    bool quadrant_unlocked(int q) const;
};

// ---------------------------------------------------------------------------
// THE SHED IS TWO CONSTRAINTS, NOT ONE
// ---------------------------------------------------------------------------
// SHED_CAPACITY is a single counter over all NUM_ITEMS, but the referee tests
// it at two moments that have nothing to do with each other:
//
//   HOUR 0        commit_unit refuses an A_BUY_PRODUCT / A_BUY_ANIMAL when
//                 shed_total() >= SHED_CAPACITY. Harvests are hours away and
//                 do not compete here at all.
//   NIGHTFALL     drop_inventories_to_shed runs at end_of_day and
//                 dump_inventory DISCARDS whatever does not fit
//                 (shed_overflows++). This is what the day's harvests, and
//                 every item a hand picked up but never spent, compete for.
//
// The filter keeps one exact counter for each (`room` and `night` in
// macro_plan_day). Neither is an estimate: within a day nothing but this
// player touches this player's shed -- commit_unit only ever indexes
// players[p], town_consume moves market inventory, and the nightly refresh
// runs after the dump -- so both totals are decidable in advance. See the
// ledger at the top of macro_plan_day for the arithmetic.
// ---------------------------------------------------------------------------

struct market_state {
    std::array<int, NUM_PRODUCTS> inventory{};
    std::array<int, NUM_PRODUCTS> prices{};
};

// ---------------------------------------------------------------------------
// Pricing (free functions so the filter can quote without a sim)
// ---------------------------------------------------------------------------
double shape_f(Shape f, double x, double T);
int    market_price(int item, int inventory);
// What n units of a market-bought PRODUCT actually cost / fetch, walking the
// price the way process_market will rather than pretending n * spot.
double buy_cost_walk(int item, int n, int inventory);
double sell_proceeds(int item, int n, int inventory);

// ===========================================================================
// MACRO OUTPUT LAYOUT
// ===========================================================================
// The spatial block is one 1x1 conv of 20 channels over the 10x10 trunk map,
// reshaped [B,C,10,10] -> [B,C*100], so indexing is channel-major:
//     base + ch * CELLS + y * BOARD_SIZE + x
//
// The globals come off the pooled aux joint: land, the nine sell
// decisions and the eight per-type COUNTS have no spatial content.
//
// THE WIDTH CHANGED again: the per-cell primary head is GONE (-9 channels).
// WHAT goes down is the TYPE SLOTS' call (up to four types, in drawn order,
// Don't_Plant stops early), HOW MANY is that type's COUNT head (halved bins:
// 1,2,4..24 crops, 1,2,4,6 animals), and WHERE is a fixed rule: animals on cells touching the
// shed first, then next to the herd, nearest first; tomato / strawberry /
// melon nearest the shed, wheat and carrot from the far end.
// MACRO_SPATIAL_CH 29 -> 20, MACRO_DIM 3042 -> 2197. `spatial` and every
// count head changed shape; load_compatible keeps every other weight.
//
//   harvest    ONE [No_Harvest, Harvest] PAIR PER TYPE -- see below
//   fertilize  [No_Fert, Fert], drawn AFTER planting, only on a plant where
//              fertilize_pays -- standing, or planted this morning. Accepting
//              one drags the payout-day water in with it.
//   feed       A_FEED on an animal that is not already forced. Accepting one
//              drags a FORCED A_CARE in with it: two hand-hours, one decision.
//
// Water, care and collect have no heads. Water follows from the must tier, the
// planting and the fertilize; care rides on every feed; collect is forced on
// every animal with fertilizer waiting.
//
// ---------------------------------------------------------------------------
// THE HARVEST HEAD IS PER TYPE
// ---------------------------------------------------------------------------
// It used to be one shared [No, Yes] pair that every ripe cell drew from, so
// "hold the tomatoes because the price is low" and "bank the wheat now" had to
// be expressed through the same two logits, separated only by whatever the
// trunk could encode about the cell. The hold/bank call is almost entirely a
// PER-PRODUCT market call -- tomato is SH_HINGE and convex below target, wheat
// is feed as well as goods, an animal's product refills on a fixed interval
// while a non-ongoing crop's harvest wipes the tile -- so the pair is now
// per type: 8 pairs, one for each of the five crops and three animals, indexed
// by tile_harvest_type.
//
// EVERY TRIPLE IS STILL H_HARVEST. They are separate CHANNELS, not separate
// HeadIds, so entropy targeting stays pooled over all of them and
// `entropy_file`'s existing `harvest` line keeps working unchanged.
//
// ---------------------------------------------------------------------------
// THE HARVEST HEAD HAS THREE BINS, AND A SELL-TIME HEAD RIDES ON THE THIRD
// ---------------------------------------------------------------------------
//   HB_NONE   leave it standing
//   HB_KEEP   harvest; the units ride in hand to the nightly sweep and are the
//             next morning's sell head's business (the old "Harvest")
//   HB_SELL   harvest, DROP the units at the shed mid-day, and SELL them at an
//             hour the SELL-TIME head picks
//
// The sell-time head is one per-cell group of MACRO_SELL_TIME_BINS logits,
// shared by every type (the cell's crop / animal one-hot already tells the
// trunk what is being sold): MACRO_SELL_HOUR_BINS hours, then two fallbacks.
//
//   hour bins     the hour is a DEADLINE for the drop -- the hand may drop
//                 earlier -- and the EXACT hour of the sell order
//   ST_NO_HARVEST leave the cell standing after all ("Sell, but only at an
//                 hour I like -- otherwise not at all")
//   ST_KEEP       harvest it and keep the units for the nightly sweep ("Sell
//                 if a good hour is open, otherwise keep")
//
// Unreachable hours are masked. WHEN EACH FALLBACK IS OFFERED:
//
//   optional harvest, first draw   hours, NO_HARVEST, KEEP (if the night has
//                                  room)
//   forced harvest, first draw     hours, KEEP (a forced harvest cannot be
//                                  left standing)
//   remask after a failed sale     the hours that STILL route on the plan as
//                                  it stands, KEEP, and NO_HARVEST only for an
//                                  optional harvest nothing was built on since
//                                  (the cell's goal is unchanged since the
//                                  harvest pass, and releasing it routes)
//
// A set with one option left is taken without a draw. A sale that is the ONLY
// way to bank a forced harvest (the night shed is already over capacity) is
// offered its hours alone; if none route, it stays kept and overflows.
//
// A FORCED harvest (must_harvest, the ripe melon) is not a decision, but how
// it is banked is: such a cell draws [HB_KEEP, HB_SELL] from the same triple,
// with HB_NONE masked.
//
// WHY THESE HOURS. town_consume runs AFTER process_market, and the shops drain
// at every turn with turn % 4 == 0 -- hours 0, 4, 8, 12, 16, 20 of every day.
// A drain lowers market inventory, which RAISES the price, so the first hour
// after a drain (4k+1) is the earliest hour that sees it. Selling at 4k+1
// strictly dominates selling at 4k (same drain count plus one, one more hour
// of slack), so the bins are exactly those hours, plus 23: the same price
// regime as 21 (no drain in between) but two more hours of route slack.
// Nothing earlier than 5 is offered: the fastest possible round trip
// (out, harvest, back, drop) lands at hour 4 only for a hand that starts at
// hour 1 next to the shed, and 5 then sees one more drain.
// ===========================================================================
constexpr int MACRO_HARVEST_TYPES = NUM_CROPS + NUM_ANIMALS;          // 8
enum HarvestBin { HB_NONE = 0, HB_KEEP = 1, HB_SELL = 2, MACRO_HARVEST_BINS = 3 };
constexpr int MACRO_HARVEST_CH   = MACRO_HARVEST_BINS * MACRO_HARVEST_TYPES;  // 24

constexpr int MACRO_SELL_HOUR_BINS = 12;
constexpr int MACRO_SELL_HOURS[MACRO_SELL_HOUR_BINS] = { 5,6,7,8, 9,10 ,13,15 ,17,19, 21, 23 };
constexpr int ST_NO_HARVEST        = MACRO_SELL_HOUR_BINS;            // 12
constexpr int ST_KEEP              = MACRO_SELL_HOUR_BINS + 1;        // 13
constexpr int MACRO_SELL_TIME_BINS = MACRO_SELL_HOUR_BINS + 2;        // 14
constexpr int MACRO_SELL_TIME_CH   = MACRO_SELL_TIME_BINS;            // 14

constexpr int MACRO_FERT_CH      = 2;    // [No_Fert, Fert]
constexpr int MACRO_FEED_CH      = 2;    // [No_Feed, Feed]

// No primary channels: WHAT gets planted / placed is decided by the COUNT
// heads (global), and WHERE by a fixed distance rule in the filter.
constexpr int MACRO_SPATIAL_CH   = MACRO_HARVEST_CH + MACRO_SELL_TIME_CH
                                 + MACRO_FERT_CH + MACRO_FEED_CH;     // 42

constexpr int MACRO_HARVEST_BASE   = 0;
constexpr int MACRO_SELL_TIME_BASE = MACRO_HARVEST_BASE   + MACRO_HARVEST_CH   * CELLS;  // 2400
constexpr int MACRO_FERT_BASE      = MACRO_SELL_TIME_BASE + MACRO_SELL_TIME_CH * CELLS;  // 3800
constexpr int MACRO_FEED_BASE      = MACRO_FERT_BASE      + MACRO_FERT_CH      * CELLS;  // 4000
constexpr int MACRO_LAND_BASE      = MACRO_FEED_BASE      + MACRO_FEED_CH      * CELLS;  // 4200

// NO HIRE HEAD. Hands are hired on demand by the passes (see the filter).

// ---- SELL, one head per product, drawn at the top of the day --------------
// Sells the MORNING STOCK at hour 0 -- which includes everything the hands
// carried home at last night's sweep.
//   every product but wheat   bin 0 = hold, bin 3 = sell all (binary; bins
//                             1 and 2 are never offered)
//   wheat                     bin 0 = hold; bins 1 / 2 / 3 = sell all but the
//                             feed for the next 3 / 2 / 1 days
constexpr int MACRO_SELL_BINS = 4;
inline int macro_wheat_keep_days(int bin) {       // bins 1..3 only
    return bin == 1 ? 3 : bin == 2 ? 2 : 1;
}
constexpr int MACRO_SELL_BASE = MACRO_LAND_BASE + 2;                    // 4202

// ---- COUNT, one head per plantable / placeable type -----------------------
// HOW MANY of each type to put down today. The bins are HALVED past 2: bin b
// means
//
//     b        0  1  2  3  4  5  6 ...
//     units    0  1  2  4  6  8 10 ...        (macro_count_value)
//
//   crops    0..MACRO_PLANT_MAX = 24   -> 14 bins
//   animals  0..MACRO_PLACE_MAX =  6   ->  5 bins
//
// THE TOP OFFERED BIN IS "AS MANY AS FIT". The filter offers every bin whose
// value is below today's ceiling, plus the first bin at or above it, and that
// last bin places exactly `ceiling` units. So with room for 5 the choices are
// 1, 2, 4 and 5 (drawn from the "6" bin): filling the day is always one
// choice away, and an odd ceiling is never unreachable.
constexpr int MACRO_PLANT_MAX   = 24;
constexpr int MACRO_PLACE_MAX   = 6;
constexpr int NUM_COUNT_TYPES   = NUM_CROPS + NUM_ANIMALS;              // 8
constexpr int macro_count_value(int bin) { return bin <= 2 ? bin : 2 * (bin - 1); }
constexpr int macro_bins_for_max(int m)  { return m < 2 ? m + 1 : m / 2 + 2; }
constexpr int MACRO_PLANT_BINS  = macro_bins_for_max(MACRO_PLANT_MAX);   // 14
constexpr int MACRO_PLACE_BINS  = macro_bins_for_max(MACRO_PLACE_MAX);   //  5
constexpr int MACRO_COUNT_BINS_MAX = MACRO_PLANT_BINS;                   // 14
constexpr int macro_count_bins(int type) {
    return type < NUM_CROPS ? MACRO_PLANT_BINS : MACRO_PLACE_BINS;
}
constexpr int MACRO_COUNT_WIDTH = NUM_CROPS   * MACRO_PLANT_BINS
                                + NUM_ANIMALS * MACRO_PLACE_BINS;       // 85
constexpr int MACRO_COUNT_BASE  = MACRO_SELL_BASE + NUM_PRODUCTS * MACRO_SELL_BINS;  // 4238
static_assert(macro_count_value(MACRO_PLANT_BINS - 1) == MACRO_PLANT_MAX, "plant bins");
static_assert(macro_count_value(MACRO_PLACE_BINS - 1) == MACRO_PLACE_MAX, "place bins");

// ---- TYPE SLOTS: WHAT to put down today, up to MACRO_TYPE_SLOTS types -------
// The count heads alone made "plant only wheat" a joint event over eight
// independent draws: wheat had to land on a non-zero bin AND every other
// type on bin 0, so exploring one type meant getting lucky on all the rest.
//
// Now the day draws up to four TYPE SLOTS, in order. Each slot is one
// [Don't_Plant, wheat, carrot, tomato, strawberry, melon, goose, cow, sheep]
// choice:
//   * a type already chosen by an earlier slot is masked, and so is any type
//     the cells / money LEFT by the earlier slots cannot support at all;
//   * a type chosen here is placed at once, so the next slot's mask reads
//     the board and the budget after it;
//   * Don't_Plant ends the day's planting: the later slots are not drawn.
// Once a slot names a type, that type's COUNT head draws how many, over bins
// 1.. up to the one covering cap (bin 0 is masked: "none" is the slot's
// Don't_Plant, not the count's).
//
// Appended at the END of the output, so every existing head keeps its index
// and load_compatible keeps every existing weight; only type_heads start fresh.
constexpr int MACRO_TYPE_SLOTS  = 4;

// THE DAY-0 OPENING: FORCED, IN THIS ORDER, BEFORE ANY TYPE SLOT. On day 0
// the filter puts these down first -- no choice set, no log-prob, no entropy,
// the melon market mask bypassed -- each type in the listed order, on the
// cells its normal placement rule picks (animals anchored at the shed, the
// herd growing outward; melons nearest the shed of what is left). Each unit
// hires to fit like any forced planting; a unit nothing affordable makes fit
// is skipped. Only AFTER all of them do the type slots draw, on whatever
// cells and money are left. With MACRO_DAY0_FORCED_CLOSES_SLOTS the forced
// types are closed to day 0's slots. `type` is a count type:
// 0..NUM_CROPS-1 a crop, NUM_CROPS + a an animal. A count of 0 disables a row.
struct Day0Forced { int type; int count; };
constexpr int count_type_animal(int item) { return NUM_CROPS + (item - GOOSE); }
constexpr Day0Forced MACRO_DAY0_FORCED[] = {
    { count_type_animal(SHEEP), 3 },
    { count_type_animal(COW),   2 },
    { MELON,                    5 },
};
constexpr int  MACRO_DAY0_FORCED_N = (int)(sizeof(MACRO_DAY0_FORCED) / sizeof(MACRO_DAY0_FORCED[0]));
constexpr bool MACRO_DAY0_FORCED_CLOSES_SLOTS = true;
constexpr int MACRO_TYPE_NONE   = 0;                                    // bin 0
constexpr int MACRO_TYPE_BINS   = NUM_COUNT_TYPES + 1;                  // 9
constexpr int MACRO_TYPE_BASE   = MACRO_COUNT_BASE + MACRO_COUNT_WIDTH; // 4323
constexpr int MACRO_TYPE_WIDTH  = MACRO_TYPE_SLOTS * MACRO_TYPE_BINS;   // 36
constexpr int MACRO_DIM         = MACRO_TYPE_BASE + MACRO_TYPE_WIDTH;   // 4359

static_assert(MACRO_DIM == 4359, "macro output width");
static_assert(MACRO_HARVEST_TYPES == NUM_COUNT_TYPES,
              "harvest pairs and count ceilings index the same type space");

inline int macro_sell_count(int bin, int stack) {
    if (bin <= 0 || stack <= 0) return 0;
    if (bin >= MACRO_SELL_BINS - 1) return stack;
    const int den = MACRO_SELL_BINS - 1;
    const int n = (stack * bin + den / 2) / den;     // nearest, not floor
    return std::clamp(n, 1, stack);
}
inline int macro_sell_index(int product, int bin) {
    return MACRO_SELL_BASE + product * MACRO_SELL_BINS + bin;
}
inline int macro_cell_index(int base, int ch, int cell) {
    return base + ch * CELLS + cell;
}
// `bin` is a HarvestBin: HB_NONE, HB_KEEP or HB_SELL.
inline int macro_harvest_index(int type, int bin, int cell) {
    return macro_cell_index(MACRO_HARVEST_BASE, type * MACRO_HARVEST_BINS + bin, cell);
}
// `bin` is an hour bin (0..MACRO_SELL_HOUR_BINS-1), ST_NO_HARVEST or ST_KEEP.
inline int macro_sell_time_index(int bin, int cell) {
    return macro_cell_index(MACRO_SELL_TIME_BASE, bin, cell);
}
// bin MACRO_TYPE_NONE is Don't_Plant; bin 1 + k is count type k.
inline int macro_type_index(int slot, int bin) {
    return MACRO_TYPE_BASE + slot * MACRO_TYPE_BINS + bin;
}
inline int macro_count_index(int type, int bin) {
    const int off = (type < NUM_CROPS)
        ? type * MACRO_PLANT_BINS
        : NUM_CROPS * MACRO_PLANT_BINS + (type - NUM_CROPS) * MACRO_PLACE_BINS;
    return MACRO_COUNT_BASE + off + bin;
}

// ---- filter knobs ---------------------------------------------------------
// THE CREW CAP: the most hands a day may have, hand 0 included (so at most
// 15 hires). Every hire is made on demand by a pass whose drawn work does not
// fit the routes; this is the only upper bound besides the money. It cannot
// go higher: simulation::do_hire refuses a 17th hand, and MacroPlanner holds
// at most MAX_UNITS routes.
constexpr int MACRO_HIRE_HARD_MAX = 16;
static_assert(MACRO_HIRE_HARD_MAX <= MAX_UNITS, "crew cap above the planner's limit");

// Worst-case order entries in a day: 9 sells + 1 land + 1 hire (ONE A_HIRE
// carrying the count) + 5 seeds + wheat + fertilizer + 3 animals == 21, so
// three turns of MAX_MARKET_ORDERS_PER_TURN. Four is kept as the cap.
//
// THE HIRE IS ONE ENTRY ON PURPOSE. process_market runs `count` do_hire calls
// for a single A_HIRE, so however many hands the day ends up hiring -- the
// hires every pass makes on demand, in whatever order they come -- the
// order hours are the same, and routes already priced against them stay
// valid.
constexpr int MACRO_MAX_ORDER_HOURS = 4;

// The most entries the DERIVED buys can ever come to: one per seed type, one
// for wheat, one for fertilizer, one per animal type. Each is a single entry
// carrying a count, so this is a hard bound rather than an estimate -- which
// is what lets the order hours be priced on the real hire and land decisions
// instead of on the largest crew the day could have afforded.
constexpr int MACRO_MAX_BUY_ENTRIES = NUM_CROPS + 2 + NUM_ANIMALS;   // 10

// Whether a standing EMPTY structure may be dug to plant on it, or to put the
// other structure's animal there. T_WEED is clearable either way.
constexpr bool MACRO_ALLOW_DIG_EMPTY_STRUCTURES = true;

// Whether a DOOMED plant counts as clearable ground for the primary pass. It
// is a weed by morning either way (plant_is_doomed), and the ongoing crops are
// why this matters: without it a tomato that finished producing last night
// holds its cell for a whole extra day -- one day to decay into a weed, one to
// dig it -- while the filter refused to plant there because the tile still read
// T_PLANT. build_task emits the A_DIG itself. The cell is offered once nothing
// is left UNBANKED on it: either it holds nothing, or today's harvest of it is
// already routed, in which case the harvest rides ahead of the dig as
// want_pre_harvest -- HARVEST, DIG, PLANT on one visit.
constexpr bool MACRO_REPLANT_DOOMED = true;

// A freshly planted crop carries consecutive_unwatered == 1, so an unwatered
// first night turns it straight to weeds. Free and never wrong, so the primary
// pass reserves the water together with the plant.
constexpr bool MACRO_FORCE_WATER_ON_PLANT = true;

// Wheat is feed. The wheat sell head only ever sees the part of the stack
// above this reserve -- except on the last day, when feed is worthless and
// the whole stack is offered.
constexpr int MACRO_WHEAT_RESERVE = 0;

// ---- HEADS ----------------------------------------------------------------
// Every choice set is tagged with the head that produced it, so entropy can be
// targeted per head at train time. A 15-way primary draw and a 2-way land draw
// want opposite amounts of noise, and averaging them into one number lets
// whichever head fires most often set the exploration for all of them.
//
// Order here IS the order of the numbers in `entropy_file`. The file is parsed
// BY NAME, so a name that no longer draws simply has no effect.
//
// H_WATER IS DEAD. It is retained so that NUM_HEADS and the HEAD_NAMES table
// stay parallel and so an existing entropy_file keeps landing every other
// value where it is named. Nothing draws it; it prints "--" for the rest of
// the project's life. Delete it only together with the table in main.cpp.
//
// H_FERTILIZE IS LIVE AGAIN: the per-cell [No_Fert, Fert] pair.
//
// H_HIRE IS DEAD too: hands are hired on demand. Kept for the same reason as
// H_WATER.
//
// THE FOUR TYPE SLOTS ARE FOUR HEADS NOW. Slot 0 is drawn every day there is
// anything to plant, slot 1 only when slot 0 named a type, slot 2 only when
// slot 1 did too, and so on -- so slot 0 sees several times the sets slot 3
// does, and a single pooled target let the busiest slot set the noise for
// all four. Each slot has its own entry in `entropy_file`
// (plant_type_1 .. plant_type_4); a bare `plant_type` line still sets all four.
//
// WHICH MODEL A HEAD READS is head_is_global(), not an index threshold: the
// sell-time head is a PER-CELL head appended after the global ones.
enum HeadId {
    H_HARVEST = 0,     // the per-type [None, Keep, Sell] triples, pooled
    H_WATER,           // DEAD
    H_PRIMARY,         // DEAD: the per-cell primary head is gone (counts decide)
    H_FERTILIZE,       // [No_Fert, Fert], per cell where fertilize_pays
    H_FEED,
    H_LAND,
    H_HIRE,            // DEAD: no hire head, hands are hired on demand
    H_SELL,
    H_PLANT_N,         // how many of each crop, 1..MACRO_PLANT_MAX once chosen
    H_PLACE_N,         // how many of each animal, 1..MACRO_PLACE_MAX once chosen
    H_PLANT_TYPE_0,    // type slot 1 (drawn most)
    H_PLANT_TYPE_1,    // type slot 2
    H_PLANT_TYPE_2,    // type slot 3
    H_PLANT_TYPE_3,    // type slot 4 (drawn least)
    H_SELL_TIME,       // per cell: the hour an HB_SELL harvest is sold at,
                       // or NO_HARVEST / KEEP
    NUM_HEADS
};

// Per-head play mode for NON-RECORDING calls (evals, opponents).
// HM_FOLLOW (0, the default) obeys the call's `greedy`. Set once before
// planning; only read inside the omp loops, so it is thread-safe. Calls that
// record draws (training rollouts) ignore it.
enum HeadMode : uint8_t { HM_FOLLOW = 0, HM_SAMPLE = 1, HM_GREEDY = 2 };
extern uint8_t g_head_mode[NUM_HEADS];
static_assert(H_PLANT_TYPE_3 - H_PLANT_TYPE_0 + 1 == MACRO_TYPE_SLOTS,
              "one type-slot head per slot");
inline int head_plant_type(int slot) { return H_PLANT_TYPE_0 + slot; }

// Which model's logits a head reads: trunk_cell (per-cell heads) or the
// global_critic policy body. The trainer takes per-model policy and entropy
// means off this, so it must agree with NetPolicyImpl::forward.
inline bool head_is_global(int h) {
    switch (h) {
        case H_LAND: case H_SELL: case H_PLANT_N: case H_PLACE_N:
        case H_PLANT_TYPE_0: case H_PLANT_TYPE_1:
        case H_PLANT_TYPE_2: case H_PLANT_TYPE_3:
            return true;
        default:
            return false;
    }
}

// ===========================================================================
// THE CELL LEDGER HOOK  --  what happened, to which cell, at what price
// ===========================================================================
// The referee does not keep books. When `simulation::ledger` is set it only
// REPORTS the events per-cell credit assignment needs, in the order they
// happen; nothing it does changes by being watched. cell_ledger.hpp turns the
// stream into per-cell daily rewards.
//
//   LE_HARVEST   units of `item` left `cell` (A_HARVEST); origin = planted /
//                placed day
//   LE_COLLECT   one FERTILIZER left an animal `cell`; origin = placed day
//   LE_CONSUME   one `item` was eaten / spent ON `cell` (A_FEED wheat,
//                A_FERTILIZE fertilizer), at `price`, the spot price then
//   LE_COST      `cell` paid `price` coins for what now stands on it (the seed
//                at A_PLANT, the animal at A_PLACE)
//   LE_SELL      one unit of `item` sold for `price`
//   LE_BUY       one unit of `item` bought for `price` (wheat / fertilizer)
// ===========================================================================
enum LedgerKind : int8_t {
    LE_HARVEST = 0, LE_COLLECT, LE_CONSUME, LE_COST, LE_SELL, LE_BUY
};

struct LedgerEvent {
    int8_t  kind;
    int8_t  player;
    int8_t  item;       // -1 where it does not apply (LE_COST)
    int16_t cell;       // -1 for market events
    int16_t origin;     // planted / placed day, LE_HARVEST and LE_COLLECT only
    int32_t n;
    int32_t price;
    int32_t turn;
};

// ===========================================================================
// The simulation
// ===========================================================================
class simulation {
public:
    int  turn = 0;
    bool log  = false;
    std::string file_name;
    std::ofstream outFile;

    player_state players[NUM_PLAYERS];
    market_state market;
    std::vector<int> unlocked_shops;   // Shop ids, with replacement

    simulation();
    explicit simulation(std::string file_name, bool log_game, uint64_t seed = 0);

    // Returns -1 while the episode runs, else 0 / 1 / 2 from check_winner.
    int apply_turn(std::vector<move> moves[NUM_PLAYERS]);

    double weed_chance = WEED_SPAWN_CHANCE;
    bool   auto_shops  = true;
    bool   auto_log    = true;

    // GHOST SEATS -- a recorded tape playing through this sim (tape.hpp).
    // A ghost player's buys, hires and land purchases never fail for money
    // (money may go negative), and it gets no random weeds: its weeds are
    // adopted from the recording each turn instead. Everything else -- the
    // board, the shed, the shared market it sells into -- is live. Cleared by
    // reset(), so a reused simulation never inherits a ghost.
    bool   ghost[NUM_PLAYERS] = { false, false };

    // THE CELL LEDGER HOOK. nullptr (the default) reports nothing. Not owned,
    // not touched by reset(): whoever sets it clears it.
    std::vector<LedgerEvent>* ledger = nullptr;

    long dropped_no_unit = 0;
    long plant_groups_blocked = 0;
    long shed_overflows = 0;

    void enable_log(const std::string& path);
    void log_frame();

    int  day()  const { return turn / TURNS_PER_DAY; }
    int  hour() const { return turn % TURNS_PER_DAY; }
    void reset(uint64_t seed = 0);

    // Most money on the last night. 0, 1, or 2 for a true draw.
    int  check_winner() const;

    double net_worth(int p) const;

private:
    uint64_t seed_ = 0;

    void report(int kind, int p, int item, int cell, int origin, int n, int price) {
        if (!ledger) return;
        ledger->push_back({ (int8_t)kind, (int8_t)p, (int8_t)item, (int16_t)cell,
                            (int16_t)origin, (int32_t)n, (int32_t)price, (int32_t)turn });
    }

    void store_to_shed(int p, int idx, int item, int n);
    void dump_inventory(int p, unit_inv& inv);
    void apply_unit_action(int p, int idx, const move& m);
    void apply_farmer_moves(std::vector<move> moves[NUM_PLAYERS]);
    void process_market(std::vector<move> moves[NUM_PLAYERS]);
    void town_consume();
    void decay_plants(int p);
    void end_of_day();

    void daily_refresh_plants(int p);
    void daily_refresh_animals(int p);
    void spawn_weeds(int p, std::mt19937& rng);
    void drop_inventories_to_shed(int p);

    bool commit_unit(int p, int kind, int item, int price);
    int  quote_unit(int kind, int item) const;
    void do_hire(int p);
    void do_buy_land(int p);
    ipos spawn_hand(int p) const;

    void refresh_prices();
    void refresh_farmer_marks();
    void write_log();
    void write_player_block(const player_state& ps);
};

// ===========================================================================
// OBSERVATION  --  channel layout
// ===========================================================================
// ASYMMETRIC ON PURPOSE. The own board is read cell by cell by the spatial
// head, so an engineered channel there lands on the exact decision it informs.
// The opponent board is pooled to 2x2 and squeezed through opp_fc into 20
// numbers before anything sees it, so a per-cell derived channel over there
// would be averaged away -- it would cost memory and buy nothing. The
// opponent therefore keeps the RAW 28, and everything derived is own-side.
//
// OWN BOARD, channels 0 .. OWN_TILE_CH-1
//    ---- raw, mostly unchanged -----------------------------------------
//    0-4    tile type one-hot
//    5      bought
//    6      hands standing here / 4
//    7-11   crop one-hot
//    12     age / 30
//    13     consecutive_unwatered
//    14     fertilizer days left / 2
//    15     lifespan (1.0 for immortal / ongoing)
//    16-18  animal one-hot
//    19     fed_today            22  fertilizer_available
//    20     cared_today          23  consecutive_unfed
//    21     FUTURE FERTILIZER APPLICATIONS the canonical schedule
//           (fert_day_for) still has room for / 2 -- NOT watered_today.
//           A hint of how much fertilizer value is left in the cell;
//           watered_today is still available derived, at 42 and 36.
//    24     pending_care_bonus / 5
//    25     yield_units / 6
//    26     harvest_ready     -- the harvest head's mask, precomputed. WHICH
//                                of the eight per-type pairs the cell draws
//                                from is tile_harvest_type, which the crop
//                                one-hot (7-11) and the animal one-hot (16-18)
//                                already carry, so no channel was added.
//    27     MUST HARVEST      -- ripe and lost by morning: the harvest head is
//                                masked OFF this cell.
//    ---- derived: state of the crop / animal ---------------------------
//    28     ongoing crop flag
//    29     yield_units / max_yield        (fullness, not absolute count)
//    30     headroom / max_yield           (what another unit could land in)
//    31     days until first yield / 10    (0 once it is producing)
//    32     decaying now: turn >= max_lifespan_step
//    33     standing yield value at spot / norm
//    34     spot price of this cell's product / base
//    ---- derived: what tonight and this hour do ------------------------
//    35     produces tonight  (ongoing crop / animal end_of_day arithmetic)
//    36     watering NOW yields a unit (non-ongoing yield window)
//    37     plant dies tonight if not watered
//    38     animal escapes tonight if not fed
//    39     care bonus BANKS tonight (cared AND fed -- the bank, not the cash)
//    ---- derived: what can be done here --------------------------------
//    40     bare structure, one PLACE from producing
//    41     diggable
//    42     MUST WATER   -- forced: life support ON A PLANT WITH A FUTURE, the
//                           yield water, or banking a fertilizer. No head
//                           reads it; there is no water head. It is what tells
//                           the primary head how much of a cell's day is
//                           already committed. Read together with 32 and 37: a
//                           tile that dies tonight and is NOT must-watered is
//                           a doomed cell, and the primary head may replant it
//                           -- after the harvest, if it still holds units.
//    43     MUST FEED    -- forced, so the feed head is masked off this cell
//    44     pending ops estimate / 4
//    ---- derived: economics and geometry -------------------------------
//    45     seed cost of the occupying crop / 100
//    46     shed_distance / MAX_SHED_DIST   (the planner's travel cost)
//    47     walkable: owned and inside MAX_SHED_DIST
//    48     locked, and in the NEXT quadrant LAND_ORDER would unlock
//    49-52  quadrant one-hot
//    53     max_yield / 6                   (static per crop/animal)
//    54     interval / 3
//    55     first_yield_day / 12
//    ---- derived: fertilizer and care ----------------------------------
//    56     FERTILIZE PAYS   -- fertilize_pays on the hour-0 tile: the mask the
//                               fertilize head is drawn under on a standing
//                               plant. (A plant sown this morning is judged on
//                               its fresh tile inside the filter.)
//    57     FEED BRINGS CARE -- animal, not yet cared for today. No head is
//                               masked on it; it is what tells the FEED head
//                               that a feed here also buys a forced care.
//    58     FERTILIZER READ TONIGHT -- a live cover lands on tonight's payout,
//                               so the bank water is already forced (42).
//
// OPPONENT BOARD, channels CH_OPP_0 .. +OPP_TILE_CH-1: the raw 0-27 above.
// ===========================================================================
constexpr int OWN_TILE_CH = 59;
constexpr int OPP_TILE_CH = 28;

constexpr int CH_OWN_0     = 0;
constexpr int CH_OPP_0     = CH_OWN_0 + OWN_TILE_CH;              // 59
constexpr int CH_SHED_MASK = CH_OPP_0 + OPP_TILE_CH;              // 87, static geometry

// Six scalar planes. The cell trunk reads the first 161 features, the
// merged global/critic MLP reads the policy half (161 + the 123-feature
// forecast block = 284), and its value head alone reads the critic tail, which
// with the projection block (projection.hpp) takes the total to 503 -- three
// past what five planes hold, hence the sixth. A plane costs 400 bytes an
// observation.
constexpr int N_SCALAR_PLANES = 6;
constexpr int CH_SCALAR_0     = CH_SHED_MASK + 1;                 // 88
constexpr int NUM_INPUT_CH    = CH_SCALAR_0 + N_SCALAR_PLANES;    // 94

// ---------------------------------------------------------------------------
// Scalar features
// ---------------------------------------------------------------------------
// The planes are read row-major as ONE flat vector: index == plane*100 +
// row*10 + col, which is exactly how the trunk reshapes and slices them.
//
// Section anchors rather than bare literals. The encoder seeks to each anchor
// before writing it, so adding a feature inside a section is a one-line change
// to that section's size and nothing downstream shifts silently.
//
//  MONEY    11  own / opp / diff at three magnitudes each, log money, money
//               per hand
//  CREW      9  crew sizes, quadrants owned, hires_today, next land price and
//               whether it is affordable, next hand cost, HOW MANY hands are
//               affordable right now
//  SHED     21  12 item counts, total, room left, shed value at two scales,
//               5 seed counts
//  CLOCK     9  day, hour, days remaining, endgame flag, and one
//               still-plantable flag per crop
//  MARKET   36  inventory deviation, price / base, price / 250, and the
//               SLIPPAGE on dumping the whole own stack
//  TOWN     20  per-tile drain, hour-0 drain, shop instances, days until the
//               next shop unlock
//  COSTS     8  seed and animal prices
//  NETWORTH  3  own net worth, three scales
//  FARM     31  the own board summarised, including the OWN per-crop and
//               per-animal cell counts
//  OPP_FARM 13  the opponent's board: 5 summary numbers plus its per-crop
//               (5) and per-animal (3) cell counts. The opponent board is
//               public, so these reveal nothing the raw planes do not.
//
// ---- the cell trunk reads up to here (SCALAR_DIM_CELL = 161) ----
//
//  FCAST   123  THE BOARD, AS AN MLP CAN READ IT (forecast.hpp). The global
//               heads no longer see a conv map, so the board reaches them as
//               arithmetic:
//                 per player (own, then opponent), per product (9):
//                   units harvestable TODAY, by TOMORROW, by DAY+3, and over
//                   the LIFETIME (crops: until the plant is done or the game
//                   ends; animal products and fertilizer: the next 10 days),
//                   plus the lifetime units at today's spot price       2 x 45
//                 own only:
//                   open ground by walking distance (near / mid / far)     3
//                   melon: how many more pay, margin of the next one       2
//                   per type: lifetime units ONE unit planted / placed
//                   today would yield, and its net value at spot        8 + 8
//                   land coming free, by walking distance (near / mid /
//                   far): freed today by a forced harvest, freed today if
//                   an optional harvest is taken, a weed by morning, and
//                   forced to come free tomorrow                       4 x 3
//               All of it is public state, so it sits in the policy half.
//
// The critic tail from SC_CRIT_0 on is everything the value function may see
// and the policy may not. It enters the merged model ONLY through the value
// head's own input layer, after the shared body, so no policy head can read it:
//
//  CRIT     29  opponent net worth and shed, seeds, a few board counts
//  PROJ    190  THE RUN-OFF PROJECTION (projection.hpp): both players' money
//               on the last night and on each of the next ten mornings, the
//               market prices those mornings, per-product proceeds on ONE
//               shared, time-ordered market (the earlier seller tanks it for
//               the later one), and what the opponent's timing costs each
//               side. Both players run the SAME canonical fertilize
//               schedule; nothing hidden is read.
// ---------------------------------------------------------------------------
constexpr int SC_MONEY    = 0;                     // 11
constexpr int SC_CREW     = SC_MONEY    + 11;      // 11 .. 19
constexpr int SC_SHED     = SC_CREW     + 9;       // 20 .. 40
constexpr int SC_CLOCK    = SC_SHED     + 21;      // 41 .. 49
constexpr int SC_MARKET   = SC_CLOCK    + 9;       // 50 .. 85
constexpr int SC_TOWN     = SC_MARKET   + 36;      // 86 .. 105
constexpr int SC_COSTS    = SC_TOWN     + 20;      // 106 .. 113
constexpr int SC_NETWORTH = SC_COSTS    + 8;       // 114 .. 116
constexpr int SC_FARM     = SC_NETWORTH + 3;       // 117 .. 147
constexpr int SC_OPP_FARM = SC_FARM     + 31;      // 148 .. 160

constexpr int SCALAR_DIM_CELL   = SC_OPP_FARM + 13;      // 161, the cell trunk
constexpr int SC_FCAST          = SCALAR_DIM_CELL;       // 161
constexpr int FCAST_DIM         = 123;                   // == fcast::FCAST_FEATURES
constexpr int SCALAR_DIM_POLICY = SC_FCAST + FCAST_DIM;  // 284, the global MLP
constexpr int SC_CRIT_0         = SCALAR_DIM_POLICY;     // 284
constexpr int CRIT_DIM_BASIC    = 29;
constexpr int SC_PROJ           = SC_CRIT_0 + CRIT_DIM_BASIC;   // 313
constexpr int PROJ_DIM          = 190;                   // == proj::PROJ_FEATURES
constexpr int SCALAR_DIM_VALUE  = SC_PROJ + PROJ_DIM;    // 503
constexpr int CRIT_DIM          = SCALAR_DIM_VALUE - SC_CRIT_0; // 219, value head only

static_assert(SCALAR_DIM_VALUE <= N_SCALAR_PLANES * CELLS,
              "scalar features no longer fit in the scalar planes");
// ===========================================================================
// Net dimensions
// ===========================================================================
// TWO MODELS, NOT THREE.
//
//   trunk_cell     the conv ResNet over the boards. The per-cell heads
//                  (harvest, fertilize, feed) read it cell by cell. Unchanged.
//   global_critic  conv-less MLPs. The day's global choices (land,
//                  sell, type slots, counts) come off the POLICY body; the
//                  value comes off its OWN body of the same shape. The board
//                  reaches both through the FCAST block (forecast.hpp).
//
// THE CRITIC DOES NOT SHARE. The value body reads the same policy-half
// scalars as the policy body, but has its own weights, so the value loss never
// touches a weight a policy head reads. The critic-only features (opponent
// shed and seeds, the run-off projection) go through the value head's own
// encoder and are joined after the value body.
//
// NAMING IS LOAD-BEARING: main.cpp puts every parameter whose name contains
// "value" in the critic's optimizer group (lr 2e-4) and the rest in the
// policy group. Value-only layers are named value_*; the shared body is not.
constexpr int MLP_HID = 320;
constexpr int MLP_OUT = 100;

// Own Board (59) + Shed Mask (1) + Opponent Board (28) = 88 channels
constexpr int SPATIAL_IN = OWN_TILE_CH + 1 + OPP_TILE_CH;             // 88
constexpr int GLOBAL_DIM = MLP_OUT;

// ---- the cell trunk (conv) ------------------------------------------------
constexpr int N_STEM_RES = 4;
constexpr int BOARD_CH   = 140;  // d_model

// ---- the merged global / critic MLP ----------------------------------------
constexpr int GLOBAL_HID      = 320;   // body width (policy and value bodies alike)
constexpr int N_GLOBAL_BLOCKS = 4;     // pre-LN residual MLP blocks
constexpr int VALUE_CRIT_HID  = 256;   // the critic tail's private encoder
constexpr int VALUE_HID       = 256;

// ---------------------------------------------------------------------------
// Torch pieces
// ---------------------------------------------------------------------------
struct drone_action { int chosen_index; };
struct policy_move  { int index; double value; int id; };
struct value_policy { torch::Tensor policy; torch::Tensor value; };

namespace F = torch::nn::functional;

// One engine PER THREAD. multinomial() draws from this inside the
// `#pragma omp parallel for` that runs macro_plan_day, and a shared
// std::mt19937 there is a data race (corrupted state, correlated draws).
static thread_local std::mt19937 gen{ std::random_device{}() };

struct CustomTensor {
public:
    float* data;
    int shape[5];
    int strides[5];
    int owns_data;

    inline CustomTensor to_4d(int d) {
        CustomTensor t(1, shape[1], shape[2], shape[3], shape[4], 0);
        t.data = this->data + (strides[0] * d);
        return t;
    }
    inline CustomTensor to_3d(int d) {
        CustomTensor t(1, 1, shape[2], shape[3], shape[4], 0);
        t.data = this->data + (strides[1] * d);
        return t;
    }
    inline CustomTensor to_1d(int d) {
        CustomTensor t(1, 1, 1, 1, shape[4], 0);
        t.data = this->data + (strides[3] * d);
        return t;
    }
    void set_zero() {
        const long n = (long)shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
        std::memset(data, 0, (size_t)n * sizeof(float));
    }

    CustomTensor() : data(nullptr), owns_data(0) {
        for (int i = 0; i < 5; ++i) { shape[i] = 0; strides[i] = 0; }
    }
    CustomTensor(const CustomTensor& o) : data(o.data), owns_data(0) {
        for (int i = 0; i < 5; ++i) { shape[i] = o.shape[i]; strides[i] = o.strides[i]; }
    }
    CustomTensor(int d0, int d1, int d2, int d3, int d4, int alloc) {
        shape[0] = d0; shape[1] = d1; shape[2] = d2; shape[3] = d3; shape[4] = d4;
        strides[4] = 1;
        strides[3] = d4;
        strides[2] = d3 * d4;
        strides[1] = d2 * d3 * d4;
        strides[0] = d1 * d2 * d3 * d4;
        owns_data = 0;
        if (alloc) { data = new float[(size_t)d0 * d1 * d2 * d3 * d4]; owns_data = 1; }
    }
    ~CustomTensor() { if (owns_data) delete[] data; }

    inline float& operator()(int l) { return data[l]; }
    inline float& operator()(int k, int l) { return data[k * strides[3] + l]; }
    inline float& operator()(int j, int k, int l) {
        return data[j * strides[2] + k * strides[3] + l];
    }
    inline CustomTensor& operator=(const CustomTensor& o) {
        if (this == &o) return *this;
        if (owns_data && data) delete[] data;
        for (int i = 0; i < 5; ++i) { shape[i] = o.shape[i]; strides[i] = o.strides[i]; }
        data = o.data;
        owns_data = 0;
        return *this;
    }
};

inline int choose_num_groups(int num_channels) {
    for (int g = std::min(8, num_channels); g >= 1; --g)
        if (num_channels % g == 0) return g;
    return 1;
}

struct ResBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::GroupNorm gn1{nullptr}, gn2{nullptr};

    ResBlock(int channels) {
        const int groups = choose_num_groups(channels);
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        gn1 = register_module("gn1", torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, channels)));
        gn2 = register_module("gn2", torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, channels)));
        torch::nn::init::constant_(gn2->weight, 0.0);
        torch::nn::init::constant_(gn2->bias, 0.0);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = conv1(x); x = gn1(x); x = torch::leaky_relu_(x, 0.01);
        x = conv2(x); x = gn2(x);
        x.add_(residual);
        torch::leaky_relu_(x, 0.01);
        return x;
    }
};

// ---------------------------------------------------------------------------
//  Trunk: the conv ResNet the per-cell heads read. Only trunk_cell uses it.
// ---------------------------------------------------------------------------
struct TrunkImpl : torch::nn::Module {
    int scalar_dim_, d_model_;

    torch::nn::Linear mlp_fc1{nullptr}, mlp_fc2{nullptr};

    // ---- stem ----
    torch::nn::Conv2d start_conv{nullptr};
    std::vector<torch::nn::ModuleHolder<ResBlock>> res;

    TrunkImpl(int scalar_dim, int d_model, int n_res, int mlp_hid = MLP_HID)
        : scalar_dim_(scalar_dim), d_model_(d_model) {

        mlp_fc1 = register_module("mlp_fc1", torch::nn::Linear(scalar_dim_, mlp_hid));
        mlp_fc2 = register_module("mlp_fc2", torch::nn::Linear(mlp_hid, MLP_OUT));

        // The 1x1 conv reads the combined 88-channel spatial stack + global vector
        start_conv = register_module("start_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(SPATIAL_IN + GLOBAL_DIM, d_model_, 1)));

        res.reserve(n_res);
        for (int i = 0; i < n_res; ++i)
            res.emplace_back(register_module("res" + std::to_string(i + 1),
                                             std::make_shared<ResBlock>(d_model_)));
    }

    // [B, d_model, 10, 10]
    torch::Tensor forward(torch::Tensor x) {
        const int B = (int)x.size(0);

        auto s = x.slice(1, CH_SCALAR_0, CH_SCALAR_0 + N_SCALAR_PLANES)
                  .reshape({B, N_SCALAR_PLANES * CELLS}).slice(1, 0, scalar_dim_);
        s = mlp_fc1(s); torch::leaky_relu_(s, 0.01);
        auto g = mlp_fc2(s); torch::leaky_relu_(g, 0.01);

        auto own_board = x.slice(1, CH_OWN_0, CH_OWN_0 + OWN_TILE_CH);
        auto shed_mask = x.slice(1, CH_SHED_MASK, CH_SHED_MASK + 1);
        auto opp_board = x.slice(1, CH_OPP_0, CH_OPP_0 + OPP_TILE_CH);
        auto spatial_features = torch::cat({own_board, shed_mask, opp_board}, 1);

        torch::Tensor h;
        {
            auto w = start_conv->weight;
            auto w_spatial = w.slice(1, 0, SPATIAL_IN);
            auto w_g       = w.slice(1, SPATIAL_IN, SPATIAL_IN + GLOBAL_DIM)
                            .reshape({d_model_, GLOBAL_DIM});
            h = torch::conv2d(spatial_features, w_spatial, start_conv->bias);
            h.add_(torch::matmul(g, w_g.t()).view({B, d_model_, 1, 1}));
            torch::leaky_relu_(h, 0.01);
        }
        for (auto& blk : res) h = blk->forward(h);
        return h;
    }
};
TORCH_MODULE(Trunk);

// ---------------------------------------------------------------------------
//  Pre-LN residual MLP block:  x + W2(leaky(W1(LN(x))))
// ---------------------------------------------------------------------------
// W2 starts at zero, so each block starts as the identity and the body is a
// plain two-layer MLP on day one -- the same trick ResBlock plays with gn2.
struct ResMlpBlockImpl : torch::nn::Module {
    torch::nn::LayerNorm ln{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    explicit ResMlpBlockImpl(int dim) {
        ln  = register_module("ln",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
        fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
        torch::nn::init::constant_(fc2->weight, 0.0);
        torch::nn::init::constant_(fc2->bias, 0.0);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto h = fc1(ln(x));
        torch::leaky_relu_(h, 0.01);
        return x + fc2(h);
    }
};
TORCH_MODULE(ResMlpBlock);

// ---------------------------------------------------------------------------
//  GlobalCritic: the global-heads body and the value, as two separate MLPs.
// ---------------------------------------------------------------------------
//   body   = ResMLP( scalars[0 : SCALAR_DIM_POLICY] )          -> every global head
//   value  = MLP( [ value_body( same scalars, own weights ),
//                   value_crit_enc( scalars[SC_CRIT_0 : SCALAR_DIM_VALUE] ) ] )
struct GlobalCriticOut {
    torch::Tensor body;    // [B, GLOBAL_HID], undefined if not asked for
    torch::Tensor value;   // [B, 1]
};

struct GlobalCriticImpl : torch::nn::Module {
    // ---- the POLICY body: every global head reads it ----
    torch::nn::Linear in_fc{nullptr};
    std::vector<ResMlpBlock> blocks;
    torch::nn::LayerNorm out_ln{nullptr};

    // ---- the VALUE body: same shape, own weights ----
    // (names carry "value": the critic's optimizer group)
    torch::nn::Linear value_in_fc{nullptr};
    std::vector<ResMlpBlock> value_blocks;
    torch::nn::LayerNorm value_out_ln{nullptr};

    // ---- the value head ----
    torch::nn::LayerNorm value_crit_ln{nullptr};
    torch::nn::Linear    value_crit_fc{nullptr};
    torch::nn::Linear    value_fc1{nullptr}, value_fc2{nullptr}, value_fc3{nullptr};

    GlobalCriticImpl() {
        in_fc = register_module("in_fc", torch::nn::Linear(SCALAR_DIM_POLICY, GLOBAL_HID));
        for (int i = 0; i < N_GLOBAL_BLOCKS; ++i)
            blocks.push_back(register_module("block" + std::to_string(i + 1),
                                             ResMlpBlock(GLOBAL_HID)));
        out_ln = register_module("out_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({GLOBAL_HID})));

        value_in_fc = register_module("value_in_fc",
            torch::nn::Linear(SCALAR_DIM_POLICY, GLOBAL_HID));
        for (int i = 0; i < N_GLOBAL_BLOCKS - 1; ++i)
            value_blocks.push_back(register_module("value_block" + std::to_string(i + 1),
                                                   ResMlpBlock(GLOBAL_HID)));
        value_out_ln = register_module("value_out_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({GLOBAL_HID})));

        value_crit_ln = register_module("value_crit_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({CRIT_DIM})));
        value_crit_fc = register_module("value_crit_fc",
            torch::nn::Linear(CRIT_DIM, VALUE_CRIT_HID));
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(GLOBAL_HID + VALUE_CRIT_HID, VALUE_HID));
        value_fc2 = register_module("value_fc2", torch::nn::Linear(VALUE_HID, VALUE_HID));
        value_fc3 = register_module("value_fc3", torch::nn::Linear(VALUE_HID, 1));
    }

    static torch::Tensor run_body(torch::Tensor s, torch::nn::Linear& fc,
                                  std::vector<ResMlpBlock>& blks, torch::nn::LayerNorm& ln) {
        auto h = fc(s);
        torch::leaky_relu_(h, 0.01);
        for (auto& blk : blks) h = blk->forward(h);
        h = ln(h);
        torch::leaky_relu_(h, 0.01);
        return h;                                             // [B, GLOBAL_HID]
    }

    // x: the full observation [B, NUM_INPUT_CH, 10, 10]. `want_policy` false
    // skips the policy body (the critic-only call).
    GlobalCriticOut forward(torch::Tensor x, bool want_policy = true) {
        const int B = (int)x.size(0);
        auto s = x.slice(1, CH_SCALAR_0, CH_SCALAR_0 + N_SCALAR_PLANES)
                  .reshape({B, N_SCALAR_PLANES * CELLS});
        auto s_pol  = s.slice(1, 0, SCALAR_DIM_POLICY);
        auto s_crit = s.slice(1, SC_CRIT_0, SCALAR_DIM_VALUE);

        GlobalCriticOut o;
        if (want_policy) o.body = run_body(s_pol, in_fc, blocks, out_ln);

        auto hv = run_body(s_pol, value_in_fc, value_blocks, value_out_ln);
        auto c = value_crit_fc(value_crit_ln(s_crit));
        torch::leaky_relu_(c, 0.01);

        auto v = value_fc1(torch::cat({hv, c}, 1));
        torch::leaky_relu_(v, 0.01);
        v = value_fc2(v);
        torch::leaky_relu_(v, 0.01);
        o.value = value_fc3(v);                               // [B, 1]
        return o;
    }
};
TORCH_MODULE(GlobalCritic);

// ---------------------------------------------------------------------------
//  NetPolicy
// ---------------------------------------------------------------------------
struct NetPolicyImpl : torch::nn::Module {
    Trunk        trunk_cell{nullptr};
    GlobalCritic global_critic{nullptr};

    torch::nn::Conv2d spatial{nullptr};
    torch::nn::Linear glob_land{nullptr};
    torch::nn::ModuleList sell_heads;
    torch::nn::ModuleList count_heads;
    torch::nn::ModuleList type_heads;     // one per type slot, MACRO_TYPE_BINS wide

    NetPolicyImpl() {
        trunk_cell    = register_module("trunk_cell",
            Trunk(SCALAR_DIM_CELL, BOARD_CH, N_STEM_RES));
        global_critic = register_module("global_critic", GlobalCritic());

        spatial = register_module("spatial",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(BOARD_CH, MACRO_SPATIAL_CH, 1)));

        glob_land = register_module("glob_land", torch::nn::Linear(GLOBAL_HID, 2));
        sell_heads = register_module("sell_heads", torch::nn::ModuleList());
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            sell_heads->push_back(torch::nn::Linear(GLOBAL_HID, MACRO_SELL_BINS));
        count_heads = register_module("count_heads", torch::nn::ModuleList());
        for (int k = 0; k < NUM_COUNT_TYPES; ++k)
            count_heads->push_back(torch::nn::Linear(GLOBAL_HID, macro_count_bins(k)));
        type_heads = register_module("type_heads", torch::nn::ModuleList());
        for (int s = 0; s < MACRO_TYPE_SLOTS; ++s)
            type_heads->push_back(torch::nn::Linear(GLOBAL_HID, MACRO_TYPE_BINS));
    }

    // The critic: .value [B,1]. .policy is left undefined. Cheap: no conv.
    value_policy forward_values(torch::Tensor x) {
        if (x.dim() == 3) x = x.unsqueeze(0);
        value_policy vp;
        vp.value = global_critic->forward(x, /*want_policy=*/false).value;
        return vp;
    }

    torch::Tensor forward_value(torch::Tensor x) { return forward_values(x).value; }

    // [B, MACRO_DIM] policy, [B,1] value.
    value_policy forward(torch::Tensor x) {
        if (x.dim() == 3) x = x.unsqueeze(0);
        const int B = (int)x.size(0);

        auto gc = global_critic->forward(x);
        value_policy vp;
        vp.value = gc.value;

        auto hc = trunk_cell->forward(x);
        auto sp = spatial(hc).reshape({B, MACRO_SPATIAL_CH * CELLS});

        const auto& gb = gc.body;
        std::vector<torch::Tensor> parts;
        parts.reserve(2 + NUM_PRODUCTS + NUM_COUNT_TYPES + MACRO_TYPE_SLOTS);
        parts.push_back(sp);
        parts.push_back(glob_land(gb));
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            parts.push_back(sell_heads[k]->as<torch::nn::Linear>()->forward(gb));
        for (int k = 0; k < NUM_COUNT_TYPES; ++k)
            parts.push_back(count_heads[k]->as<torch::nn::Linear>()->forward(gb));
        for (int s = 0; s < MACRO_TYPE_SLOTS; ++s)
            parts.push_back(type_heads[s]->as<torch::nn::Linear>()->forward(gb));

        vp.policy = torch::cat(parts, 1);                        // [B, MACRO_DIM]
        return vp;
    }

    value_policy forward_chunked(torch::Tensor x, int num_chunks = 3) {
        if (x.dim() == 3) x = x.unsqueeze(0);
        const int B  = (int)x.size(0);
        const int cs = (B + num_chunks - 1) / num_chunks;
        std::vector<torch::Tensor> pol, val;
        for (int c = 0; c < num_chunks; ++c) {
            const int s = c * cs, e = std::min(s + cs, B);
            if (s >= e) break;
            auto r = forward(x.slice(0, s, e));
            pol.push_back(r.policy);
            val.push_back(r.value);
        }
        value_policy vp;
        vp.policy = torch::cat(pol, 0);
        vp.value  = torch::cat(val, 0);
        return vp;
    }
};
TORCH_MODULE(NetPolicy);

using namespace torch::indexing;

// ---------------------------------------------------------------------------
// policy.cpp
// ---------------------------------------------------------------------------
struct MacroDay;   // macro_planner.hpp

// One observation per player. No pending-intent projection any more: the
// decision is taken at hour 0, when nothing is in flight.
void cell_to_input(CustomTensor input1, CustomTensor input2, simulation& game);

// THE BRIDGE. Reads MACRO_DIM floats of raw logits, resolves every head in the
// order the spec fixes, and fills `d` with a grid that is legal, paid for and
// routable. Pass nullptr for the three recording lists on the greedy path.
void macro_plan_day(const float* out, simulation& game, int player, MacroDay& d,
                    std::vector<std::vector<drone_action>>* actions,
                    std::vector<double>* log_probs,
                    std::vector<int>* head_ids,
                    float temperature, bool greedy, int print);

// THE LAST DAY, with no network behind it. Fills `d` with a fixed liquidation:
// harvest every unit that can be banked, collect every waiting fertilizer,
// carry it all to the shed and sell it, using the SMALLEST crew that gets the
// job done. Takes no logits, records no choice set, and is called instead of
// macro_plan_day on day NUM_DAYS-1.
void macro_liquidate_day(simulation& game, int player, MacroDay& d);

// Softmax / sampling helpers, shared with the filter.
void my_softmax(std::vector<policy_move>& logits,
                std::vector<policy_move>& exps, float temperature);
int  multinomial(std::vector<policy_move>& policyy);
int  get_max_index(std::vector<policy_move>& policyy);

// `entropy` comes back PER CHOICE SET, not pooled: the trainer buckets by head
// id and targets each head separately.
torch::Tensor policy_to_jointlog(
    torch::Tensor output,
    torch::Tensor actions_indices,
    torch::Tensor output_indices,
    torch::Tensor valid_action_indices,
    torch::Tensor chosen_move_indices,
    int num_groups_in,
    torch::Tensor& entropy,
    int batch_size);

// +1 / 0 / -1 on who ended the 30th day with more money.
inline float macro_terminal_reward(int result, int player, double my_money, double opp_money) {
    // 1. The original binary win/loss signal (+1 / 0 / -1)
    float base_reward = 0.0f;
    if (result == 0 || result == 1) {
        base_reward = (result == player) ? 1.0f : -1.0f;
    }

    return base_reward;
}

#endif  // GAME_HPP