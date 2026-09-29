#include "game.hpp"
#include "macro_planner.hpp"
#include "projection.hpp"
#include "forecast.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>

// ===========================================================================
// OBSERVATION
// ===========================================================================
// One call fills BOTH players' observations: input1 is player 0's view,
// input2 is player 1's. There is no pending-intent projection -- the decision
// is taken at hour 0, when nothing is in flight and every hand is standing at
// the shed.
//
// WHAT THE FEATURE ENGINEERING IS FOR. The network makes ONE decision per day
// and gets one bit of reward 30 days later, so there is no budget for it to
// learn arithmetic the environment already performs. Two rules held
// throughout:
//
//   1. NOTHING NEW IS REVEALED. Every derived feature is a pure function of
//      state the player already sees, except the block after SC_CRIT_0, which
//      is the critic's and is sliced off before the policy trunks read it.
//   2. THE MASKS ARE SHARED, NOT REIMPLEMENTED. harvest_ready, must_harvest,
//      must_water, must_feed, fertilize_pays,
//      must_care_after_feed and the rest live in game.hpp and the filter gates
//      on the same definitions.
//
// WHAT CHANGED WITH THE FERTILIZE HEAD (no channel moved, two changed meaning)
//
//   * CHANNEL 21 is the number of applications the CANONICAL schedule
//     (fert_day_for) still has room for, over 2. A hint, not a commitment.
//   * CHANNEL 42 is still MUST WATER, over the same three reasons: life
//     support, the non-ongoing yield water, and banking a live fertilizer.
//   * CHANNEL 56 is FERTILIZE PAYS -- the fertilize head's mask on the hour-0
//     tile.
//   * CHANNEL 58 is FERTILIZER READ TONIGHT: a live cover lands on tonight's
//     payout, so its bank water is already forced.
//
// WHAT CHANGED WITH THE DEAD-PLANT REWORK (no channel moved)
//
//   * CHANNEL 42 now goes quiet on a DOOMED plant: must_water is gated on
//     plant_has_future, so an ongoing crop the morning after its last
//     production night is no longer reported as needing an hour. Read with 32
//     (decaying) and 37 (dies tonight), a cell that dies tonight and is NOT
//     must-watered is exactly the cell the primary pass may replant today.
//   * CHANNEL 44 (pending ops) stops counting a water on such a cell for the
//     same reason. Still a pure function of the raw 0..27, so it is safe over
//     the opponent's board.
//
// CHANNELS 27, 42 AND 43 are the forced flags the primary, fertilize and feed
// heads read to see how much of a cell's day is already spoken for. Channel 56
// is the fertilize head's mask. Channel 57 is must_care_after_feed: no head is
// masked on it, but it is exactly the condition under which a feed drags a
// forced care with it.
//
// THE PROJECTION (critic only, SC_PROJ .. SCALAR_DIM_VALUE). projection.hpp
// plays both boards forward under a fixed canonical policy and replays the
// resulting sales on ONE shared market in time order, so the player whose
// harvest comes in first sells first and the other sells into the dip. Both
// players selling on the same morning is reproduced unit by unit, the way
// process_market pairs their orders. Both players run the same canonical
// fertilize schedule; nothing hidden is read. 190 features: terminal money, per-product
// proceeds and realised prices, the cost of the opponent's timing, and each
// of the next ten mornings' cash and prices.
//
// PER-TYPE COUNTS. The count heads choose "how many of each crop / animal",
// so both sides' current per-type cell counts are policy inputs: the own ones
// in FARM, the opponent's in OPP_FARM.
//
// THE FORECAST (policy, SC_FCAST .. SCALAR_DIM_POLICY). The global heads and
// the critic are one conv-less MLP now, so the board reaches them through
// forecast.hpp: per product, what each side can harvest today, by tomorrow,
// by day+3 and over the lifetime; what one more unit of each type would still
// yield; the open ground by distance; and the melon market.
// ===========================================================================

namespace {

// fib(0)=1, fib(1)=1, fib(2)=2, ... as in do_hire.
int obs_fib(int n) {
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { const int t = a + b; a = b; b = t; }
    return a;
}

// ---------------------------------------------------------------------------
// THE SCHEDULE'S REMAINING APPLICATIONS  --  channel 21
// ---------------------------------------------------------------------------
// How many fertilizer applications the CANONICAL schedule still has room for,
// judged from age and current cover. It says what the crop could still use;
// the fertilize head decides what it actually gets, day by day, under the
// fertilize_pays mask (channel 56).
//
// THE BANDS MIRROR fert_day_for EXACTLY and must be kept in step with it. An
// age band that outran the schedule would tell the policy a strawberry at age
// 14 still owes an application when the last one was due at 13, which is a
// straight lie about the cell.
//
//   wheat, carrot   one application
//   tomato          applications at ages 7 and 10
//   strawberry      applications at ages 9 and 13
//
// Melon is never fertilized, so it stays at 0.
//
// Normalised by 2, not 4: two is the most the canonical schedule ever names, so dividing by
// the number of GoalBools would waste half the channel's range.
float future_fertilization_need(const tile& t, int day) {
    float future_fert = 0.0f;
    if (t.type == T_PLANT && t.crop >= 0) {
        const int age = day - t.phase;
        int needed = 0;
        if (t.crop == WHEAT || t.crop == CARROT) {
            int target_age = (t.crop == WHEAT) ? 2 : 1;
            if (age <= CROPS[t.crop].max_yield_day) needed = (t.fertilized >= t.phase + target_age) ? 0 : 1;
        } else if (t.crop == TOMATO) {
            if (age <= 7)       needed = 2 - ((t.fertilized >= t.phase + 7) ? 1 : 0);
            else if (age <= 10) needed = 1 - ((t.fertilized >= t.phase + 10) ? 1 : 0);
        } else if (t.crop == STRAWBERRY) {
            if (age <= 9)       needed = 2 - ((t.fertilized >= t.phase + 9) ? 1 : 0);
            else if (age <= 13) needed = 1 - ((t.fertilized >= t.phase + 13) ? 1 : 0);
        }
        future_fert = (float)needed;
    }
    return future_fert;
}

// ---------------------------------------------------------------------------
// Farm census
// ---------------------------------------------------------------------------
struct FarmAgg {
    float owned = 0, plants = 0, weeds = 0, empty_owned = 0;
    float coops = 0, pastures = 0, bare_structs = 0;
    float per_crop[NUM_CROPS] = { 0 };
    float animals[NUM_ANIMALS] = { 0 };
    float ripe = 0, yield_units = 0, yield_value = 0;
    float need_water = 0, dying = 0, need_feed = 0, escaping = 0;
    // must_ops is the FORCED OP COUNT -- the work the forced pass hires for
    // before any head's work. It includes the care that rides on each forced feed.
    // fert_cells counts plants where a fertilizer would pay today.
    // care_cells counts animals where a feed WOULD buy a forced care, which is
    // the extra hour the feed head is implicitly choosing.
    float must_ops = 0, collectable = 0, produces_tonight = 0;
    float fert_cells = 0, care_cells = 0;
    float ops = 0;

    float animals_total() const {
        float n = 0;
        for (int a = 0; a < NUM_ANIMALS; ++a) n += animals[a];
        return n;
    }
};

FarmAgg farm_agg(const player_state& ps, const market_state& mk, int day, int turn) {
    FarmAgg a;
    for (int y = 0; y < BOARD_SIZE; ++y)
    for (int x = 0; x < BOARD_SIZE; ++x) {
        const tile& t = ps.board[y][x];
        if (!t.bought) continue;
        a.owned += 1.0f;

        switch (t.type) {
            case T_EMPTY:   a.empty_owned += 1.0f; break;
            case T_WEED:    a.weeds       += 1.0f; break;
            case T_COOP:    a.coops       += 1.0f; break;
            case T_PASTURE: a.pastures    += 1.0f; break;
            default: break;
        }
        if (structure_place_ready(t)) a.bare_structs += 1.0f;

        if (t.type == T_PLANT && t.crop >= 0) {
            a.plants += 1.0f;
            a.per_crop[t.crop] += 1.0f;
            // Only cells a water would actually buy something on: a doomed
            // plant is a weed by morning and is counted nowhere.
            if (!t.watered_today &&
                (plant_has_future(t, day) || water_yields_now(t, day)))
                a.need_water += 1.0f;
            if (plant_dies_tonight(t, day))  a.dying       += 1.0f;
            if (plant_produces_tonight(t, day)) a.produces_tonight += 1.0f;
            // Where the fertilize head is offered today.
            if (fertilize_pays(t, day)) a.fert_cells += 1.0f;
        }
        if (t.animal >= 0) {
            a.animals[t.animal] += 1.0f;
            if (!t.fed_today)                    a.need_feed   += 1.0f;
            if (animal_escapes_tonight(t, day))  a.escaping    += 1.0f;
            if (t.fertilizer_available)          a.collectable += 1.0f;
            if (animal_produces_tonight(t, day)) a.produces_tonight += 1.0f;
            if (must_care_after_feed(t, day))    a.care_cells  += 1.0f;
        }
        if (harvest_ready(t, day)) a.ripe += 1.0f;
        a.must_ops += (float)cell_musts(t, day, turn).ops();

        a.yield_units += (float)t.yield_units;
        const int prod = tile_product(t);
        if (prod >= 0 && t.yield_units > 0)
            a.yield_value += (float)t.yield_units * (float)mk.prices[prod];

        a.ops += (float)tile_pending_ops(t, day);
    }
    return a;
}

// What the sell head would ACTUALLY realise on the whole shed, walking the
// price the way process_market does rather than pretending n * spot.
double shed_walk_value(const player_state& ps, const market_state& mk) {
    double v = 0.0;
    for (int k = 0; k < NUM_PRODUCTS; ++k)
        if (ps.shed[k] > 0) v += sell_proceeds(k, ps.shed[k], mk.inventory[k]);
    return v;
}

}  // namespace

void cell_to_input(CustomTensor input1, CustomTensor input2, simulation& game)
{
    CustomTensor inputs[2] = { input1, input2 };
    const int day = game.day();

    // ---- board planes ---------------------------------------------------
    // `full` is what splits own from opponent: the derived channels 28..58 are
    // only written for the own board.
    auto encode = [&](CustomTensor& inp, const player_state& ps, int c0, bool full) {
        for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const tile& t = ps.board[y][x];

            // ---- raw, 0..27 ----------------------------------------------
            inp(c0 + t.type, y, x) = 1.0f;
            inp(c0 + 5,  y, x) = (float)t.bought;
            inp(c0 + 6,  y, x) = t.has_farmer / 4.0f;
            if (t.crop   >= 0) inp(c0 + 7  + t.crop,   y, x) = 1.0f;
            if (t.animal >= 0) inp(c0 + 16 + t.animal, y, x) = 1.0f;
            if (t.type == T_PLANT || t.animal >= 0)
                inp(c0 + 12, y, x) = (day - t.phase) / 30.0f;
            inp(c0 + 13, y, x) = (float)t.consecutive_unwatered;
            if (t.fertilized >= day) {
                inp(c0 + 14, y, x) = std::clamp((t.fertilized - day + 1) / 3.0f, 0.0f, 1.0f);
            }

            // FIX: Give immortal/ongoing crops a distinct 1.0 value instead of
            // overlapping with 0.0
            if (t.type == T_PLANT || t.animal >= 0) {
                if (t.max_lifespan_step < 0) {
                    inp(c0 + 15, y, x) = 1.0f;
                } else {
                    inp(c0 + 15, y, x) = std::clamp((t.max_lifespan_step - game.turn) / 48.0f, -1.0f, 1.0f);
                }
            }
            inp(c0 + 19, y, x) = (float)t.fed_today;
            inp(c0 + 20, y, x) = (float)t.cared_today;
            // 21: NOT watered_today any more. See future_fertilization_need.
            // Over 2, the most the schedule ever names, so the channel uses its range.
            inp(c0 + 21, y, x) = future_fertilization_need(t, day) / 2.0f;
            inp(c0 + 22, y, x) = (float)t.fertilizer_available;
            inp(c0 + 23, y, x) = (float)t.consecutive_unfed;
            inp(c0 + 24, y, x) = t.pending_care_bonus / 5.0f;
            inp(c0 + 25, y, x) = t.yield_units / 6.0f;
            inp(c0 + 26, y, x) = harvest_ready(t, day) ? 1.0f : 0.0f;
            // The FORCED HARVEST flag: ripe and lost by morning, so the
            // harvest head is masked OFF this cell and the op happens anyway.
            inp(c0 + 27, y, x) = must_harvest(t, day, game.turn) ? 1.0f : 0.0f;
            if (!full) continue;

            // ---- derived: state of the crop / animal ---------------------
            const int maxy  = tile_max_yield(t);
            const int fyd   = tile_first_yield_day(t);
            const int intvl = tile_interval(t);
            const int prod  = tile_product(t);

            if (t.type == T_PLANT && t.crop >= 0)
                inp(c0 + 28, y, x) = CROPS[t.crop].ongoing ? 1.0f : 0.0f;
            if (maxy > 0) {
                inp(c0 + 29, y, x) = std::clamp((float)t.yield_units / maxy, 0.0f, 1.0f);
                inp(c0 + 30, y, x) =
                    std::clamp((float)(maxy - t.yield_units) / maxy, 0.0f, 1.0f);
            }
            if (fyd >= 0) {
                const int wait = fyd - (day - t.phase);
                inp(c0 + 31, y, x) = std::clamp(wait / 10.0f, 0.0f, 1.0f);
            }
            if (t.max_lifespan_step >= 0 && game.turn >= t.max_lifespan_step)
                inp(c0 + 32, y, x) = 1.0f;              // decay_plants is eating it
            if (prod >= 0) {
                float prod_norm[] = {250,250,400,1000,1500,200,1500,1500,250};

                // FIX: Divisor changed to 2000.0f to accommodate late-game spikes
                inp(c0 + 33, y, x) = std::clamp(
                    t.yield_units * (float)game.market.prices[prod] / prod_norm[prod], 0.0f, 2.0f);
                inp(c0 + 34, y, x) =
                    game.market.prices[prod] / (float)MARKET[prod].base;
            }

            // ---- derived: what tonight and this hour do ------------------
            inp(c0 + 35, y, x) = (plant_produces_tonight(t, day) ||
                                  animal_produces_tonight(t, day)) ? 1.0f : 0.0f;
            inp(c0 + 36, y, x) = water_yields_now(t, day)         ? 1.0f : 0.0f;
            inp(c0 + 37, y, x) = plant_dies_tonight(t, day)       ? 1.0f : 0.0f;
            inp(c0 + 38, y, x) = animal_escapes_tonight(t, day)   ? 1.0f : 0.0f;
            // 39 is the BANK, not the cash: cared and fed tonight.
            inp(c0 + 39, y, x) = care_banks_tonight(t, day)       ? 1.0f : 0.0f;

            // ---- derived: what can be done here --------------------------
            inp(c0 + 40, y, x) = structure_place_ready(t) ? 1.0f : 0.0f;
            inp(c0 + 41, y, x) = tile_diggable(t)         ? 1.0f : 0.0f;
            // 42 / 43 are MUST WATER and MUST FEED. No head is masked on 42 --
            // there is no water head -- but it is how much of this cell's day
            // is already committed, which is what the primary head needs. A
            // DOOMED plant reads 0 here and 1 at 37: that pair is the cell the
            // primary pass is allowed to dig and replant today.
            {
                const CellMust cm = cell_musts(t, day, game.turn);
                inp(c0 + 42, y, x) = cm.water ? 1.0f : 0.0f;
                inp(c0 + 43, y, x) = cm.feed  ? 1.0f : 0.0f;
            }
            inp(c0 + 44, y, x) = std::clamp(tile_pending_ops(t, day) / 4.0f, 0.0f, 1.0f);

            // ---- derived: economics and geometry -------------------------
            if (t.type == T_PLANT && t.crop >= 0)
                inp(c0 + 45, y, x) = CROPS[t.crop].seed_cost / 100.0f;
            inp(c0 + 46, y, x) = shed_distance(x, y) / (float)MAX_SHED_DIST;
            inp(c0 + 47, y, x) = (t.bought && shed_distance(x, y) < MAX_SHED_DIST)
                               ? 1.0f : 0.0f;
            {
                const int q = quadrant_of(x, y);
                const int extra = (int)ps.unlocked_quadrants.size() - 1;
                if (!t.bought && extra < 3 && LAND_ORDER[extra] == q)
                    inp(c0 + 48, y, x) = 1.0f;          // one land buy from usable
                inp(c0 + 49 + q, y, x) = 1.0f;
            }
            if (maxy  > 0) inp(c0 + 53, y, x) = std::clamp(maxy / 6.0f, 0.0f, 1.0f);
            if (intvl > 0) inp(c0 + 54, y, x) = std::clamp(intvl / 3.0f, 0.0f, 1.0f);
            if (fyd  >= 0) inp(c0 + 55, y, x) = std::clamp(fyd / 12.0f, 0.0f, 1.0f);

            // ---- derived: fertilizer and care -----------------------------
            // 56: a fertilizer spent here today would add a unit -- the
            //     fertilize head's mask on a standing plant.
            // 57: a feed here would also buy a FORCED care.
            // 58: a live cover is read tonight, so its bank water is forced.
            inp(c0 + 56, y, x) = fertilize_pays(t, day)       ? 1.0f : 0.0f;
            inp(c0 + 57, y, x) = must_care_after_feed(t, day) ? 1.0f : 0.0f;
            inp(c0 + 58, y, x) = (t.type == T_PLANT && t.fertilized >= day &&
                                  fert_read_tonight(t, day)) ? 1.0f : 0.0f;
        }
    };

    // ---- work shared by both viewers, done once per game ----------------
    // Each of these is a pure function of the game state, not of who is
    // looking; only which half is "me" differs. Computing them per viewer did
    // everything twice.
    //
    //   * the board planes: a board's raw channels 0..27 are the same numbers
    //     whether it is encoded as "own" (CH_OWN_0) or "opponent" (CH_OPP_0),
    //     so each board is encoded once, in full, as its owner's own board,
    //     and those 28 planes are copied into the other viewer's opponent slot.
    //   * the run-off projection: `viewer` only picks which side
    //     write_features reads as "me"; the Result is otherwise identical.
    //   * the farm census, net worth, and the forecast's per-board blocks.
    for (int p = 0; p < NUM_PLAYERS; ++p)
        encode(inputs[p], game.players[p], CH_OWN_0, /*full=*/true);
    for (int p = 0; p < NUM_PLAYERS; ++p)
        std::memcpy(&inputs[1 - p](CH_OPP_0, 0, 0), &inputs[p](CH_OWN_0, 0, 0),
                    sizeof(float) * (size_t)OPP_TILE_CH * CELLS);

    proj::Result pr = proj::project(game.players, game.market,
                                    game.unlocked_shops, day, game.turn,
                                    /*viewer=*/0);
    const FarmAgg ag[NUM_PLAYERS] = {
        farm_agg(game.players[0], game.market, day, game.turn),
        farm_agg(game.players[1], game.market, day, game.turn),
    };
    const float net[NUM_PLAYERS] = { (float)game.net_worth(0), (float)game.net_worth(1) };
    fcast::Shared fsh;
    fcast::build_shared(game.players, game.market, day, game.turn, fsh);

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        CustomTensor& inp = inputs[p];
        const player_state& me = game.players[p];
        const player_state& op = game.players[1 - p];

        for (const ipos& t : shed_access_tiles())
            inp(CH_SHED_MASK, t.y, t.x) = 1.0f;

        const int S = CH_SCALAR_0;
        auto put = [&](int i, float v) {
            inp(S + i / 100, (i % 100) / 10, i % 10) = v;
        };
        int cur = 0;
        auto at = [&](int i) { cur = i; };
        auto W  = [&](float v) { put(cur++, v); };
        auto endat = [&](int i) {
#ifndef NDEBUG
            if (cur != i) {
                std::fprintf(stderr,
                    "cell_to_input: section ended at %d, layout says %d\n", cur, i);
                std::abort();
            }
#else
            (void)i;
#endif
        };

        // THE RUN-OFF PROJECTION. Both boards run the same canonical policy
        // off public state. Critic-only; written at SC_PROJ below. Computed
        // once above; this viewer only changes which side is "me".
        pr.viewer = p;

        const FarmAgg& ag_me = ag[p];
        const FarmAgg& ag_op = ag[1 - p];
        const int   hands   = (int)me.farmers.size();
        const float net_me  = net[p];
        const float net_op  = net[1 - p];

        // =================================================================
        // MONEY  (11)
        // =================================================================
        const float diff = (float)(op.money - me.money);
        endat(SC_MONEY);
        at(SC_MONEY);
        W(std::clamp((float)me.money /   1000.0f, -1.0f, 1.0f));
        W(std::clamp((float)me.money /  10000.0f, -1.0f, 1.0f));
        W((float)me.money / 100000.0f);
        W(std::clamp((float)op.money /   1000.0f, -1.0f, 1.0f));
        W(std::clamp((float)op.money /  10000.0f, -1.0f, 1.0f));
        W((float)op.money / 100000.0f);
        W(std::clamp(diff /   1000.0f, -1.0f, 1.0f));
        W(std::clamp(diff /  10000.0f, -1.0f, 1.0f));
        W(diff / 100000.0f);
        W(std::clamp((float)std::log1p(std::max(0.0, me.money)) / 12.0f, 0.0f, 1.0f));
        W(std::clamp((float)me.money / (1000.0f * std::max(1, hands)), 0.0f, 1.0f));

        // =================================================================
        // CREW / LAND  (9)
        // =================================================================
        endat(SC_CREW);
        at(SC_CREW);
        W(hands / 16.0f);
        W(op.farmers.size() / 16.0f);
        W((me.unlocked_quadrants.size() - 1) / 3.0f);
        W((op.unlocked_quadrants.size() - 1) / 3.0f);
        W(me.hires_today / 8.0f);
        {
            const int extra = (int)me.unlocked_quadrants.size() - 1;
            const float price = (extra < 3) ? (float)LAND_PRICES[extra] : 0.0f;
            W(price / 4000.0f);
            W((extra < 3 && me.money >= price) ? 1.0f : 0.0f);
            W(FARM_HAND_COST_MULT * obs_fib(me.hires_today) / 8.0f);
            int afford = 0;
            double spend = 0.0;
            while (hands + afford < std::min(MACRO_HIRE_HARD_MAX, MAX_UNITS)) {
                const double c = FARM_HAND_COST_MULT
                               * (double)obs_fib(me.hires_today + afford);
                if (spend + c > me.money) break;
                spend += c;
                ++afford;
            }
            W(afford / 16.0f);
        }

        // =================================================================
        // SHED  (21)
        // =================================================================
        endat(SC_SHED);
        at(SC_SHED);
        for (int k = 0; k < NUM_ITEMS; ++k) W(me.shed[k] / 50.0f);
        W(me.shed_total() / (float)SHED_CAPACITY);
        W((SHED_CAPACITY - me.shed_total()) / (float)SHED_CAPACITY);
        {
            const float v = (float)shed_walk_value(me, game.market);
            W(std::clamp(v /  10000.0f, 0.0f, 1.0f));
            W(v / 100000.0f);
        }
        for (int c = 0; c < NUM_CROPS; ++c) W(me.seeds[c] / 20.0f);

        // =================================================================
        // CLOCK  (9)
        // =================================================================
        endat(SC_CLOCK);
        at(SC_CLOCK);
        W(day / (float)NUM_DAYS);
        W(game.hour() / (float)TURNS_PER_DAY);
        W((NUM_DAYS - day) / (float)NUM_DAYS);
        W((NUM_DAYS - day <= 3) ? 1.0f : 0.0f);     // liquidate-now regime
        for (int c = 0; c < NUM_CROPS; ++c)
            W((day + CROPS[c].first_yield_day <= PLANT_HORIZON_DAY - 1) ? 1.0f : 0.0f);

        // =================================================================
        // MARKET  (36)
        // =================================================================
        endat(SC_MARKET);
        at(SC_MARKET);
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W(std::clamp((game.market.inventory[k] - 10000) / 500.0f, -1.0f, 1.0f));
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W(game.market.prices[k] / (float)MARKET[k].base);
        float prod_norm[] = {80,80,140,200,300,80,200,240,100};

        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W(game.market.prices[k] / prod_norm[k]);
        for (int k = 0; k < NUM_PRODUCTS; ++k) {
            const int stack = me.shed[k];
            if (stack <= 0 || game.market.prices[k] <= 0) { W(1.0f); continue; }
            const double got = sell_proceeds(k, stack, game.market.inventory[k]);
            W(std::clamp((float)(got / ((double)stack * game.market.prices[k])),
                         0.0f, 1.0f));
        }

        // =================================================================
        // TOWN  (20)
        // =================================================================
        endat(SC_TOWN);
        at(SC_TOWN);
        float drain[NUM_PRODUCTS] = { 0 };
        for (int s : game.unlocked_shops) {
            const std::vector<int>& prods = SHOP_PRODUCTS[s];
            const float mult = (prods.size() == 1) ? 2.0f : 1.0f;
            for (int item : prods) drain[item] += mult;
        }
        for (int k = 0; k < NUM_PRODUCTS; ++k) W(drain[k] / 12.0f);
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W((drain[k] + (k == FERTILIZER ? 0.0f : 1.0f)) / 12.0f);
        W(game.unlocked_shops.size() / (float)MAX_SHOP_INSTANCES);
        W(((TOWN_SHOP_UNLOCK_INTERVAL - (day + 1) % TOWN_SHOP_UNLOCK_INTERVAL)
           % TOWN_SHOP_UNLOCK_INTERVAL) / (float)TOWN_SHOP_UNLOCK_INTERVAL);

        // =================================================================
        // COSTS  (8)
        // =================================================================
        endat(SC_COSTS);
        at(SC_COSTS);
        for (int c = 0; c < NUM_CROPS; ++c)   W(CROPS[c].seed_cost / 100.0f);
        for (int a = 0; a < NUM_ANIMALS; ++a) W(ANIMALS[a].cost / 500.0f);

        // =================================================================
        // NET WORTH  (3)
        // =================================================================
        endat(SC_NETWORTH);
        at(SC_NETWORTH);
        W(std::clamp(net_me /   1000.0f, -1.0f, 1.0f));
        W(std::clamp(net_me /  10000.0f, -1.0f, 1.0f));
        W(net_me / 100000.0f);

        // =================================================================
        // FARM CENSUS  (31)  -- includes OWN per-crop and per-animal counts
        // =================================================================
        endat(SC_FARM);
        at(SC_FARM);
        W(ag_me.owned / 100.0f);
        W(ag_me.plants / 50.0f);
        for (int c = 0; c < NUM_CROPS; ++c) W(ag_me.per_crop[c] / 25.0f);
        W(ag_me.weeds / 50.0f);
        W(ag_me.empty_owned / 100.0f);
        W(ag_me.coops / 25.0f);
        W(ag_me.pastures / 25.0f);
        for (int a = 0; a < NUM_ANIMALS; ++a) W(ag_me.animals[a] / 10.0f);
        W(ag_me.bare_structs / 10.0f);
        W(ag_me.ripe / 50.0f);
        W(ag_me.yield_units / 100.0f);
        W(std::clamp(ag_me.yield_value / 10000.0f, 0.0f, 1.0f));
        W(ag_me.need_water / 50.0f);
        W(ag_me.dying / 20.0f);
        W(ag_me.need_feed / 10.0f);
        W(ag_me.escaping / 10.0f);
        // THE FORCED LOAD, forced cares included: the hands the forced pass
        // hires first, before any head's work.
        W(std::clamp(ag_me.must_ops / 50.0f, 0.0f, 1.0f));
        W(ag_me.collectable / 10.0f);
        // Plants where a fertilizer would pay today (the fertilize head's
        // candidate count).
        W(std::clamp(ag_me.fert_cells / 20.0f, 0.0f, 1.0f));
        W(std::clamp(ag_me.care_cells / 10.0f, 0.0f, 1.0f));
        W(ag_me.produces_tonight / 50.0f);
        W(std::clamp(ag_me.ops / 100.0f, 0.0f, 1.0f));
        W(std::clamp(ag_me.ops /
                     (float)(MACRO_HIRE_HARD_MAX * (TURNS_PER_DAY - 1)), 0.0f, 2.0f));
        W((ag_me.plants + ag_me.animals_total()) / std::max(1.0f, ag_me.owned));
        W(std::accumulate(me.seeds.begin(), me.seeds.end(), 0) / 50.0f);

        // =================================================================
        // OPPONENT CENSUS  (13)  -- summary + per-crop + per-animal
        // =================================================================
        endat(SC_OPP_FARM);
        at(SC_OPP_FARM);
        W(ag_op.plants / 50.0f);
        W(ag_op.animals_total() / 10.0f);
        W(ag_op.ripe / 50.0f);
        W(std::clamp(ag_op.yield_value / 10000.0f, 0.0f, 1.0f));
        W(std::clamp(ag_op.ops / 100.0f, 0.0f, 1.0f));
        for (int c = 0; c < NUM_CROPS; ++c)   W(ag_op.per_crop[c] / 25.0f);
        for (int a = 0; a < NUM_ANIMALS; ++a) W(ag_op.animals[a] / 10.0f);

        // =================================================================
        // FORECAST  (111)  -- the board for the conv-less global MLP
        // =================================================================
        // Policy-visible: pure functions of public tile state, the own shed
        // and the market. See forecast.hpp.
        endat(SC_FCAST);
        at(SC_FCAST);
        fcast::write_features_shared(fsh, game.players, p, game.market,
                                     game.unlocked_shops, day, game.turn, W);

        // =================================================================
        // CRITIC ONLY, from SC_CRIT_0 on  (29)
        // =================================================================
        endat(SC_CRIT_0);
        at(SC_CRIT_0);
        const float net_diff = net_op - net_me;
        W(std::clamp(net_op   /   1000.0f, -1.0f, 1.0f));
        W(std::clamp(net_op   /  10000.0f, -1.0f, 1.0f));
        W((net_op   / 100000.0f));
        W(std::clamp(net_diff /   1000.0f, -1.0f, 1.0f));
        W(std::clamp(net_diff /  10000.0f, -1.0f, 1.0f));
        W(std::clamp(net_diff / 100000.0f, -1.0f, 1.0f));
        for (int k = 0; k < NUM_ITEMS; ++k) W(op.shed[k] / 50.0f);
        for (int c = 0; c < NUM_CROPS; ++c) W(op.seeds[c] / 20.0f);
        W(ag_op.owned / 100.0f);
        W(ag_op.weeds / 50.0f);
        W(ag_op.yield_units / 100.0f);
        {
            const float v = (float)shed_walk_value(op, game.market);
            W(std::clamp(v /  10000.0f, 0.0f, 1.0f));
            W(v / 100000.0f);
        }
        W(op.shed_total() / (float)SHED_CAPACITY);

        // =================================================================
        // CRITIC ONLY: THE PROJECTION  (190)  -- see projection.hpp
        // =================================================================
        endat(SC_PROJ);
        at(SC_PROJ);
        proj::write_features(pr, W);
        endat(SCALAR_DIM_VALUE);
    }
}

// ===========================================================================
// Sampling helpers
// ===========================================================================
int get_max_index(std::vector<policy_move>& policyy) {
    double max_value = -std::numeric_limits<double>::infinity();
    int index = 0;
    for (size_t i = 0; i < policyy.size(); ++i)
        if (policyy[i].value > max_value) { max_value = policyy[i].value; index = (int)i; }
    return index;
}

void my_softmax(std::vector<policy_move>& logits,
                std::vector<policy_move>& exps, float temperature)
{
    double max_val = -std::numeric_limits<double>::infinity();
    for (const auto& l : logits) max_val = std::max(max_val, l.value / temperature);

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i].value = std::exp(logits[i].value / temperature - max_val);
        sum += exps[i].value;
    }
    for (size_t i = 0; i < logits.size(); ++i) exps[i].value /= sum;
}

int multinomial(std::vector<policy_move>& policyy) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    const double r = dis(gen);
    double cumulative = 0.0;
    for (size_t i = 0; i < policyy.size(); ++i) {
        cumulative += policyy[i].value;
        if (r < cumulative) return (int)i;
    }
    return (int)policyy.size() - 1;
}

// ===========================================================================
// THE FILTER
// ===========================================================================
// SEQUENTIAL ACTION RESOLUTION. The order is fixed and the state is threaded
// through it: each pass decides against a PROJECTED board that already carries
// the earlier passes' accepted changes, and against a running budget that
// already carries their costs.
//
// THE ORDER:
//
//   0  SELL          proceeds fund everything
//   1  MUST SET      built off the live board: the forced harvests (an
//                    animal's only when it goes unfed), the three kinds of
//                    forced water, and the forced feeds. No care, no collect.
//   5  LAND
//   2b ORDER HOURS   fixed here, after land
//   6  FORCED PASS   reserved before any head draws. Hires on demand. When
//                    the goods run out a feed is dropped, and its animal's
//                    harvest put back (it escapes tonight).
//   7  HARVEST       every harvestable cell, in order of standing value,
//                    draws the cell's OWN TYPE's [None, Keep, Sell] triple
//                    (a forced one draws [Keep, Sell]: HOW it is banked).
//                    Keep rides in hand to the nightly sweep; Sell draws the
//                    SELL-TIME head, drops the units at the shed by that hour
//                    and sells them AT it. A bin the routes cannot absorb
//                    hires hands until it fits (up to the crew cap).
//   8a WHERE         a fixed distance rule per type (see the pass).
//   8b TYPE + COUNT  up to MACRO_TYPE_SLOTS type slots, drawn in order. Each
//                    slot picks Don't_Plant or one type not yet chosen whose
//                    ceiling (cells + money LEFT by the earlier slots) is >0.
//                    Don't_Plant ends planting for the day. A chosen type then
//                    draws its COUNT over 1..ceiling and is placed at once.
//                    No BUILD slot: a PLACE builds its structure itself. A
//                    DRAWN UNIT IS FORCED: if the routes cannot absorb it,
//                    hands are hired mid-plan (one at a time, up to the hard
//                    cap and the budget) until they can.
//                       want_watered     the planting-day water, always.
//                       want_pre_harvest a doomed ongoing crop whose units
//                                        are already being banked today is
//                                        replanted HARVEST, DIG, PLANT.
//   8c FERTILIZE     drawn per cell, AFTER primary, on every plant -- standing
//                    or sown this morning -- where fertilize_pays. Accepting
//                    one drags in the payout-day water. Never on a melon.
//                    Hires on demand.
//   9  FEED          drawn. Accepting a feed FORCES a care alongside it
//                    whenever must_care_after_feed holds. Hires on demand.
//                    On an animal the FORCED pass fed, the same pair is drawn
//                    as [No_Care, Care] by the feed head itself (H_FEED).
//   9b ESCAPE        forced: an animal still unfed tonight has its units
//                    harvested, since the escape would take them.
//   9c COLLECT       drawn per animal with fertilizer waiting (H_COLLECT),
//                    while the night shed has a slot. Hires on demand.
//  10  DROPS         the deferred Sell harvests become mid-day drops in the
//                    hours left, hiring on demand. A sale that no longer fits
//                    REMASKS its hours on the plan as it stands and redraws
//                    among them, KEEP, and (when nothing was built on the
//                    harvest since) NO_HARVEST.
//
// THERE IS NO HIRE HEAD. The day starts with the crew it has (hand 0 alone:
// nobody else survives the night) and every pass above hires, one hand at a
// time, when what it drew does not fit the routes -- up to the crew cap
// (16 hands) and what the day can pay for. Hands already on the day are
// always used before a new one is paid for, so the priority order above is
// also the order in which the crew grows.
//
// Then the routes are optimised, the morning orders chunked, and the script
// flattened. ALL HIRES are one A_HIRE entry carrying the count, so a mid-plan
// hire never moves the order hours the routes were priced against.
//
// THERE IS NO WATER PASS: every water worth an hour is implied by the must
// tier, the planting, or an accepted fertilize.
//
// THREE LEDGERS FOR ONE SHED. `room` is what commit_unit tests at hour 0;
// `night` charges every KEPT harvested unit to the nightly sweep, so the dump
// at end_of_day never hits the overflow bin; `mid_units` charges every SOLD
// harvested unit to the mid-day shed (room >= mid_units), so no mid-day drop
// does either.
//
// PRIORITY WHEN THE SHED IS TIGHT: harvest first, fertilizer last. A forced
// harvest is never refused and buys its slots out of the collects, newest
// first; a drawn harvest is all-or-nothing (A_HARVEST empties the tile, so a
// partial fill destroys the remainder) and counts the standing collects as
// room it may take; a collect into a full shed is refused outright; and a
// pre-water is only taken when the unit it grows has somewhere to land.
//
// DOOMED CELLS ARE GROUND. A plant that is a weed by morning either way
// (plant_is_doomed) is offered to the primary pass as clearable, so the ongoing
// crops turn over the morning after their last production night instead of
// costing a day to decay and a day to dig. cell_musts already refuses to spend
// an hour watering them. A doomed ongoing crop still holding units is forced
// to harvest that morning; the replant rides on the same visit as HARVEST,
// DIG, PLANT (want_pre_harvest).
//
// GRADIENT RULE. A head is drawn only where it had a real choice:
//   * illegal / unaffordable / pointless / forced -> never put in a candidate
//     set, so no log-prob and no entropy.
//   * a set with one candidate -> nothing to decide, nothing recorded.
//   * over capacity -> only knowable after the draw, so the set is recorded
//     and then retracted from all three parallel lists.
//   * a cell reached after every ceiling is used up is never drawn.
//
// WHEAT keeps MACRO_WHEAT_RESERVE in the shed for feed; only the excess is
// offered to the sell head, except on the last day.
//
// MONEY keeps a FEED RESERVE: enough to buy the wheat for today's FORCED
// feeds not yet secured plus tomorrow's FORCED feeds (every animal that will
// be must_feed tomorrow, given what today's plan feeds and cares), net of the
// wheat already in the shed. Only forced feeds may spend it. Hire, land,
// seeds, animals, fertilizer and optional feeds see what is left after it.
// ===========================================================================

// Per-head play mode (see HeadMode in game.hpp). Defined OUTSIDE the
// anonymous namespace so eval tools can set it through the extern.
uint8_t g_head_mode[NUM_HEADS] = {};   // all HM_FOLLOW

namespace {

// Every field of two goals equal. Pass 10 uses it to tell whether anything
// was asked of a cell after its harvest was routed.
bool goal_same(const cell_goal& a, const cell_goal& b) {
    return a.target_type == b.target_type && a.target_crop == b.target_crop
        && a.target_animal == b.target_animal
        && a.want_watered == b.want_watered && a.want_fertilized == b.want_fertilized
        && a.want_fed == b.want_fed && a.want_cared == b.want_cared
        && a.want_collect_fertilizer == b.want_collect_fertilizer
        && a.want_harvest == b.want_harvest && a.want_pre_water == b.want_pre_water
        && a.want_pre_harvest == b.want_pre_harvest && a.replant == b.replant
        && a.sell_hour == b.sell_hour && a.in_denominator == b.in_denominator;
}

int macro_fib(int n) {   // fib(0)=1, fib(1)=1, fib(2)=2, ...  as in do_hire
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { const int t = a + b; a = b; b = t; }
    return a;
}

// A replica of simulation::spawn_hand.
ipos macro_spawn(const std::vector<ipos>& farmers) {
    const std::array<ipos, 4>& tiles = shed_access_tiles();
    int occ[4] = { 0, 0, 0, 0 };
    for (const ipos& f : farmers)
        for (int i = 0; i < 4; ++i)
            if (tiles[i] == f) occ[i] += 1;
    int best = 0;
    for (int i = 1; i < 4; ++i) if (occ[i] < occ[best]) best = i;
    return tiles[best];
}

// ---------------------------------------------------------------------------
// One draw = one candidate set, recorded as (all candidate flat indices,
// chosen swapped to the back), which is the grouping policy_to_jointlog
// re-derives at train time.
// ---------------------------------------------------------------------------
struct MacroDraw {
    const float* out = nullptr;
    float temp = 1.0f;
    bool  greedy = false;
    std::vector<std::vector<drone_action>>* actions = nullptr;
    std::vector<double>* logs  = nullptr;
    std::vector<int>*    heads = nullptr;

    std::vector<policy_move>  lg, pb;
    std::vector<drone_action> cand;
    bool recorded = false;
    int  n_recorded = 0;

    // Returns the POSITION in `idx` that was chosen, or -1 for an empty set.
    int pick(const int* idx, int n, int head) {
        recorded = false;
        if (n <= 0) return -1;
        if (n == 1) return 0;             // nothing to decide, nothing to log

        lg.clear(); pb.clear(); cand.clear();
        for (int i = 0; i < n; ++i) {
            lg.push_back({ idx[i], (double)out[idx[i]], 0 });
            pb.push_back({ idx[i], 0.0, 0 });
            cand.push_back({ idx[i] });
        }
        bool g = greedy;
        if (!logs && head >= 0 && head < NUM_HEADS) {   // evals / opponents only
            if      (g_head_mode[head] == HM_SAMPLE) g = false;
            else if (g_head_mode[head] == HM_GREEDY) g = true;
        }
        my_softmax(lg, pb, g ? 1.0f : temp);
        const int i = g ? get_max_index(pb) : multinomial(pb);

        if (!g) {
            if (logs)  logs->push_back(std::log(pb[i].value + 1e-8));
            if (heads) heads->push_back(head);
            if (actions) {
                std::swap(cand[i], cand.back());
                actions->push_back(cand);
            }
            recorded = true;
            ++n_recorded;
        }
        return i;
    }

    // A position in the three parallel lists. rollback() undoes every set
    // recorded since, which is what a harvest needs when the SELL bin and its
    // sell-time draw both have to go because no crew can route them.
    struct Mark { size_t logs = 0, heads = 0, actions = 0; int n = 0; };
    Mark mark() const {
        Mark m;
        m.logs    = logs    ? logs->size()    : 0;
        m.heads   = heads   ? heads->size()   : 0;
        m.actions = actions ? actions->size() : 0;
        m.n = n_recorded;
        return m;
    }
    void rollback(const Mark& m) {
        if (logs    && logs->size()    > m.logs)    logs->resize(m.logs);
        if (heads   && heads->size()   > m.heads)   heads->resize(m.heads);
        if (actions && actions->size() > m.actions) actions->resize(m.actions);
        n_recorded = m.n;
        recorded = false;
    }

    // Remove the sets recorded in [from, to). For a draw the day ends up NOT
    // carrying out (a deferred sale that found no hours), so the head is not
    // credited with it. Erase later ranges first, or earlier marks go stale.
    void erase(const Mark& from, const Mark& to) {
        if (logs)    logs->erase(logs->begin() + from.logs, logs->begin() + to.logs);
        if (heads)   heads->erase(heads->begin() + from.heads, heads->begin() + to.heads);
        if (actions) actions->erase(actions->begin() + from.actions, actions->begin() + to.actions);
        n_recorded -= (to.n - from.n);
        recorded = false;
    }

    // Undo the most recent recorded set. Only valid immediately after the pick
    // that made it.
    void retract() {
        if (!recorded) return;
        if (logs    && !logs->empty())    logs->pop_back();
        if (heads   && !heads->empty())   heads->pop_back();
        if (actions && !actions->empty()) actions->pop_back();
        recorded = false;
        --n_recorded;
    }

    int binary(int base, int cell, int head) {
        const int idx[2] = { macro_cell_index(base, 0, cell),
                             macro_cell_index(base, 1, cell) };
        return pick(idx, 2, head);
    }
};

}  // namespace

// THE REFEREE'S HIRE RULE. The referee takes ONE hand per HIRE entry (and
// one quadrant per BUY_LAND), and every entry is one of the
// MAX_MARKET_ORDERS_PER_TURN slots. The trainer's planner counts a hire of
// any size as one entry; the submission counts it as the referee does.
static int referee_entries(const move& m) {
    if (m.type == A_HIRE || m.type == A_BUY_LAND) return std::max(1, m.count);
    return 1;
}
static void referee_expand(std::vector<move>& orders) {
    std::vector<move> flat;
    flat.reserve(orders.size() + 16);
    for (const move& m : orders) {
        const int n = referee_entries(m);
        if (n == 1) { flat.push_back(m); continue; }
        for (int k = 0; k < n; ++k) { move e = m; e.count = 1; flat.push_back(e); }
    }
    orders.swap(flat);
}

// force_order_hours > 0 replaces the section-2b estimate. *needed_hours comes
// back > 0 when the referee-counted orders did not fit the hours the routes
// were priced on, and the caller plans the day again with that many.
static void macro_plan_day_impl(const float* out, simulation& game, int player, MacroDay& d,
                    std::vector<std::vector<drone_action>>* actions,
                    std::vector<double>* log_probs,
                    std::vector<int>* head_ids,
                    float temperature, bool greedy, int print,
                    int force_order_hours, int* needed_hours)
{
    (void)print;
    d.clear();
    *needed_hours = 0;

    const player_state& live = game.players[player];
    const int day  = game.day();
    const int turn = game.turn;

    MacroDraw draw;
    draw.out = out; draw.temp = temperature; draw.greedy = greedy;
    draw.actions = actions; draw.logs = log_probs; draw.heads = head_ids;

    player_state proj = live;

    double budget = live.money;

    // =====================================================================
    // THE TWO SHED LEDGERS
    // =====================================================================
    // `room` -- HOUR 0. What commit_unit tests when it refuses a buy on a full
    // shed. Sells free it, buys take it, harvests are irrelevant to it.
    int room = SHED_CAPACITY - live.shed_total();

    // `night` -- what the shed would hold if every harvested unit stayed in
    // it all day:
    //
    //     start        live.shed_total()
    //   - sold         gone at hour 0
    //   + bought       arrives at hour 0 ...
    //   - picked up    ... and A_PICKUP takes it straight back out
    //   - from stock   an input taken from the MORNING stock and spent: -1
    //   + harvested    every unit A_HARVEST puts in a hand, pre-water included
    //   + collected    every A_COLLECT_FERTILIZER
    //
    // Keeping it under SHED_CAPACITY means the nightly dump cannot overflow.
    int night = live.shed_total();

    // `mid_units` -- MID-DAY. Every unit an HB_SELL harvest will drop at the
    // shed before its sell order goes out. At any hour the shed holds at most
    // what it held after the morning orders (SHED_CAPACITY - room) plus the
    // sell units dropped and not yet sold, so keeping
    //
    //     room >= mid_units
    //
    // means no mid-day drop can ever hit a full shed. Buys spend `room`, sells
    // spend `mid_units`, and both check the same inequality. A sold harvest
    // never reaches the nightly dump, so it is NOT charged to `night`.
    int mid_units = 0;

    // Accepted A_COLLECT_FERTILIZER cells, newest last. THE SHEDDING STACK:
    // fertilizer is the lowest-priority thing in the shed, so when a harvest
    // needs a slot these are given up first, one at a time, and the collect is
    // released from the route with it.
    std::vector<int> collect_cells;

    // =====================================================================
    // 0. SELL -- one head per product, drawn at the top of the day
    // =====================================================================
    int sold[NUM_PRODUCTS] = { 0 };
    int kept[NUM_ITEMS];
    for (int k = 0; k < NUM_ITEMS; ++k) kept[k] = live.shed[k];

    int n_sells = 0;
    const bool last_day = (day >= NUM_DAYS - 1);
    // One day's feed: one wheat per animal standing on the board this morning.
    int wheat_per_day = 0;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x)
            if (live.board[y][x].bought && live.board[y][x].animal >= 0) ++wheat_per_day;
    for (int k = 0; k < NUM_PRODUCTS; ++k) {
        // Wheat keeps a feed reserve; only the excess is sellable, except on
        // the last day when the whole stack is.
        int stack = live.shed[k];
        if (k == WHEAT && !last_day)
            stack = std::max(0, stack - MACRO_WHEAT_RESERVE);
        if (stack <= 0) { ++d.stats.rej_illegal; continue; }

        // WHAT CAN BE SOLD.
        //   every product but wheat   BINARY: hold it all, or sell it all
        //                             (bins 0 and "all"; the 1/3 and 2/3
        //                             outputs are never offered, so they take
        //                             no gradient)
        //   wheat                     hold it all, or sell everything EXCEPT
        //                             the feed for the next 3, 2 or 1 days
        //                             (macro_wheat_keep_days). A day's feed is
        //                             one wheat per animal on the board this
        //                             morning.
        // One candidate per DISTINCT count, first bin kept: with no animals
        // every "keep N days" sells the whole stack, and with a stack no
        // bigger than the feed they all sell nothing, so they collapse.
        int idx[MACRO_SELL_BINS], cnt[MACRO_SELL_BINS], n_c = 0;
        auto offer = [&](int b, int c) {
            for (int j = 0; j < n_c; ++j) if (cnt[j] == c) return;
            idx[n_c] = macro_sell_index(k, b);
            cnt[n_c] = c;
            ++n_c;
        };
        if (k != WHEAT) {
            offer(0, 0);
            offer(MACRO_SELL_BINS - 1, stack);
        } else {
            offer(0, 0);
            for (int b = 1; b < MACRO_SELL_BINS; ++b) {
                const int keep = last_day ? 0 : macro_wheat_keep_days(b) * wheat_per_day;
                offer(b, std::max(0, stack - keep));
            }
        }
        const int pos = draw.pick(idx, n_c, H_SELL);

        const int n = (pos >= 0) ? cnt[pos] : 0;
        if (n <= 0) continue;
        const double got = sell_proceeds(k, n, game.market.inventory[k]);
        budget += got;
        d.stats.raised += got;
        room  += n;      // gone before the buys resolve
        night -= n;      // and gone before the dump
        kept[k] -= n;
        sold[k] = n;
        ++n_sells;
    }

    int item_need[NUM_ITEMS] = { 0 };
    int item_bought[NUM_ITEMS] = { 0 };
    int seed_need[NUM_CROPS] = { 0 };

    auto unit_price = [&](int item) -> double {
        if (item == WHEAT || item == FERTILIZER)
            return (double)market_price(item, game.market.inventory[item]
                                              - 1 - item_bought[item]);
        if (item >= GOOSE && item < GOOSE + NUM_ANIMALS)
            return (double)ANIMALS[item - GOOSE].cost;
        return 0.0;
    };
    // ---- THE FEED RESERVE ---------------------------------------------
    // Money set aside for the FORCED feeds of today and tomorrow:
    //   today     must_feed cells whose forced feed has not been processed
    //             yet (filled from the must set, counted down by the forced
    //             pass)
    //   tomorrow  every animal on the PROJECTED board that will be must_feed
    //             tomorrow, found by running tonight's daily_refresh_animals
    //             on a copy of its tile. Feeding an animal today usually takes
    //             it off tomorrow's list; an animal placed today is on it.
    // Wheat already in the shed covers the need first. Only forced feeds may
    // spend the reserve; everything optional sees `spendable()`.
    int forced_feed_pending = 0;       // today's forced feeds not yet processed

    // Would this (projected) animal tile be must_feed tomorrow morning?
    // Mirrors simulation::daily_refresh_animals, then asks must_feed.
    auto forced_tomorrow = [&](const tile& pt) -> bool {
        if (pt.animal < 0) return false;
        tile t = pt;
        t.consecutive_unfed = t.fed_today ? 0 : t.consecutive_unfed + 1;
        if (t.consecutive_unfed >= 2) return false;     // gone tonight
        const AnimalDef& a = ANIMALS[t.animal];
        const int since = (day + 1) - t.phase - a.first_yield_day;
        if (since >= 0 && a.interval > 0 && since % a.interval == 0) {
            t.yield_units = std::min(a.max_held,
                t.yield_units + 1 + (t.fed_today ? t.pending_care_bonus : 0));
            t.pending_care_bonus = 0;
        }
        if (t.cared_today && t.fed_today) t.pending_care_bonus += 1;
        t.fed_today = 0;
        t.cared_today = 0;
        return must_feed(t, day + 1);
    };

    // Cached: rescanned only after an animal tile on `proj` changes (a feed,
    // a care, a placement). Every such site sets tmr_dirty.
    bool tmr_dirty = true;
    int  tmr_cache = 0;
    auto forced_tomorrow_count = [&]() -> int {
        if (!tmr_dirty) return tmr_cache;
        int n = 0;
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x)
                if (forced_tomorrow(proj.board[y][x])) ++n;
        tmr_cache = n;
        tmr_dirty = false;
        return n;
    };

    auto feed_reserve = [&]() -> double {
        const int need  = forced_feed_pending + forced_tomorrow_count();
        const int stock = std::max(0, kept[WHEAT] - item_need[WHEAT]);
        const int buy   = std::max(0, need - stock);
        if (buy <= 0) return 0.0;
        // Priced one unit at a time up the buy walk, as the buys will be.
        double c = 0.0;
        for (int j = 0; j < buy; ++j)
            c += (double)market_price(WHEAT, game.market.inventory[WHEAT]
                                             - 1 - item_bought[WHEAT] - j);
        return c;
    };
    auto spendable = [&]() -> double { return budget - feed_reserve(); };

    auto can_secure = [&](int item) -> bool {
        if (item_need[item] < kept[item]) return true;      // free, in the shed
        if (room - mid_units <= 0) return false;
        if (item == WHEAT) return budget >= unit_price(item);
        // A new animal is unfed tonight, so it is a forced feed tomorrow.
        double extra = 0.0;
        if (item >= GOOSE && item < GOOSE + NUM_ANIMALS)
            extra = unit_price(WHEAT);
        return spendable() >= unit_price(item) + extra;
    };
    // Never called unless can_secure just said yes.
    //   from the MORNING STOCK -- picked up and spent today: night -= 1.
    //   BOUGHT -- arrives at hour 0 (room -= 1) and is picked back out, so it
    //     is net zero on `night`.
    auto secure = [&](int item) {
        if (item_need[item] < kept[item]) { ++item_need[item]; --night; return; }
        const double p = unit_price(item);
        budget -= p;
        d.stats.spent += p;
        ++item_bought[item];
        ++item_need[item];
        --room;
    };
    auto can_secure_seed = [&](int c) -> bool {
        if (seed_need[c] < live.seeds[c]) return true;
        return spendable() >= (double)CROPS[c].seed_cost;
    };
    // Seeds never touch the shed, so neither counter moves here.
    auto secure_seed = [&](int c) {
        if (seed_need[c] >= live.seeds[c]) {
            budget -= (double)CROPS[c].seed_cost;
            d.stats.spent += (double)CROPS[c].seed_cost;
        }
        ++seed_need[c];
    };

    // =====================================================================
    // 1. THE MUST SET
    // =====================================================================
    struct MustCell { int cell; CellMust m; };
    std::vector<MustCell> musts;
    musts.reserve(32);
    for (int cell = 0; cell < CELLS; ++cell) {
        const tile& t = live.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        const CellMust m = cell_musts(t, day, turn);
        if (!m.any()) continue;
        musts.push_back({ cell, m });
        if (m.feed) ++forced_feed_pending;
        d.stats.must_ops += m.ops();
    }

    // ---- MUST PRIORITY (BUGFIX) -----------------------------------------
    // The forced pass walks `musts` IN ORDER, and when
    // the crew is short the later entries are the ones shed. They used to be
    // in CELL-INDEX order, so a life-support water on a 10-coin wheat in the
    // top row took the last free hour ahead of the feed that kept a 400-coin
    // cow. Order them by what is LOST if they are dropped, largest first.
    {
        auto spot = [&](int prod) -> double {
            return (prod >= 0 && prod < NUM_PRODUCTS) ? (double)game.market.prices[prod] : 0.0;
        };
        auto must_loss = [&](const MustCell& mc) -> double {
            const tile& t = live.board[mc.cell / BOARD_SIZE][mc.cell % BOARD_SIZE];
            const double standing = t.yield_units * spot(tile_product(t));
            double v = 0.0;
            // an escaping animal takes its purchase price AND what it holds
            if (mc.m.feed && animal_escapes_tonight(t, day) && t.animal >= 0)
                v += ANIMALS[t.animal].cost + standing;
            else if (mc.m.feed)                   // a care bonus about to be wiped
                v += t.pending_care_bonus * spot(tile_product(t));
            if (mc.m.harvest) v += standing;
            if (mc.m.water && t.type == T_PLANT && t.crop >= 0) {
                if (must_water(t, day))           // the whole plant
                    v += CROPS[t.crop].seed_cost + standing + spot(t.crop);
                else                              // one more unit
                    v += spot(t.crop);
            }
            if (mc.m.prewater && t.crop >= 0) v += spot(t.crop);
            if (mc.m.collect) v += spot(FERTILIZER);
            return v;
        };
        std::stable_sort(musts.begin(), musts.end(),
            [&](const MustCell& a, const MustCell& b) { return must_loss(a) > must_loss(b); });
    }

    // The coins today's FORCED feeds will actually spend on wheat, net of
    // the stock -- as many of them as `money` can pay for, walked up the buy
    // price. The hard part of the feed reserve: tomorrow's part may be spent
    // on hands that serve today's musts, this part may not.
    //
    // A hand is affordable for a must exactly when paying for it does not
    // cost a forced feed its wheat:  money - hand >= today_feed_cost(money).
    // Capping at `money` matters when the day cannot feed everything anyway:
    // the uncapped sum then exceeds the budget and blocked EVERY hire, so a
    // crew of one watched the rest of the must set die for want of a 1-coin
    // hand.
    auto today_feed_cost = [&](double money, int extra_feeds) -> double {
        const int need  = forced_feed_pending + extra_feeds;
        const int stock = std::max(0, kept[WHEAT] - item_need[WHEAT]);
        double c = 0.0;
        for (int j = 0; j < need - stock; ++j) {
            const double p = (double)market_price(WHEAT, game.market.inventory[WHEAT]
                                                          - 1 - item_bought[WHEAT] - j);
            if (c + p > money) break;
            c += p;
        }
        return c;
    };

    // The goal a must implies, on top of "leave the cell as it is".
    auto must_goal = [&](const CellMust& m, const tile& t) {
        cell_goal g{};
        if (t.type == T_PLANT && t.crop >= 0) {
            g.target_type = T_PLANT;
            g.target_crop = t.crop;
        } else if (t.type == T_COOP || t.type == T_PASTURE) {
            g.target_type   = t.type;
            g.target_animal = t.animal;
        } else {
            g.target_type = T_EMPTY;
        }
        g.in_denominator = 1;
        if (m.harvest) {
            const bool in_place = (t.animal >= 0) ||
                (t.type == T_PLANT && t.crop >= 0 && CROPS[t.crop].ongoing);
            if (in_place) {
                g.want_harvest = 1;
            } else {
                g.target_type   = T_EMPTY;   // A_HARVEST wipes the tile
                g.target_crop   = -1;
                g.target_animal = -1;
            }
        }
        if (m.prewater) g.want_pre_water = 1;
        if (m.water)    g.want_watered   = 1;
        if (m.feed)     g.want_fed       = 1;
        // Only ever set together with feed. build_task emits A_FEED first.
        if (m.care)     g.want_cared     = 1;
        if (m.collect)  g.want_collect_fertilizer = 1;
        return g;
    };

    // =====================================================================
    // SELL THE SLOWEST-RECOVERING STOCK
    // =====================================================================
    // One hour-0 sell of held stock, used whenever the day has to raise room
    // or coins it did not draw for. It sells the product whose price RECOVERS
    // slowest: the one the town drains least (a product no shop buys only
    // gets the town centre's 1 a day; fertilizer gets nothing at all), since
    // holding that product buys the least price over time. Only what the day
    // does not spend itself (kept - item_need), and wheat only above two
    // days of feed (it recovers fastest anyway, so it goes last). Sells go
    // out before every
    // buy, and the order hours are priced for all NUM_PRODUCTS sell entries,
    // so this can be called at any point in the day.
    auto drain_per_day = [&](int k) -> int {
        int r = (k == FERTILIZER) ? 0 : 1;                      // town centre
        const int per_day = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL;
        for (const int s : game.unlocked_shops) {
            const std::vector<int>& pr = SHOP_PRODUCTS[s];
            if (std::find(pr.begin(), pr.end(), k) != pr.end())
                r += per_day * (pr.size() == 1 ? 2 : 1);
        }
        return r;
    };
    auto sell_slowest = [&](int n) -> int {
        int done = 0;
        for (; done < n; ++done) {
            int best = -1;
            for (int k = 0; k < NUM_PRODUCTS; ++k) {
                // Wheat is the feed: two days of it for every animal stays.
                const int keep_back = (k == WHEAT) ? 2 * wheat_per_day : 0;
                if (kept[k] - item_need[k] - keep_back <= 0) continue;
                if (best < 0 || drain_per_day(k) < drain_per_day(best) ||
                    (drain_per_day(k) == drain_per_day(best) && kept[k] > kept[best]))
                    best = k;
            }
            if (best < 0) break;
            const double got = sell_proceeds(best, sold[best] + 1, game.market.inventory[best])
                             - sell_proceeds(best, sold[best],     game.market.inventory[best]);
            if (sold[best] == 0) ++n_sells;
            ++sold[best];
            --kept[best];
            budget += got;
            d.stats.raised += got;
            ++room;
            --night;
        }
        return done;
    };

    // ---- LEAVE ROOM IN THE SHED (BUGFIX) ----------------------------------
    // The forced tier needs shed slots: the wheat its feeds have to BUY
    // (commit_unit refuses a buy on a full shed -- an animal used to escape
    // with thousands in the bank), and every unit its harvests and collects
    // bring home (the nightly sweep DISCARDS what does not fit). Free that
    // many slots now, before anything else is decided.
    {
        int wheat_buy = 0, inflow = 0;
        for (const MustCell& mc : musts) {
            const tile& t = live.board[mc.cell / BOARD_SIZE][mc.cell % BOARD_SIZE];
            if (mc.m.feed) ++wheat_buy;
            if (mc.m.harvest) inflow += t.yield_units + (mc.m.prewater ? 2 : 0);
            if (mc.m.collect) ++inflow;
        }
        wheat_buy = std::max(0, wheat_buy - kept[WHEAT]);
        const int need = wheat_buy + inflow;
        if (need > room) sell_slowest(need - room);   // room: hour-0 free slots
    }

    // =====================================================================
    // 2. THE CREW CAP  --  there is no hire head
    // =====================================================================
    // Every hand past the ones the day starts with is hired ON DEMAND by the
    // pass whose drawn work does not fit (forced_hire, commit_hiring), up to
    // this cap and what the day can pay for.
    const int ceiling = std::min(MACRO_HIRE_HARD_MAX, MAX_UNITS);

    auto order_hours_for = [&](int entries) {
        return std::clamp(
            (entries + MAX_MARKET_ORDERS_PER_TURN - 1) / MAX_MARKET_ORDERS_PER_TURN,
            1, MACRO_MAX_ORDER_HOURS);
    };

    // =====================================================================
    // 5. GLOBAL -- BUY LAND
    // =====================================================================
    {
        const int extra = (int)live.unlocked_quadrants.size() - 1;   // NW is free
        if (extra >= 2) {
            ++d.stats.rej_illegal;
        } else if (spendable() < LAND_PRICES[extra]) {
            ++d.stats.rej_budget;
        } else {
            const int idx[2] = { MACRO_LAND_BASE + 0, MACRO_LAND_BASE + 1 };
            if (draw.pick(idx, 2, H_LAND) == 1) {
                budget -= LAND_PRICES[extra];
                d.stats.spent += LAND_PRICES[extra];
                const int q = LAND_ORDER[extra];
                for (int y = 0; y < BOARD_SIZE; ++y)
                    for (int x = 0; x < BOARD_SIZE; ++x)
                        if (quadrant_of(x, y) == q) proj.board[y][x].bought = 1;
                proj.unlocked_quadrants.push_back(q);
                d.stats.bought_land = true;
            }
        }
    }

    // =====================================================================
    // 2b. THE ORDER HOURS, NOW EXACTLY
    // =====================================================================
    // An upper bound on the morning chunks, so the planned clock is never
    // longer than the hours the script will actually get.
    // The hire entry is counted unconditionally: the primary pass may still
    // hire for a forced planting, and it must not change these hours.
    d.n_order_hours = order_hours_for(
        NUM_PRODUCTS + (d.stats.bought_land ? 1 : 0) + 1 + MACRO_MAX_BUY_ENTRIES);
    // THE REFEREE: that "+ 1" is a hire of ANY size -- the trainer's rule.
    // The estimate stays exactly the trainer's (the policy was trained on
    // these budgets); the chunking below checks it against the referee's
    // count, and a day that does not fit is planned again with more hours.
    if (force_order_hours > 0)
        d.n_order_hours = std::clamp(force_order_hours, 1, MACRO_MAX_ORDER_HOURS);
    const int est_order_hours = d.n_order_hours;
    const int turn_budget = TURNS_PER_DAY - d.n_order_hours;

    d.planner.init(proj, day, turn_budget);

    // ---- the accumulating grid ------------------------------------------
    cell_goal g[CELLS];
    for (int cell = 0; cell < CELLS; ++cell) {
        const tile& t = live.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        g[cell] = cell_goal{};
        if (t.type == T_PLANT && t.crop >= 0) {
            g[cell].target_type = T_PLANT;
            g[cell].target_crop = t.crop;
        } else if (t.type == T_COOP || t.type == T_PASTURE) {
            g[cell].target_type   = t.type;
            g[cell].target_animal = t.animal;
        } else {
            g[cell].target_type = T_EMPTY;
        }
    }

    // Install a tentative goal if the planner can still route it.
    // Build the task `trial` implies for this cell. False == nothing to do.
    auto task_for = [&](int cell, const cell_goal& trial, MacroPlanner::Task& t) -> bool {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        t.x = x; t.y = y;
        tile base_t = live.board[y][x];
        base_t.bought = proj.board[y][x].bought;
        return MacroPlanner::build_task(trial, base_t, day, t);
    };

    auto commit = [&](int cell, const cell_goal& trial) -> bool {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        MacroPlanner::Task t;
        t.x = x; t.y = y;

        // The task is ALWAYS built against the tile as it stands at hour 0,
        // with the goal carrying everything the day has asked of this cell so
        // far. Building it off `proj` instead reads the cell's EARLIER commits
        // as already done (animal placed, fed, collected, harvested) and drops
        // their ops from the rebuilt task -- goal_gate_passed is written for
        // the start-of-day tile. Only the bought flag comes from `proj`: land
        // bought today is still locked in `live`.
        tile base_t = live.board[y][x];
        base_t.bought = proj.board[y][x].bought;

        if (!MacroPlanner::build_task(trial, base_t, day, t)) {
            // Nothing left to ask for: drop whatever an earlier goal routed.
            // (A release the rest of the route cannot survive is refused and
            // the old goal stands -- see MacroPlanner::detach.)
            if (!d.planner.release(cell)) return false;
            g[cell] = trial;                   // already satisfied: free
            return true;
        }
        // DEEP: before a commit says no, the planner tries ejection and a
        // full re-optimisation -- every refusal here costs a hire or a draw.
        if (!d.planner.reserve(cell, t, /*deep=*/true)) return false;
        g[cell] = trial;
        return true;
    };

    // ---- GIVE UP ONE COLLECT --------------------------------------------
    // Fertilizer is last in the queue for a shed slot. Re-committing the goal
    // without the collect can only SHRINK the route, so it cannot fail.
    auto drop_collect = [&]() -> bool {
        if (collect_cells.empty()) return false;
        const int cell = collect_cells.back();
        collect_cells.pop_back();
        cell_goal ng = g[cell];
        ng.want_collect_fertilizer = 0;
        // BUGFIX: "a shorter task cannot fail to route" is not true -- the
        // lazy drop is a greedy rule, so removing ops can break a route that
        // holds a sell. If the planner refuses, the collect still happens:
        // leave the ledger alone and let the caller try the next one.
        if (!commit(cell, ng)) return true;
        proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE].fertilizer_available = 1;
        --night;
        ++d.stats.must_dropped;
        return true;
    };

    // The harvests the harvest pass draws on, with the value that orders them.
    // `forced` marks a must-harvest already routed as HB_KEEP by pass 6: it
    // only draws HOW it is banked ([Keep, Sell]), never whether.
    struct HCand { int cell; double value; bool forced; int units; };
    std::vector<HCand> hcands;
    auto product_price = [&](int prod) -> double {
        return (prod >= 0 && prod < NUM_PRODUCTS) ? (double)game.market.prices[prod] : 0.0;
    };

    // A hand hired INSIDE the forced pass, whenever a must does not fit the
    // crew so far (there is no hire head: this is how the forced tier gets
    // its hands). Paid out of the budget, never out of today's forced wheat
    // (`extra` is wheat this must itself still has to buy).
    auto forced_hire = [&](double extra) -> bool {
        if ((int)proj.farmers.size() >= ceiling) return false;
        const double c = FARM_HAND_COST_MULT
                       * (double)macro_fib(live.hires_today + d.stats.hired);
        // Short of coins with goods in the shed: a held unit sold at hour 0
        // buys the hand (they cost 1, 1, 2, 3, ...), rather than letting the
        // must it would serve die.
        while (budget - c < today_feed_cost(budget, 0) + extra && sell_slowest(1) > 0) {}
        if (budget - c < today_feed_cost(budget, 0) + extra) return false;
        budget -= c;
        d.stats.spent += c;
        proj.farmers.push_back(macro_spawn(proj.farmers));
        proj.carried.push_back(unit_inv{});
        if (!d.planner.add_hand(proj.farmers.back())) return false;
        ++d.stats.hired;
        return true;
    };

    // =====================================================================
    // 6. THE FORCED PASS
    // =====================================================================
    for (const MustCell& mc : musts) {
        const int y = mc.cell / BOARD_SIZE, x = mc.cell % BOARD_SIZE;
        const tile& lt = live.board[y][x];
        tile& pt = proj.board[y][x];

        CellMust m = mc.m;

        // This forced feed leaves the "today" part of the reserve now: from
        // here it is either secured below or dropped for want of wheat.
        if (m.feed) --forced_feed_pending;

        // ---- goods first ------------------------------------------------
        // BUGFIX: short of coins with goods in the shed, sell a held unit
        // rather than lose the animal. (Shed ROOM is already reserved at the
        // top of the day -- see LEAVE ROOM IN THE SHED.)
        while (m.feed && !can_secure(WHEAT) && sell_slowest(1) > 0) {}
        if (m.feed && !can_secure(WHEAT)) {   // no wheat, no feeding it
            m.feed = false;
            ++d.stats.must_dropped;
            if (m.care) {                     // and no fed night to care on
                m.care = false;
                ++d.stats.must_dropped;
            }
            // cell_musts dropped this animal's harvest BECAUSE it was being
            // fed. Unfed, it escapes tonight: bank what it holds.
            if (lt.animal >= 0 && animal_escapes_tonight(lt, day) &&
                harvest_ready(lt, day) && !m.harvest) {
                m.harvest = true;
                ++d.stats.must_ops;
            }
        }
        // A collect into a shed already full at nightfall is refused.
        if (m.collect && night >= SHED_CAPACITY) {
            m.collect = false;
            ++d.stats.must_dropped;
        }
        if (!m.any()) continue;

        // ---- then hours -------------------------------------------------
        // Before shedding anything forced, tighten the routes once and retry.
        // Shed order: care, collect.
        bool ok = commit(mc.cell, must_goal(m, lt));
        if (!ok && d.planner.tighten()) ok = commit(mc.cell, must_goal(m, lt));
        // BUGFIX: a forced op is worth a hand before it is worth shedding.
        {
            const double own_wheat =
                (m.feed && item_need[WHEAT] >= kept[WHEAT]) ? unit_price(WHEAT) : 0.0;
            while (!ok && forced_hire(own_wheat)) {
                ++d.stats.must_hires;
                ok = commit(mc.cell, must_goal(m, lt));
            }
        }
        if (!ok && m.care) {
            m.care = false;
            ++d.stats.must_dropped;
            ok = commit(mc.cell, must_goal(m, lt));
        }
        if (!ok && m.collect) {
            m.collect = false;
            ++d.stats.must_dropped;
            ok = m.any() ? commit(mc.cell, must_goal(m, lt)) : false;
        }
        if (!ok) {
            d.stats.must_dropped += m.ops();
            ++d.stats.rej_capacity;
            continue;
        }
        d.stats.must_placed += m.ops();

        if (m.harvest) {
            const bool in_place = (pt.animal >= 0) ||
                (pt.type == T_PLANT && pt.crop >= 0 && CROPS[pt.crop].ongoing);
            // Forced harvests are never refused for want of room. The
            // pre-water is part of the charge.
            int units = pt.yield_units;
            if (m.prewater && pt.type == T_PLANT && pt.crop >= 0) {
                const int bonus = (lt.fertilized >= day) ? 2 : 1;
                units = std::min(CROPS[pt.crop].max_yield, units + bonus);
            }
            night += units;
            ++d.stats.harvest_keep;
            if (units > 0) {
                const int prod = tile_product(pt);
                hcands.push_back({ mc.cell, (double)units * product_price(prod),
                                   /*forced=*/true, units });
            }
            if (in_place) pt.yield_units = 0;
            else          pt = tile{ T_EMPTY, 1 };
            tmr_dirty = true;   // an animal's held units feed must_feed
        }
        if (m.water) pt.watered_today = 1;
        if (m.feed)  { secure(WHEAT); pt.fed_today = 1; tmr_dirty = true; }
        if (m.care)  { pt.cared_today = 1; ++d.stats.cared; tmr_dirty = true; }
        if (m.collect) {
            pt.fertilizer_available = 0;
            ++night;                        // a real inflow, charged like any
            collect_cells.push_back(mc.cell);
        }
        // A forced harvest that overflows the day is paid for out of the
        // collects, newest first.
        while (night > SHED_CAPACITY && drop_collect()) {}
    }

    // =====================================================================
    // HIRING ON DEMAND  --  shared by the harvest and planting passes
    // =====================================================================
    // A drawn unit of work the routes cannot absorb -- even after the planner's
    // deep search (ejection, re-optimisation) -- hires hands one at a time,
    // up to the hard crew cap and what the day can pay for, until it fits.
    auto next_hand_cost = [&](int k) -> double {
        return FARM_HAND_COST_MULT * (double)macro_fib(live.hires_today + d.stats.hired + k);
    };
    // The fewest extra hands that make `trial` routable, measured on a scratch
    // copy of the live plan; -1 if no affordable crew under the cap does it.
    // A cell already routed is released in the scratch first, so a NEW hand
    // may take it whole.
    auto hires_to_fit = [&](int cell, const cell_goal& trial, double extra) -> int {
        MacroPlanner::Task tk;
        if (!task_for(cell, trial, tk)) return 0;
        MacroPlanner sc = d.planner;
        sc.release(cell);
        std::vector<ipos> crew = proj.farmers;
        double spend = 0.0;
        for (int n = 1; ; ++n) {
            if ((int)crew.size() >= ceiling) return -1;
            const double c = next_hand_cost(n - 1);
            if (spend + c + extra > spendable()) return -1;
            spend += c;
            crew.push_back(macro_spawn(crew));
            if (!sc.add_hand(crew.back())) return -1;
            if (sc.reserve(cell, tk)) return n;
        }
    };
    auto hire_one = [&](int& counter) {
        const double c = next_hand_cost(0);
        budget -= c;
        d.stats.spent += c;
        proj.farmers.push_back(macro_spawn(proj.farmers));
        proj.carried.push_back(unit_inv{});
        d.planner.add_hand(proj.farmers.back());
        ++d.stats.hired;
        ++counter;
    };
    // commit(), and if the cell was already routed on a hand that cannot
    // absorb the bigger task, let it move to any hand.
    //
    // BUGFIX: on failure the plan is restored from a SNAPSHOT. It used to
    // re-commit the old goal on the grounds that it "fitted a moment ago" --
    // but the release and the failed reserve may have re-shaped every route
    // in between, so that commit could fail, and then the cell's earlier
    // goal (a forced feed or harvest, say) stayed in g[] with nothing routed
    // to carry it out.
    auto commit_moving = [&](int cell, const cell_goal& trial) -> bool {
        if (commit(cell, trial)) return true;
        if (!d.planner.routed(cell)) return false;
        const MacroPlanner snap = d.planner;
        if (d.planner.release(cell)) {
            MacroPlanner::Task tk;
            if (task_for(cell, trial, tk) && d.planner.reserve(cell, tk, true)) {
                g[cell] = trial;
                return true;
            }
        }
        d.planner = snap;
        return false;
    };
    // commit_moving(), then the same after tightening every route, and only
    // then hire until it fits (or cannot). Hands the day already has are
    // always used before a new one is paid for.
    auto commit_hiring = [&](int cell, const cell_goal& trial, double extra,
                             int& counter) -> bool {
        if (commit_moving(cell, trial)) return true;
        if (d.planner.tighten() && commit_moving(cell, trial)) return true;
        const int need = hires_to_fit(cell, trial, extra);
        if (need <= 0) return false;
        for (int k = 0; k < need; ++k) hire_one(counter);
        return commit_moving(cell, trial);
    };

    // =====================================================================
    // 7. HARVEST  --  [None, Keep, Sell] per type, in order of STANDING VALUE
    // =====================================================================
    // Every harvest the day could bank, highest units x spot price first, so
    // the most valuable units get first claim on the shed and the hours:
    //
    //   OPTIONAL  every cell still harvest_ready on the projected board.
    //             Draws the cell's OWN TYPE's triple:
    //               HB_NONE  leave it
    //               HB_KEEP  offered while the night's shed can take it
    //                        (counting the collects it may evict)
    //               HB_SELL  offered while the MID-DAY bound can take it AND
    //                        some sell hour is reachable by any hand at all
    //   FORCED    a must-harvest routed as HB_KEEP by pass 6. The harvest is
    //             not a choice; the banking is: [HB_KEEP, HB_SELL] from the
    //             same triple, HB_NONE masked.
    //
    // HB_SELL then draws the SELL-TIME head: the hours a hand could still make
    // (MacroPlanner::min_drop_hour, a lower bound), plus
    //   ST_NO_HARVEST  optional harvests only: leave the cell standing
    //   ST_KEEP        whenever the units could be kept: bank them for the
    //                  nightly sweep instead
    // The hour is the drop's DEADLINE and the sell's EXACT hour.
    //
    // A SALE THAT DOES NOT ROUTE IS REMASKED, NOT DROPPED. Hands are hired
    // first (commit_hiring). If it still does not fit, the hours are re-tested
    // EXACTLY on the plan as it stands (exact_sell_bins) and the sell-time head
    // is drawn again over the ones that still route plus the fallbacks that
    // are still legal. The failed draw is taken out of the record, and if only
    // one option is left, the harvest head's Sell goes with it: no choice was
    // made at all.

    // The goal a harvest of `cell` implies on top of whatever it holds.
    auto harvest_goal = [&](int cell, bool in_place, bool take_pre) {
        cell_goal trial = g[cell];
        trial.in_denominator = 1;
        if (in_place) {
            trial.want_harvest = 1;
        } else {
            if (take_pre) trial.want_pre_water = 1;
            trial.want_watered  = 0;
            trial.target_type   = T_EMPTY;
            trial.target_crop   = -1;
            trial.target_animal = -1;
        }
        return trial;
    };

    // ---- THE SELL-TIME HEAD -----------------------------------------------
    enum SellKind { SK_NONE = 0, SK_HOUR, SK_NO_HARVEST, SK_KEEP };
    struct SellPick { int kind = SK_NONE; int hour = -1; };

    // FIRST DRAW's hours: the bins a hand could POSSIBLY make for `trial` (a
    // goal whose sell_hour is set, to anything, so its task is priced as a
    // sale) -- a fresh hand doing nothing else. Cheap, and only a lower bound:
    // an hour offered here can still fail to route, which the remask catches.
    auto sell_bins_for = [&](int cell, const cell_goal& trial, int* bins) -> int {
        MacroPlanner::Task tk;
        if (!task_for(cell, trial, tk) || !tk.sells()) return 0;
        const int lo = d.planner.min_drop_hour(cell, tk);
        int n = 0;
        for (int b = 0; b < MACRO_SELL_HOUR_BINS; ++b)
            if (MACRO_SELL_HOURS[b] >= lo) bins[n++] = b;
        return n;
    };

    // One draw over `bins` and the fallbacks allowed. A single option is taken
    // without a draw (draw.recorded then reads false).
    auto draw_sell = [&](int cell, const int* bins, int nb,
                         bool allow_no, bool allow_keep) -> SellPick {
        int idx[MACRO_SELL_TIME_BINS], kind[MACRO_SELL_TIME_BINS], hour[MACRO_SELL_TIME_BINS];
        int n = 0;
        for (int i = 0; i < nb; ++i) {
            idx[n] = macro_sell_time_index(bins[i], cell);
            kind[n] = SK_HOUR;
            hour[n] = MACRO_SELL_HOURS[bins[i]];
            ++n;
        }
        if (allow_no) {
            idx[n] = macro_sell_time_index(ST_NO_HARVEST, cell);
            kind[n] = SK_NO_HARVEST; hour[n] = -1; ++n;
        }
        if (allow_keep) {
            idx[n] = macro_sell_time_index(ST_KEEP, cell);
            kind[n] = SK_KEEP; hour[n] = -1; ++n;
        }
        SellPick r;
        const int pos = draw.pick(idx, n, H_SELL_TIME);
        if (pos < 0) return r;
        r.kind = kind[pos];
        r.hour = hour[pos];
        return r;
    };

    // EXACT: would a sale of `cell` at trial.sell_hour route on the plan as it
    // stands -- on the current crew, after a tighten, or with hands the day can
    // still hire? Every test runs against a snapshot and is undone, so nothing
    // is paid for and no route moves.
    auto sale_fits = [&](int cell, const cell_goal& trial) -> bool {
        MacroPlanner::Task tk;
        if (!task_for(cell, trial, tk) || !tk.sells()) return false;
        if (d.planner.min_drop_hour(cell, tk) > trial.sell_hour) return false;
        const MacroPlanner snap = d.planner;
        const cell_goal gs = g[cell];
        bool ok = commit_moving(cell, trial);
        if (!ok && d.planner.tighten()) ok = commit_moving(cell, trial);
        d.planner = snap;
        g[cell] = gs;
        return ok || hires_to_fit(cell, trial, 0.0) > 0;
    };

    // THE REMASK: the hour bins that still route. A later hour only LOOSENS a
    // sale (the drop's deadline moves out, nothing else changes), so the
    // routable bins are a suffix, and a binary search finds where it starts:
    // four exact tests instead of twelve.
    auto exact_sell_bins = [&](int cell, cell_goal trial, int* bins) -> int {
        int lo = 0, hi = MACRO_SELL_HOUR_BINS;          // first routable bin
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            trial.sell_hour = MACRO_SELL_HOURS[mid];
            if (sale_fits(cell, trial)) hi = mid;
            else                        lo = mid + 1;
        }
        int n = 0;
        for (int b = lo; b < MACRO_SELL_HOUR_BINS; ++b) bins[n++] = b;
        return n;
    };

    // REDRAW after a sale failed to route: remask, draw over the hours that
    // still route plus the allowed fallbacks, and route the hour it picks
    // (hiring as needed). An hour that STILL fails -- the remask's hire test
    // is a shallower search than the real commit -- is struck off, its draw
    // retracted, and the set redrawn. `recorded` says whether the final pick
    // was a real choice (two or more options) and stays in the record.
    auto redraw_sale = [&](int cell, cell_goal trial, bool hours_ok,
                           bool allow_no, bool allow_keep, bool& recorded) -> SellPick {
        int bins[MACRO_SELL_HOUR_BINS];
        int nb = hours_ok ? exact_sell_bins(cell, trial, bins) : 0;
        for (;;) {
            const SellPick p = draw_sell(cell, bins, nb, allow_no, allow_keep);
            recorded = draw.recorded;
            if (p.kind != SK_HOUR) return p;
            trial.sell_hour = p.hour;
            if (commit_hiring(cell, trial, 0.0, d.stats.harvest_hires)) return p;
            if (recorded) draw.retract();
            recorded = false;
            int k = 0;
            for (int i = 0; i < nb; ++i)
                if (MACRO_SELL_HOURS[bins[i]] != p.hour) bins[k++] = bins[i];
            nb = k;
        }
    };

    // ---- DROPS GET THE LEFTOVER HOURS -------------------------------------
    // A plant is worth more than the timing of a sale, so an HB_SELL that
    // could be KEPT is not routed as a sale here. The harvest is routed as
    // HB_KEEP (the units ride home at night) and the sale is PENDING: once
    // planting, fertilizing and feeding have taken the hours they want, pass
    // 10 turns each pending sale into a real mid-day drop, hiring if it must.
    // If it no longer fits, pass 10 remasks (see above). Only a sale that
    // CANNOT be kept -- the night is full -- is routed at once.
    //
    // A pending sale remembers enough to UNDO its harvest, which is what
    // NO_HARVEST means on a remask: the goal and projected tile from before
    // the harvest, the goal right after it (to tell whether anything was
    // built on the cell since), and the collects it evicted from the shed.
    struct PendingSell {
        int  cell = -1, hour = -1, units = 0;
        bool forced = false;
        // [from, st_from): the harvest head's Sell.  [st_from, to): the hour.
        MacroDraw::Mark from, st_from, to;
        cell_goal g_before, g_after;
        tile      t_before;
        std::vector<int> shed_for;
    };
    std::vector<PendingSell> pending_sells;

    // Collects a failed or undone harvest had evicted from the shed, put back
    // newest first, each only if its route still takes it.
    auto restore_collects = [&](const std::vector<int>& cells) {
        for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
            const int c = *it;
            cell_goal ng = g[c];
            ng.want_collect_fertilizer = 1;
            if (!commit(c, ng)) continue;
            proj.board[c / BOARD_SIZE][c % BOARD_SIZE].fertilizer_available = 0;
            ++night;
            collect_cells.push_back(c);
            --d.stats.must_dropped;
        }
    };

    // ---- a FORCED harvest: [Keep, Sell] ----------------------------------
    auto bank_forced = [&](const HCand& hc) {
        const int cell = hc.cell;
        const tile& lt = live.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        const int htype = tile_harvest_type(lt);
        if (htype < 0 || !g[cell].in_denominator) return;
        if (room - mid_units < hc.units) return;             // mid-day bound
        // Wheat / carrot: no sell time. Kept while the night has room.
        const bool timed = harvest_type_has_sell_time(htype);
        if (!timed && night <= SHED_CAPACITY) return;

        cell_goal trial = g[cell];
        trial.sell_hour = TURNS_PER_DAY - 1;                  // priced as a sale
        int tb[MACRO_SELL_HOUR_BINS];
        const int nb = sell_bins_for(cell, trial, tb);
        if (nb <= 0) return;                                  // Keep is the only option

        if (!timed) {
            // The night is full: sell at the fixed hour, no draw. The bins are
            // a suffix ending at 23, so nb > 0 means 23 is in reach.
            trial.sell_hour = MACRO_FORCED_SELL_HOUR;
            if (!commit_hiring(cell, trial, 0.0, d.stats.harvest_hires)) {
                ++d.stats.sell_fallback;                      // stays HB_KEEP
                return;
            }
            night -= hc.units;
            mid_units += hc.units;
            --d.stats.harvest_keep;
            ++d.stats.harvest_sell;
            return;
        }

        const MacroDraw::Mark mk = draw.mark();
        if (night <= SHED_CAPACITY) {
            const int hidx[2] = { macro_harvest_index(htype, HB_KEEP, cell),
                                  macro_harvest_index(htype, HB_SELL, cell) };
            if (draw.pick(hidx, 2, H_HARVEST) != 1) return;   // Keep: already routed
            // A forced harvest cannot be left standing: hours and KEEP.
            const MacroDraw::Mark st = draw.mark();
            const SellPick sp = draw_sell(cell, tb, nb, /*no=*/false, /*keep=*/true);
            if (sp.kind != SK_HOUR) { ++d.stats.sell_keep; return; }
            // Already routed as Keep; the drop waits for leftover hours.
            PendingSell ps;
            ps.cell = cell; ps.hour = sp.hour; ps.units = hc.units; ps.forced = true;
            ps.from = mk; ps.st_from = st; ps.to = draw.mark();
            pending_sells.push_back(std::move(ps));
            return;
        }

        // KEEP IS NOT OFFERED INTO AN OVERFULL NIGHT. A forced harvest is never
        // refused, so when the night is already past capacity a kept unit is a
        // unit the sweep throws away. Selling is then the only way to bank it:
        // the hours alone, no harvest-head draw. (hcands is in value order, so
        // the most valuable forced harvests are switched first.)
        const SellPick sp = draw_sell(cell, tb, nb, false, false);
        bool routed = false;
        if (sp.kind == SK_HOUR) {
            trial.sell_hour = sp.hour;
            routed = commit_hiring(cell, trial, 0.0, d.stats.harvest_hires);
        }
        if (!routed) {
            draw.rollback(mk);
            bool rec = false;
            const SellPick rp = redraw_sale(cell, trial, true, false, false, rec);
            routed = rp.kind == SK_HOUR;
            if (routed) ++d.stats.sell_redrawn;
        }
        if (!routed) { ++d.stats.sell_fallback; return; }     // stays HB_KEEP
        night -= hc.units;
        mid_units += hc.units;
        --d.stats.harvest_keep;
        ++d.stats.harvest_sell;
    };

    // ---- an OPTIONAL harvest: [None, Keep, Sell] -------------------------
    auto try_harvest = [&](int cell) -> bool {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        tile& pt = proj.board[y][x];
        if (!harvest_ready(pt, day)) { ++d.stats.rej_illegal; return false; }

        const int htype = tile_harvest_type(pt);
        if (htype < 0) { ++d.stats.rej_illegal; return false; }

        const bool in_place = (pt.animal >= 0) ||
                              (pt.type == T_PLANT && pt.crop >= 0 &&
                               CROPS[pt.crop].ongoing);

        const int base_units = pt.yield_units;
        int pre_bonus = 0;
        bool water_adds = false;
        if (!in_place) {
            tile before = pt;
            before.watered_today = live.board[y][x].watered_today;
            water_adds = water_yields_now(before, day);
            if (water_adds) {
                const int bonus = (live.board[y][x].fertilized >= day) ? 2 : 1;
                pre_bonus = std::min(CROPS[pt.crop].max_yield,
                                     base_units + bonus) - base_units;
            }
        }
        const bool had_water = g[cell].want_watered != 0;
        const cell_goal g_before = g[cell];
        const tile      t_before = pt;

        // ---- which bins are on offer -------------------------------------
        // KEEP: exact room at nightfall, all or nothing (A_HARVEST empties the
        // tile, so a partial fill would destroy the remainder), counting the
        // collects it may evict.
        const bool keep_ok =
            night + base_units <= SHED_CAPACITY + (int)collect_cells.size();
        // SELL: the mid-day bound, and an hour some hand could still make.
        const bool sell_pre = water_adds && room - mid_units >= base_units + pre_bonus;
        cell_goal sell_trial = harvest_goal(cell, in_place, sell_pre);
        sell_trial.sell_hour = TURNS_PER_DAY - 1;             // priced as a sale
        int tb[MACRO_SELL_HOUR_BINS];
        const int nb = (room - mid_units >= base_units)
                     ? sell_bins_for(cell, sell_trial, tb) : 0;
        // Wheat / carrot: no sell time. Sell is offered only when the night
        // cannot take the units, and then always at the fixed hour.
        const bool timed = harvest_type_has_sell_time(htype);
        const bool sell_ok = nb > 0 && (timed || !keep_ok);

        if (!keep_ok && !sell_ok) { ++d.stats.rej_budget; return false; }

        int hidx[MACRO_HARVEST_BINS], bin_of[MACRO_HARVEST_BINS], n = 0;
        hidx[n] = macro_harvest_index(htype, HB_NONE, cell); bin_of[n++] = HB_NONE;
        if (keep_ok) { hidx[n] = macro_harvest_index(htype, HB_KEEP, cell); bin_of[n++] = HB_KEEP; }
        if (sell_ok) { hidx[n] = macro_harvest_index(htype, HB_SELL, cell); bin_of[n++] = HB_SELL; }

        const MacroDraw::Mark mk = draw.mark();
        const int pos = draw.pick(hidx, n, H_HARVEST);
        const int bin = (pos >= 0) ? bin_of[pos] : HB_NONE;
        if (bin == HB_NONE) return false;

        // ---- HB_SELL: the sell-time head ----------------------------------
        int deferred_hour = -1;
        MacroDraw::Mark st;
        if (bin == HB_SELL) {
            st = draw.mark();
            SellPick sp;
            if (timed) {
                sp = draw_sell(cell, tb, nb, /*no=*/true, /*keep=*/keep_ok);
            } else {                                          // no draw: hour 23
                sp.kind = SK_HOUR;
                sp.hour = MACRO_FORCED_SELL_HOUR;
            }
            if (sp.kind == SK_NO_HARVEST) { ++d.stats.sell_skip; return false; }
            if (sp.kind == SK_KEEP) {
                ++d.stats.sell_keep;                          // routed below as HB_KEEP
            } else if (keep_ok) {
                deferred_hour = sp.hour;                      // HB_KEEP now, a drop in pass 10
            } else {
                // Cannot be kept (the night is full): route the sale now.
                cell_goal trial = sell_trial;
                trial.sell_hour = sp.hour;
                int units = base_units + (sell_pre ? pre_bonus : 0);
                bool ok = commit(cell, trial);
                if (!ok && sell_pre) {                    // the pre-water goes first
                    trial.want_pre_water = 0;
                    units = base_units;
                    ok = commit(cell, trial);
                }
                if (!ok) ok = commit_hiring(cell, trial, 0.0, d.stats.harvest_hires);
                if (!ok && !timed) {
                    // Hour 23 is the loosest deadline there is, so no earlier
                    // hour would route either: the harvest is off.
                    draw.rollback(mk);
                    ++d.stats.sell_fallback;
                    ++d.stats.rej_capacity;
                    return false;
                }
                if (!ok) {
                    // REMASK, without the pre-water and without KEEP (the
                    // night is full): the hours that still route, or leave it.
                    draw.rollback(st);
                    trial.want_pre_water = 0;
                    units = base_units;
                    bool rec = false;
                    const SellPick rp = redraw_sale(cell, trial, true,
                                                    /*no=*/true, /*keep=*/false, rec);
                    if (rp.kind == SK_HOUR) {
                        ok = true;
                        trial.sell_hour = rp.hour;
                        ++d.stats.sell_redrawn;
                    } else {
                        if (rec) ++d.stats.sell_skip;          // chose to leave it
                        else {                                 // nothing left to choose
                            draw.rollback(mk);
                            ++d.stats.sell_fallback;
                            ++d.stats.rej_capacity;
                        }
                        return false;
                    }
                }
                if (trial.want_pre_water && !had_water) {
                    ++d.stats.must_ops;
                    ++d.stats.must_placed;
                }
                mid_units += units;
                ++d.stats.harvest_sell;
                if (in_place) pt.yield_units = 0;
                else          pt = tile{ T_EMPTY, 1 };
                tmr_dirty = true;
                return true;
            }
        }

        // ---- HB_KEEP ------------------------------------------------------
        // Collects given up for THIS harvest, newest first, so they can be
        // put back if the harvest then fails to route (or is undone).
        std::vector<int> shed_for_this;
        while (night + base_units > SHED_CAPACITY && !collect_cells.empty()) {
            shed_for_this.push_back(collect_cells.back());
            drop_collect();
        }

        const bool take_pre =
            water_adds && (night + base_units + pre_bonus <= SHED_CAPACITY);
        if (!take_pre) pre_bonus = 0;

        cell_goal trial = harvest_goal(cell, in_place, take_pre);
        const bool must_pre = take_pre && !had_water;

        bool ok = commit(cell, trial);
        if (!ok && take_pre) {
            trial.want_pre_water = 0;
            pre_bonus = 0;
            ok = commit(cell, trial);
        }
        if (!ok) ok = commit_hiring(cell, trial, 0.0, d.stats.harvest_hires);
        if (must_pre) {
            ++d.stats.must_ops;
            if (ok && trial.want_pre_water) ++d.stats.must_placed;
            else                            ++d.stats.must_dropped;
        }
        if (!ok) {
            draw.rollback(mk);
            ++d.stats.rej_capacity;
            // The harvest is off, so the collects it evicted are owed back.
            restore_collects(shed_for_this);
            return false;
        }

        night += base_units + pre_bonus;
        ++d.stats.harvest_keep;
        if (deferred_hour >= 0) {
            PendingSell ps;
            ps.cell = cell; ps.hour = deferred_hour; ps.units = base_units + pre_bonus;
            ps.forced = false;
            ps.from = mk; ps.st_from = st; ps.to = draw.mark();
            ps.g_before = g_before;
            ps.g_after  = g[cell];
            ps.t_before = t_before;
            ps.shed_for = shed_for_this;
            pending_sells.push_back(std::move(ps));
        }

        if (in_place) pt.yield_units = 0;
        else          pt = tile{ T_EMPTY, 1 };
        tmr_dirty = true;       // an animal's held units feed must_feed
        return true;
    };

    for (int cell = 0; cell < CELLS; ++cell) {
        const tile& pt = proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        if (!harvest_ready(pt, day)) continue;
        // A ripe melon is a FORCED harvest (melon_harvest_now). If it is still
        // standing here the forced pass could not route it, and a drawn
        // harvest would not route either -- so the melon triple is never drawn.
        if (melon_harvest_now(pt, day)) { ++d.stats.rej_capacity; continue; }
        hcands.push_back({ cell, pt.yield_units * product_price(tile_product(pt)),
                           /*forced=*/false, pt.yield_units });
    }
    std::stable_sort(hcands.begin(), hcands.end(),
                     [](const HCand& a, const HCand& b) { return a.value > b.value; });

    for (const HCand& hc : hcands) {
        if (hc.forced) bank_forced(hc);
        else           try_harvest(hc.cell);
    }

    // =====================================================================
    // 8. PLANTING  --  a count per type, placed by distance to the shed
    // =====================================================================
    // HARVEST, DIG, PLANT. A doomed ONGOING crop (it rots from tomorrow) whose
    // standing units are already being banked today -- almost always by the
    // forced harvest on the morning after its last production night. The
    // harvest stays; it just moves ahead of the dig as want_pre_harvest.
    auto pre_harvest_replant = [&](int cell) -> bool {
        if (!MACRO_REPLANT_DOOMED || !g[cell].want_harvest) return false;
        const tile& lt = live.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        return lt.type == T_PLANT && lt.crop >= 0 && CROPS[lt.crop].ongoing
            && plant_is_doomed(lt, day, turn);
    };
    // A cell can take a new occupant if it is owned, holds no live animal, and
    // no earlier pass kept its plant -- except a harvest that can ride ahead of
    // the dig.
    auto cell_open = [&](int cell) -> bool {
        const tile& pt = proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        if (!pt.bought || pt.animal >= 0) return false;
        if (g[cell].want_harvest && !pre_harvest_replant(cell)) return false;
        if (g[cell].want_watered || g[cell].want_fertilized) return false;
        return true;
    };
    // GROUND, or something a dig turns into ground today. A doomed plant is
    // ground only once there is nothing left UNBANKED on it: on `proj` a
    // harvest already routed today has zeroed its yield.
    auto clearable = [&](const tile& pt) {
        if (pt.type == T_EMPTY || pt.type == T_WEED) return true;
        if (MACRO_ALLOW_DIG_EMPTY_STRUCTURES &&
            (pt.type == T_COOP || pt.type == T_PASTURE)) return true;
        return MACRO_REPLANT_DOOMED && plant_is_doomed(pt, day, turn)
            && !harvest_ready(pt, day);
    };
    auto crop_in_horizon = [&](int c) {
        return day + CROPS[c].first_yield_day <= PLANT_HORIZON_DAY - 1;
    };
    // ---- THE MELON MASK -------------------------------------------------
    // Melon has no shop buying it: only the town centre drains one a day. So
    // the melon count is capped at the number of NEW melons whose walked
    // proceeds still beat their seed, after every melon already standing on
    // BOTH boards (and the own melon stock) has been sold first, each on its
    // straight-away harvest day. 0 masks melon out of the type slots.
    // Read off `live` and the opponent's live board: every melon standing at
    // hour 0 goes to market whether or not it is harvested today.
    const fcast::MelonMarket melon_mkt = fcast::melon_profit(
        game.players, player, game.market, game.unlocked_shops, day, turn,
        MACRO_PLANT_MAX);
    d.stats.melon_cap = melon_mkt.max_new;
    // Can `type` go on this (open) cell, board-wise? Money is checked apart.
    auto type_fits_cell = [&](int type, const tile& pt) -> bool {
        if (type < NUM_CROPS) return clearable(pt) && crop_in_horizon(type);
        const int a = type - NUM_CROPS;
        if (!animal_in_horizon(a, day)) return false;
        return clearable(pt) || pt.type == ANIMALS[a].structure;
    };
    auto type_affordable_now = [&](int type) -> bool {
        if (type < NUM_CROPS) return can_secure_seed(type);
        return can_secure(animal_item(type - NUM_CROPS));
    };

    // ---- FORCED PLANTING: hires on demand --------------------------------
    // What the seed or animal will cost if it has to be bought.
    auto occupant_cost = [&](int crop, int anim) -> double {
        if (crop >= 0)
            return (seed_need[crop] < live.seeds[crop]) ? 0.0 : (double)CROPS[crop].seed_cost;
        const int it = animal_item(anim);
        return (item_need[it] < kept[it]) ? 0.0 : unit_price(it);
    };

    // ---- PLACE ONE UNIT of `type` on `cell` -------------------------------
    // Forced: if the routes cannot absorb it, hire until they can (within the
    // spendable money and the crew cap). False if nothing makes it fit; the
    // cell is then left as it was.
    auto place_one = [&](int cell, int type) -> bool {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        tile& pt = proj.board[y][x];
        const int crop = (type < NUM_CROPS) ? type : -1;
        const int anim = (type < NUM_CROPS) ? -1 : type - NUM_CROPS;

        cell_goal trial = g[cell];
        trial.in_denominator = 1;
        // HARVEST, DIG, PLANT: the standing units are banked BEFORE the dig.
        // Left as want_harvest the op would be ordered after the new occupant
        // and land on an empty plant or a fresh animal.
        if (pre_harvest_replant(cell)) {
            trial.want_harvest     = 0;
            trial.want_pre_harvest = 1;
        }

        if (crop >= 0) {
            trial.target_type   = T_PLANT;
            trial.target_crop   = crop;
            trial.target_animal = -1;
            trial.replant = 1;
            trial.want_fed = trial.want_cared = trial.want_collect_fertilizer = 0;
            trial.want_fertilized = 0;       // the fertilize pass decides
            // A fresh plant carries consecutive_unwatered == 1: an unwatered
            // first night is weeds, so the water rides with the seed.
            trial.want_watered = MACRO_FORCE_WATER_ON_PLANT ? 1 : 0;
        } else {
            // build_task emits (DIG,) BUILD, PLACE as the cell requires.
            trial.target_type   = ANIMALS[anim].structure;
            trial.target_crop   = -1;
            trial.target_animal = anim;
            trial.want_watered = trial.want_fertilized = 0;
        }

        // Forced: the crew grows (up to the hard cap) until the unit fits.
        if (!commit_hiring(cell, trial, occupant_cost(crop, anim),
                           d.stats.plant_hires)) return false;

        if (crop >= 0) {
            secure_seed(crop);
            pt = tile{};
            pt.type   = T_PLANT;
            pt.bought = 1;
            pt.crop   = crop;
            pt.phase  = day;
            pt.consecutive_unwatered = 1;       // planting day counts as unwatered
            pt.yield_units = CROPS[crop].ongoing ? 0 : 1;
            pt.max_lifespan_step = CROPS[crop].ongoing
                ? -1 : (day + CROPS[crop].max_yield_day + 1) * TURNS_PER_DAY;
            if (trial.want_watered) {
                pt.watered_today = 1;
                ++d.stats.must_ops;
                ++d.stats.must_placed;
            }
        } else {
            secure(animal_item(anim));
            pt = tile{};
            pt.type   = ANIMALS[anim].structure;
            pt.bought = 1;
            pt.animal = anim;
            pt.phase  = day;
            tmr_dirty = true;                   // a forced feed tomorrow
        }
        return true;
    };

    // ---- 8a. WHERE: the distance rule ------------------------------------
    // Walking distance to the 2x2 shed block (4-way moves, so dx + dy), then
    // the Chebyshev ring, then the cell index for a stable order.
    auto shed_walk = [](int cell) -> int {
        const int x = cell % BOARD_SIZE, y = cell / BOARD_SIZE;
        const int h = BOARD_SIZE / 2;
        const int dx = (x < h - 1) ? (h - 1 - x) : (x > h ? x - h : 0);
        const int dy = (y < h - 1) ? (h - 1 - y) : (y > h ? y - h : 0);
        return dx + dy;
    };
    // ANIMAL ANCHORING. An animal may only go on a cell that TOUCHES the
    // shed (the 2x2 block itself or 4-adjacent to it, shed_walk <= 1) or is
    // 4-adjacent to a cell holding an animal -- on the PROJECTED board, so an
    // animal placed earlier today anchors its neighbours too. The herd grows
    // outward from the shed as one connected block.
    //
    // PREFERENCE: a free cell touching the shed ALWAYS beats a cell that is
    // only next to another animal, even when that animal itself touches the
    // shed. Within each tier, nearest the shed first (walk, then ring).
    auto touches_shed = [&](int cell) -> bool { return shed_walk(cell) <= 1; };
    auto next_to_animal = [&](int cell) -> bool {
        static const int DX[4] = { 0, 0, 1, -1 };
        static const int DY[4] = { -1, 1, 0, 0 };
        const int x = cell % BOARD_SIZE, y = cell / BOARD_SIZE;
        for (int dir = 0; dir < 4; ++dir) {
            const int nx = x + DX[dir], ny = y + DY[dir];
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
            if (proj.board[ny][nx].animal >= 0) return true;
        }
        return false;
    };
    auto animal_anchored = [&](int cell) -> bool {
        return touches_shed(cell) || next_to_animal(cell);
    };

    // The end of the board each type fills from:
    //   animals                    NEAREST the shed, anchored (see above)
    //   tomato, strawberry, melon  nearest of what is left
    //   wheat, carrot              the FAR end
    auto far_first = [](int k) -> bool { return k == WHEAT || k == CARROT; };

    // Every open cell type `k` can go on, in its placement order, and the
    // CEILING on how many of it today can take: min(reachable cells, what the
    // money funds, the head's top bin). Read on the board and budget as they
    // stand NOW, so a later slot only sees what the earlier ones left.
    struct Cand { int walk, ring, pen, cell; };
    auto type_options = [&](int k, std::vector<Cand>& cands, bool tally) -> int {
        const bool is_crop = k < NUM_CROPS;
        const bool far     = far_first(k);

        // Ties: an animal prefers a cell that already has its structure (no
        // build), a crop prefers one that does not (no coop dug up).
        cands.clear();
        cands.reserve(CELLS);
        for (int cell = 0; cell < CELLS; ++cell) {
            if (!cell_open(cell)) continue;
            const tile& pt = proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
            if (!type_fits_cell(k, pt)) continue;
            const bool is_struct = (pt.type == T_COOP || pt.type == T_PASTURE);
            const int  pen = is_crop
                ? (is_struct ? 1 : 0)
                : (pt.type == ANIMALS[k - NUM_CROPS].structure ? 0 : 1);
            cands.push_back({ shed_walk(cell),
                              shed_distance(cell % BOARD_SIZE, cell / BOARD_SIZE),
                              pen, cell });
        }
        std::sort(cands.begin(), cands.end(), [&](const Cand& a, const Cand& b) {
            if (a.walk != b.walk) return far ? a.walk > b.walk : a.walk < b.walk;
            if (a.ring != b.ring) return far ? a.ring > b.ring : a.ring < b.ring;
            if (a.pen  != b.pen)  return a.pen < b.pen;
            return a.cell < b.cell;
        });

        // How many of these cells could the type actually reach today? For a
        // crop, all of them. For an animal, only the ones connected to an
        // anchor through other fitting cells: each placement anchors its
        // neighbours, so that connected set is exactly what a full draw
        // could fill.
        int reachable = (int)cands.size();
        if (!is_crop) {
            bool fit[CELLS] = { false };
            for (const Cand& c : cands) fit[c.cell] = true;
            bool seen[CELLS] = { false };
            std::vector<int> stack;
            for (const Cand& c : cands)
                if (animal_anchored(c.cell)) { seen[c.cell] = true; stack.push_back(c.cell); }
            reachable = 0;
            while (!stack.empty()) {
                const int cell = stack.back(); stack.pop_back();
                ++reachable;
                const int x = cell % BOARD_SIZE, y = cell / BOARD_SIZE;
                const int nb[4][2] = { { x, y - 1 }, { x, y + 1 }, { x + 1, y }, { x - 1, y } };
                for (const auto& q : nb) {
                    if (q[0] < 0 || q[0] >= BOARD_SIZE || q[1] < 0 || q[1] >= BOARD_SIZE) continue;
                    const int n = q[1] * BOARD_SIZE + q[0];
                    if (fit[n] && !seen[n]) { seen[n] = true; stack.push_back(n); }
                }
            }
        }

        // What the money can fund right now, net of the feed reserve.
        int fundable = 0;
        if (is_crop) {
            const int on_hand = std::max(0, live.seeds[k] - seed_need[k]);
            const double avail = spendable();
            const int buy = avail > 0.0
                ? (int)std::floor(avail / (double)CROPS[k].seed_cost) : 0;
            fundable = on_hand + buy;
        } else {
            const int item = animal_item(k - NUM_CROPS);
            const int on_hand = std::max(0, kept[item] - item_need[item]);
            const double avail = spendable();
            const double per   = (double)ANIMALS[k - NUM_CROPS].cost
                               + unit_price(WHEAT);   // its forced feed tomorrow
            int buy = avail > 0.0 ? (int)std::floor(avail / per) : 0;
            // BUY room, not the harvest limit: commit_unit refuses an
            // A_BUY_ANIMAL on a full shed.
            buy = std::min(buy, std::max(0, room - mid_units));
            fundable = on_hand + buy;
        }

        const int type_max = is_crop ? MACRO_PLANT_MAX : MACRO_PLACE_MAX;
        int cap = std::min({ reachable, fundable, type_max });
        if (k == MELON && melon_mkt.max_new < cap) {
            if (tally && cap > 0 && melon_mkt.max_new <= 0) ++d.stats.rej_melon;
            cap = melon_mkt.max_new;
        }
        if (cap <= 0 && tally) {
            if (k == MELON && melon_mkt.max_new <= 0) {}   // counted above
            else if (!cands.empty()) ++d.stats.rej_budget;
            else                     ++d.stats.rej_illegal;
        }
        return std::max(0, cap);
    };

    // Put down up to `want` units of type k, in the type's placement order.
    auto place_type = [&](int k, int want, const std::vector<Cand>& cands) {
        if (k < NUM_CROPS) {
            for (const Cand& c : cands) {
                if (want <= 0) break;
                if (!type_affordable_now(k)) { ++d.stats.rej_budget; break; }
                if (!cell_open(c.cell)) continue;
                if (place_one(c.cell, k)) --want;
                else                      ++d.stats.rej_capacity;
            }
            return;
        }
        // Animals: every placement can anchor new neighbours, so the eligible
        // set is re-read after each one. Tier 0 is a cell touching the shed,
        // tier 1 a cell next to an animal; `cands` is sorted nearest-first, so
        // the first hit in the first non-empty tier is the pick. A cell that
        // failed to fit is not retried.
        bool tried[CELLS] = { false };
        while (want > 0) {
            if (!type_affordable_now(k)) { ++d.stats.rej_budget; break; }
            const Cand* best = nullptr;
            for (int tier = 0; tier < 2 && !best; ++tier) {
                for (const Cand& c : cands) {
                    if (tried[c.cell] || !cell_open(c.cell)) continue;
                    const bool ok = (tier == 0) ? touches_shed(c.cell)
                                                : next_to_animal(c.cell);
                    if (!ok) continue;
                    best = &c;
                    break;
                }
            }
            if (!best) break;
            tried[best->cell] = true;
            if (place_one(best->cell, k)) --want;
            else                          ++d.stats.rej_capacity;
        }
    };

    // ---- 8b. TYPE SLOTS, then HOW MANY -----------------------------------
    // Slot s offers Don't_Plant plus every type not yet chosen whose ceiling
    // is still > 0 after slots 0..s-1 were placed. Don't_Plant stops here.
    // If Don't_Plant is the ONLY option there is nothing to decide, so the
    // slot is not drawn (and nothing is recorded), and planting ends.
    bool type_used[NUM_COUNT_TYPES] = { false };

    // ---- DAY 0: the forced opening (MACRO_DAY0_FORCED) -------------------
    // Not a decision: nothing is drawn or recorded. 3 sheep, then 2 cows,
    // then 5 melons, each type placed by its normal rule (place_type), on the
    // board as the previous forced type left it. type_options only supplies
    // the placement order; its ceiling (and so the melon market mask) is
    // ignored. place_one hires to fit; a unit that cannot be afforded or
    // fitted is skipped. The type slots below then draw on what is left.
    if (day == 0) {
        for (int i = 0; i < MACRO_DAY0_FORCED_N; ++i) {
            const Day0Forced& f = MACRO_DAY0_FORCED[i];
            if (f.count <= 0 || f.type < 0 || f.type >= NUM_COUNT_TYPES) continue;
            if (MACRO_DAY0_FORCED_CLOSES_SLOTS) type_used[f.type] = true;
            std::vector<Cand> fc;
            type_options(f.type, fc, /*tally=*/false);
            place_type(f.type, f.count, fc);
        }
    }

    for (int slot = 0; slot < MACRO_TYPE_SLOTS; ++slot) {
        std::vector<Cand> opts[NUM_COUNT_TYPES];
        int caps[NUM_COUNT_TYPES] = { 0 };

        int idx[MACRO_TYPE_BINS];
        int type_of[MACRO_TYPE_BINS];
        int n = 0;
        idx[n] = macro_type_index(slot, MACRO_TYPE_NONE); type_of[n] = -1; ++n;
        for (int k = 0; k < NUM_COUNT_TYPES; ++k) {
            if (type_used[k]) continue;
            caps[k] = type_options(k, opts[k], /*tally=*/slot == 0);
            if (caps[k] <= 0) continue;
            idx[n] = macro_type_index(slot, 1 + k); type_of[n] = k; ++n;
        }
        if (n <= 1) break;

        // Each slot is its OWN head, with its own entropy target: slot 0 is
        // drawn on every planting day, slot 3 only after three types.
        const int pos = draw.pick(idx, n, head_plant_type(slot));
        const int k   = (pos >= 0) ? type_of[pos] : -1;
        if (k < 0) break;                         // Don't_Plant: stop the chain
        type_used[k] = true;

        // COUNT over the HALVED bins (0,1,2,4,6,...): every bin whose value
        // is below the ceiling, then the first bin at or above it, which
        // places exactly `cap` ("as many as fit"). Bin 0 is never offered:
        // the slot already said "some".
        const int cap  = caps[k];
        const int head = (k < NUM_CROPS) ? H_PLANT_N : H_PLACE_N;
        int cidx[MACRO_COUNT_BINS_MAX], cval[MACRO_COUNT_BINS_MAX], nc = 0;
        for (int b = 1; b < macro_count_bins(k); ++b) {
            const int v = macro_count_value(b);
            cidx[nc] = macro_count_index(k, b);
            cval[nc] = std::min(v, cap);
            ++nc;
            if (v >= cap) break;
        }
        const int cpos = draw.pick(cidx, nc, head);
        const int want = (cpos >= 0) ? cval[cpos] : cap;

        place_type(k, want, opts[k]);
    }

    // =====================================================================
    // 8c. FERTILIZE PASS  (drawn) -- AFTER planting
    // =====================================================================
    // Every plant on the projected board -- standing, or sown this morning --
    // where one fertilizer would add a unit (fertilize_pays: never a melon,
    // only on days the new cover adds, only with headroom, never after
    // MACRO_FERT_LAST_DAY). Anything else is never offered, so the head is
    // only drawn where both answers were defensible.
    //
    // Accepting drags in the water the payout needs if TODAY is a payout day
    // (A_FERTILIZE is emitted before A_WATER, which is the order the read
    // requires). Later payout days are banked by must_bank_water off the
    // cover this leaves on the tile.
    //
    // HIRES ON DEMAND: an accepted fertilize the routes cannot absorb hires
    // hands until it fits (commit_hiring), keeping the fertilizer's own price
    // in hand if it still has to be bought.
    for (int cell = 0; cell < CELLS; ++cell) {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        tile& pt = proj.board[y][x];

        if (!pt.bought || pt.type != T_PLANT || pt.crop < 0) continue;
        // The goal must still be THIS plant (a wiping harvest retargets it).
        if (!goal_is_plant(g[cell]) || g[cell].target_crop != pt.crop) {
            ++d.stats.rej_illegal; continue;
        }
        if (plant_is_doomed(pt, day, turn))  { ++d.stats.rej_illegal; continue; }
        if (!fertilize_pays(pt, day))        { ++d.stats.rej_illegal; continue; }
        if (!can_secure(FERTILIZER))         { ++d.stats.rej_budget;  continue; }

        if (draw.binary(MACRO_FERT_BASE, cell, H_FERTILIZE) != 1) continue;

        const cell_goal g_prev = g[cell];
        cell_goal trial = g[cell];
        trial.in_denominator = 1;
        trial.want_fertilized = 1;
        const bool add_water = fert_payout_day(pt, day) &&
                               !trial.want_watered && !pt.watered_today;
        if (add_water) trial.want_watered = 1;

        const double fert_extra =
            (item_need[FERTILIZER] < kept[FERTILIZER]) ? 0.0 : unit_price(FERTILIZER);
        if (!commit_hiring(cell, trial, fert_extra, d.stats.fert_hires)) {
            draw.retract(); ++d.stats.rej_capacity; continue;
        }
        // hires_to_fit keeps the fertilizer's price in hand, so this should
        // never fire; if it does, take the op back rather than buy on credit.
        if (!can_secure(FERTILIZER)) {
            commit(cell, g_prev);
            draw.retract(); ++d.stats.rej_budget; continue;
        }

        secure(FERTILIZER);
        pt.fertilized = std::max(pt.fertilized, day + 2);
        ++d.stats.fertilized;
        if (add_water) {
            pt.watered_today = 1;
            ++d.stats.must_ops;
            ++d.stats.must_placed;
        }
    }

    // =====================================================================
    // 9. FEED PASS  (drawn) -- with the FORCED CARE
    // =====================================================================
    // If the feed + care pair does not fit, hands are hired until it does
    // (commit_hiring). Only if no affordable crew fits the pair is the care
    // shed and the feed retried alone (hiring again); only if THAT fails is
    // the draw retracted.
    for (int cell = 0; cell < CELLS; ++cell) {
        const int y = cell / BOARD_SIZE, x = cell % BOARD_SIZE;
        tile& pt = proj.board[y][x];

        // ---- the care on a FORCED feed: the FEED HEAD decides it ---------
        // The forced pass fed this animal and left the care open. The feed
        // head is drawn again on this cell, its pair read as [No_Care, Care].
        // Drawn here, after planting and fertilizing, so a care never takes
        // an hour a planting wanted -- and hires like any drawn op.
        if (pt.animal >= 0 && pt.fed_today && g[cell].want_fed &&
            !g[cell].want_cared && must_care_after_feed(pt, day)) {
            if (draw.binary(MACRO_FEED_BASE, cell, H_FEED) != 1) continue;
            cell_goal trial = g[cell];
            trial.want_cared = 1;
            if (!commit_hiring(cell, trial, 0.0, d.stats.feed_hires)) {
                draw.retract(); ++d.stats.rej_capacity; continue;
            }
            pt.cared_today = 1;
            ++d.stats.cared;
            tmr_dirty = true;
            continue;
        }

        if (pt.animal < 0 || pt.fed_today) { ++d.stats.rej_illegal; continue; }
        if (!goal_is_animal(g[cell]))      { ++d.stats.rej_illegal; continue; }
        // An OPTIONAL feed spends wheat in the shed or spare money. It may
        // also spend reserve money when it only moves tomorrow's forced feed
        // of this same animal to today (fed today, it is off tomorrow's list).
        auto feed_wheat_ok = [&]() -> bool {
            if (item_need[WHEAT] < kept[WHEAT]) return true;
            if (room - mid_units <= 0) return false;
            const double p = unit_price(WHEAT);
            return spendable() >= p || (forced_tomorrow(pt) && budget >= p);
        };
        if (!feed_wheat_ok())              { ++d.stats.rej_budget;  continue; }

        if (draw.binary(MACRO_FEED_BASE, cell, H_FEED) != 1) continue;

        const bool with_care = must_care_after_feed(pt, day);

        const cell_goal g_prev = g[cell];
        cell_goal trial = g[cell];
        trial.in_denominator = 1;
        trial.want_fed = 1;
        if (with_care) trial.want_cared = 1;

        const double wheat_extra =
            (item_need[WHEAT] < kept[WHEAT]) ? 0.0 : unit_price(WHEAT);
        bool cared_now = with_care;
        bool ok = commit_hiring(cell, trial, wheat_extra, d.stats.feed_hires);
        if (!ok && with_care) {
            trial.want_cared = 0;
            cared_now = false;
            ok = commit_hiring(cell, trial, wheat_extra, d.stats.feed_hires);
            if (ok) { ++d.stats.must_ops; ++d.stats.must_dropped; }
        }
        if (!ok) { draw.retract(); ++d.stats.rej_capacity; continue; }
        // Same guard as the fertilize pass: the wheat must still be there.
        if (!feed_wheat_ok()) {
            commit(cell, g_prev);
            if (cared_now != with_care) { --d.stats.must_ops; --d.stats.must_dropped; }
            draw.retract(); ++d.stats.rej_budget; continue;
        }

        secure(WHEAT);
        pt.fed_today = 1;
        tmr_dirty = true;
        if (cared_now) {
            pt.cared_today = 1;
            ++d.stats.cared;
            ++d.stats.must_ops;
            ++d.stats.must_placed;
        }
    }

    // =====================================================================
    // 9b. ESCAPE HARVEST  --  forced, no draw
    // =====================================================================
    // An animal that is still unfed tonight after every pass (the feed head
    // declined it after the feed cutoff, or there was no wheat) escapes, and
    // the units it holds go with it. cell_musts no longer forces an animal's
    // harvest on its own, so bank them here, kept for the nightly sweep. A
    // cell the harvest pass already emptied reads 0 units on `proj`.
    for (int cell = 0; cell < CELLS; ++cell) {
        tile& pt = proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        if (pt.animal < 0 || pt.yield_units <= 0 || !pt.bought) continue;
        if (!animal_escapes_tonight(pt, day) || g[cell].want_harvest) continue;
        if (!goal_is_animal(g[cell])) continue;
        const int units = pt.yield_units;
        cell_goal trial = harvest_goal(cell, /*in_place=*/true, false);
        if (!commit_hiring(cell, trial, 0.0, d.stats.collect_hires)) {
            ++d.stats.must_dropped;
            continue;
        }
        night += units;
        pt.yield_units = 0;
        ++d.stats.harvest_keep;
        ++d.stats.escape_harvests;
        tmr_dirty = true;
    }

    // =====================================================================
    // 9c. COLLECT  (drawn)  --  [No_Collect, Collect]
    // =====================================================================
    // Every animal with fertilizer waiting draws the collect pair, once the
    // harvests, plantings and feeds have taken the hours and the shed. The
    // unit is kept for the nightly sweep, so it needs a night slot: a full
    // night is not offered. Accepting one hires to fit, like every drawn op.
    for (int cell = 0; cell < CELLS; ++cell) {
        tile& pt = proj.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        if (!must_collect(pt))              continue;
        if (!goal_is_animal(g[cell]))       { ++d.stats.rej_illegal; continue; }
        if (night >= SHED_CAPACITY)         { ++d.stats.rej_budget;  continue; }
        if (draw.binary(MACRO_COLLECT_BASE, cell, H_COLLECT) != 1) continue;
        cell_goal trial = g[cell];
        trial.in_denominator = 1;
        trial.want_collect_fertilizer = 1;
        if (!commit_hiring(cell, trial, 0.0, d.stats.collect_hires)) {
            draw.retract(); ++d.stats.rej_capacity; continue;
        }
        pt.fertilizer_available = 0;
        ++night;
        collect_cells.push_back(cell);
        ++d.stats.collected;
    }

    // =====================================================================
    // 10. PENDING SALES  --  the mid-day drops, in whatever hours are left
    // =====================================================================
    // Each pending sale is routed at its drawn hour, hiring if it must. If it
    // no longer fits -- planting, fertilize and feed took the hours, and the
    // crew cap or the money leaves nothing to hire -- the sell-time head is
    // REMASKED on the plan as it stands and drawn again:
    //
    //   hours        the ones that still route (exact_sell_bins)
    //   ST_KEEP      always: the harvest is already routed as a keep
    //   ST_NO_HARVEST only for an OPTIONAL harvest whose cell goal is
    //                unchanged since the harvest pass (nothing -- a replant,
    //                a fertilize, a feed -- was built on it) and whose release
    //                routes. Taking it UNDOES the harvest.
    //
    // The first hour's draw is erased from the record. If the redraw had only
    // one option, the harvest head's Sell is erased with it: that harvest
    // ended up a plain keep that nobody chose.
    auto sold_mid = [&](const PendingSell& ps) {
        night     -= ps.units;
        mid_units += ps.units;
        --d.stats.harvest_keep;
        ++d.stats.harvest_sell;
    };
    // Would putting the cell back to its pre-harvest goal route? Snapshot test.
    auto can_undo = [&](const PendingSell& ps) -> bool {
        const MacroPlanner snap = d.planner;
        const cell_goal gs = g[ps.cell];
        const bool ok = commit(ps.cell, ps.g_before);
        d.planner = snap;
        g[ps.cell] = gs;
        return ok;
    };
    // ST_NO_HARVEST: the cell as it was before the harvest pass touched it.
    auto undo_harvest = [&](const PendingSell& ps) {
        commit(ps.cell, ps.g_before);          // can_undo just routed it
        proj.board[ps.cell / BOARD_SIZE][ps.cell % BOARD_SIZE] = ps.t_before;
        night -= ps.units;
        --d.stats.harvest_keep;
        tmr_dirty = true;
        restore_collects(ps.shed_for);
    };
    {
        struct Erase { MacroDraw::Mark from, to; };
        std::vector<Erase> erases;             // ascending, like pending_sells
        for (size_t i = 0; i < pending_sells.size(); ++i) {
            const PendingSell& ps = pending_sells[i];
            const bool room_ok = room - mid_units >= ps.units;
            cell_goal trial = g[ps.cell];
            if (room_ok) {
                trial.sell_hour = ps.hour;
                if (commit_hiring(ps.cell, trial, 0.0, d.stats.harvest_hires)) {
                    sold_mid(ps);
                    continue;
                }
            }

            // ---- the drawn hour no longer fits: REMASK -------------------
            trial.sell_hour = -1;
            const bool allow_no = !ps.forced && goal_same(g[ps.cell], ps.g_after)
                                && can_undo(ps);
            bool rec = false;
            const SellPick rp = redraw_sale(ps.cell, trial, room_ok,
                                            allow_no, /*keep=*/true, rec);
            erases.push_back({ rec ? ps.st_from : ps.from, ps.to });
            if (rp.kind == SK_HOUR) {
                sold_mid(ps);
                ++d.stats.sell_redrawn;
            } else if (rp.kind == SK_NO_HARVEST) {
                undo_harvest(ps);
                ++d.stats.sell_skip;
            } else if (rec) {
                ++d.stats.sell_keep;
            } else {
                ++d.stats.sell_fallback;
            }
        }
        // Later ranges first, or the earlier marks go stale. The redraws were
        // appended after every pending mark, so they are only shifted.
        for (size_t i = erases.size(); i-- > 0; )
            draw.erase(erases[i].from, erases[i].to);
    }

    // =====================================================================
    // ROUTE, ORDER, SCRIPT, SELL
    // =====================================================================
    d.planner.optimize();

    std::vector<move> orders;
    for (int k = 0; k < NUM_PRODUCTS; ++k)
        if (sold[k] > 0) orders.push_back(order_sell(k, sold[k]));
    // BUGFIX: ORDER = PRIORITY. The referee spends in list order, and when
    // the realised money comes in under the projection (the opponent trades
    // in lockstep on the same market, so shared products fetch less than
    // quoted) the entries at the END are the ones that fail. So:
    //   1. the hire -- a hand that fails to spawn loses its WHOLE route (every
    //      move addressed to it is dropped), and hands cost 1, 1, 2, 3, ...
    //   2. wheat -- today's forced feeds. It used to come after the seeds.
    //   3. everything discretionary.
    // ONE entry for every hire of the day, whichever pass made it, so the
    // order hours never move.
    if (d.stats.hired > 0)
        orders.push_back({ MARKET_ACTOR, A_HIRE, NO_ITEM, d.stats.hired });
    if (item_bought[WHEAT] > 0)
        orders.push_back(order_buy_product(WHEAT, item_bought[WHEAT]));
    // DAY 0: the forced opening's buys are the first discretionary entries,
    // in the opening's order (sheep, cows, melons), so a short market fails
    // the slots' purchases, not these.
    bool seed_done[NUM_CROPS] = { false };
    bool anim_done[NUM_ANIMALS] = { false };
    if (day == 0) {
        for (int i = 0; i < MACRO_DAY0_FORCED_N; ++i) {
            const int t = MACRO_DAY0_FORCED[i].type;
            if (MACRO_DAY0_FORCED[i].count <= 0 || t < 0 || t >= NUM_COUNT_TYPES) continue;
            if (t < NUM_CROPS) {
                if (seed_done[t]) continue;
                seed_done[t] = true;
                const int want = seed_need[t] - live.seeds[t];
                if (want > 0) orders.push_back(order_buy_seed(t, want));
            } else {
                const int a = t - NUM_CROPS;
                if (anim_done[a]) continue;
                anim_done[a] = true;
                if (item_bought[animal_item(a)] > 0)
                    orders.push_back(order_buy_animal(a, item_bought[animal_item(a)]));
            }
        }
    }
    if (d.stats.bought_land) orders.push_back(order_buy_land());
    for (int c = 0; c < NUM_CROPS; ++c) {
        if (seed_done[c]) continue;
        const int want = seed_need[c] - live.seeds[c];
        if (want > 0) orders.push_back(order_buy_seed(c, want));
    }
    if (item_bought[FERTILIZER] > 0)
        orders.push_back(order_buy_product(FERTILIZER, item_bought[FERTILIZER]));
    for (int a = 0; a < NUM_ANIMALS; ++a)
        if (!anim_done[a] && item_bought[animal_item(a)] > 0)
            orders.push_back(order_buy_animal(a, item_bought[animal_item(a)]));

    // Chunk by REFEREE entries: a hire of N becomes N single-hand entries, in
    // place, so the list's priority order (sells, hires, wheat, ...) holds.
    referee_expand(orders);
    d.order_chunks.clear();
    for (size_t i = 0; i < orders.size(); i += MAX_MARKET_ORDERS_PER_TURN) {
        if ((int)d.order_chunks.size() >= MACRO_MAX_ORDER_HOURS) break;
        const size_t e = std::min(orders.size(), i + MAX_MARKET_ORDERS_PER_TURN);
        d.order_chunks.emplace_back(orders.begin() + i, orders.begin() + e);
    }
    if (d.order_chunks.empty()) d.order_chunks.emplace_back();
    d.n_order_hours = (int)d.order_chunks.size();
    // The routes were priced on est_order_hours of mornings. More chunks than
    // that and a hand would be addressed before it exists, or a plant before
    // its seed arrives: ask for a re-plan with the real number.
    if (d.n_order_hours > est_order_hours) *needed_hours = d.n_order_hours;

    // Should be impossible now (see MacroPlanner::detach). Counted, never
    // silently flattened into a hand that stops mid-route.
    d.stats.broken_routes = d.planner.broken_routes();
    d.planner.flatten(d.n_order_hours);

    // ---- THE TIMED SELLS ------------------------------------------------
    // One entry per (hour, product), carrying the EXACT units the script
    // drops for it (MacroPlanner::harvest_units, decay included). Every one of
    // those drops lands at or before its hour, so the sell never reaches into
    // stock the morning sell head chose to hold.
    {
        int per[TURNS_PER_DAY][NUM_PRODUCTS] = {};
        for (const MacroPlanner::SellRec& r : d.planner.sells())
            if (r.hour >= 0 && r.hour < TURNS_PER_DAY &&
                r.product >= 0 && r.product < NUM_PRODUCTS && r.units > 0)
                per[r.hour][r.product] += r.units;
        for (int h = 0; h < TURNS_PER_DAY; ++h)
            for (int k = 0; k < NUM_PRODUCTS; ++k)
                if (per[h][k] > 0) {
                    d.timed_orders[h].push_back(order_sell(k, per[h][k]));
                    d.stats.sell_units += per[h][k];
                }
    }
    d.stats.busy_hands = d.planner.busy_hands();
    d.stats.night_ledger = night;

    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x)
            d.grid[y][x] = g[y * BOARD_SIZE + x];

    d.stats.drawn = draw.n_recorded;
    d.stats.accepted_cells = d.planner.placed();
    d.stats.accepted_ops   = d.planner.total_ops();
}

// ===========================================================================
// THE LAST DAY  --  no network, no draws, no decisions
// ===========================================================================
// check_winner reads `money`. Nothing else on the board outlives tonight, so
// every unit that can reach the shed and be sold today is worth exactly its
// price, and everything else is worth zero.
//
//   1. apply_turn runs apply_farmer_moves BEFORE process_market, so an A_DROP
//      at hour h is in the shed when hour h's SELL order is quoted.
//   2. drop_inventories_to_shed runs AFTER the last market of the day, so
//      every trip ends at the shed with an explicit A_DROP.
//   3. Hands hired at hour 0 exist from hour 1; hand 0 works from hour 0.
//
// The crew is as small as the work allows, routes are built by marginal
// insertion cost, and decay is modelled so a rotting stack is visited first.
// ===========================================================================
// THE PUBLIC ENTRY. Plans the day; if the day's hires, counted the way the
// referee counts them, pushed the orders past the hours its routes were
// priced on, the day is planned again with more order hours. Fewer route
// hours can call for MORE hands, so each round asks for at least one more
// morning; there are only MACRO_MAX_ORDER_HOURS. Anything the first attempt
// recorded is rolled back, so a re-planned day leaves one set of samples.
void macro_plan_day(const float* out, simulation& game, int player, MacroDay& d,
                    std::vector<std::vector<drone_action>>* actions,
                    std::vector<double>* log_probs,
                    std::vector<int>* head_ids,
                    float temperature, bool greedy, int print)
{
    const size_t na = actions   ? actions->size()   : 0;
    const size_t nl = log_probs ? log_probs->size() : 0;
    const size_t nh = head_ids  ? head_ids->size()  : 0;

    int need = 0;
    macro_plan_day_impl(out, game, player, d, actions, log_probs, head_ids,
                        temperature, greedy, print, 0, &need);
    int hours = 0, rounds = 0;
    while (need > 0 && hours < MACRO_MAX_ORDER_HOURS) {
        hours = std::min(MACRO_MAX_ORDER_HOURS, std::max(need, hours + 1));
        if (actions)   actions->resize(na);
        if (log_probs) log_probs->resize(nl);
        if (head_ids)  head_ids->resize(nh);
        need = 0;
        macro_plan_day_impl(out, game, player, d, actions, log_probs, head_ids,
                            temperature, greedy, print, hours, &need);
        ++rounds;
    }
    d.stats.replanned = rounds;
    // Nine sells + land + fifteen hires + ten buys is 35 entries, which fits
    // MACRO_MAX_ORDER_HOURS (4) mornings, so this stays 0 -- counted, not hidden.
    if (need > 0) d.stats.replan_overflow = 1;
}

void macro_liquidate_day(simulation& game, int player, MacroDay& d)
{
    d.clear();

    const player_state& live = game.players[player];
    const int day   = game.day();
    const int turn0 = game.turn;              // the turn index of hour 0 today

    // MacroDay objects are reused day to day and MacroDay::emit always calls
    // planner.emit, so yesterday's script has to be cleared or it replays.
    {
        player_state none = live;
        none.farmers.clear();
        none.carried.clear();
        d.planner.init(none, day, 0);
    }

    const MacroGeo& geo = macro_geo((int)live.unlocked_quadrants.size());

    // Nearest shed-access tile and the steps to it.
    auto shed_exit = [&](int cell) {
        int best = -1, bd = MacroGeo::INF;
        for (const ipos& tl : shed_access_tiles()) {
            const int c2 = tl.y * BOARD_SIZE + tl.x;
            const int dd = geo.D(cell, c2);
            if (dd < bd) { bd = dd; best = c2; }
        }
        return std::pair<int, int>{ best, bd };
    };

    struct Job {
        int cell = 0;
        int ops[3] = { A_PASS, A_PASS, A_PASS };
        int n_ops = 0;
        int item = -1;
        int base_units  = 0;      // tile stack at hour 0, before any water
        int water_bonus = 0;      // what A_WATER would add, 0 if there is no water op
        int max_yield   = 0;
        int mls  = -1;            // max_lifespan_step, or -1 for a cell that cannot decay
        int fert = 0;
        int back = 0;             // steps from this cell to the nearest shed tile
        double unit_value = 0.0;  // spot price of this cell's product
        double fert_value = 0.0;
        bool taken = false;
    };
    std::vector<Job> jobs;
    jobs.reserve(64);

    for (int cell = 0; cell < CELLS; ++cell) {
        const tile& t = live.board[cell / BOARD_SIZE][cell % BOARD_SIZE];
        if (!t.bought) continue;

        Job j;
        j.cell = cell;
        j.base_units = t.yield_units;
        const bool ready = harvest_ready(t, day);

        // The yield water, guarded on the harvest being LEGAL afterwards.
        bool water = false;
        if (t.type == T_PLANT && t.crop >= 0 && water_yields_now(t, day) &&
            day - t.phase >= CROPS[t.crop].first_yield_day) {
            water = true;
            j.water_bonus = (t.fertilized >= day) ? 2 : 1;
            j.max_yield   = CROPS[t.crop].max_yield;
        }

        if (water) j.ops[j.n_ops++] = A_WATER;
        if ((j.base_units > 0 && ready) || water) {
            j.ops[j.n_ops++] = A_HARVEST;
            j.item = tile_product(t);
        }
        if (t.animal >= 0 && t.fertilizer_available) {
            j.ops[j.n_ops++] = A_COLLECT_FERTILIZER;
            j.fert = 1;
        }
        if (j.n_ops == 0) continue;

        const auto ex = shed_exit(cell);
        if (ex.first < 0) continue;
        j.back = ex.second;

        if (t.type == T_PLANT) j.mls = t.max_lifespan_step;
        if (j.item >= 0) j.unit_value = (double)game.market.prices[j.item];
        if (j.fert)      j.fert_value = (double)game.market.prices[FERTILIZER];

        jobs.push_back(j);
    }

    // ---- DECAY, MODELLED -------------------------------------------------
    auto decay_lost = [&](const Job& j, int hour) -> int {
        if (j.mls < 0 || j.base_units <= 0) return 0;
        const int lo = std::max(j.mls, turn0);
        const int hi = turn0 + std::max(0, hour - 1); // Exclusive of the harvest turn
        const int span = hi - lo;
        if (span <= 0) return 0;
        const int first = ((lo - j.mls) % 2 == 0) ? 0 : 1;
        if (first >= span) return 0;
        return (span - first + 1) / 2;
    };
    auto units_at = [&](const Job& j, int hour) -> int {
        int u = std::max(0, j.base_units - decay_lost(j, hour));
        if (j.water_bonus > 0 && (u > 0 || j.base_units == 0))
            u = std::min(j.max_yield, u + j.water_bonus);
        return u;
    };
    auto value_at = [&](const Job& j, int hour) -> double {
        return (double)units_at(j, hour) * j.unit_value + j.fert_value;
    };
    auto is_rotting = [&](const Job& j) {
        return j.mls >= 0 && j.base_units > 0 && j.mls < turn0 + TURNS_PER_DAY;
    };

    // Livestock is not sellable, so it holds its slots all day.
    int reserved = 0;
    for (int a = 0; a < NUM_ANIMALS; ++a) reserved += live.shed[animal_item(a)];
    const int room = std::max(0, SHED_CAPACITY - reserved);

    // ---- routing ---------------------------------------------------------
    static const int DX[4] = { 0, 0, 1, -1 };
    static const int DY[4] = { -1, 1, 0, 0 };

    auto walk = [&](std::vector<move>& s, int hi, int from, int to) -> int {
        int cur = from;
        while (cur != to) {
            const int dir = geo.DIR(cur, to);
            if (dir < 0) break;                     // unreachable: stop here
            s.push_back(act(hi, dir));              // dir == A_MOVE_*
            cur = (cur / BOARD_SIZE + DY[dir]) * BOARD_SIZE
                + (cur % BOARD_SIZE + DX[dir]);
        }
        return cur;
    };
    auto back_from = [&](int cell) { return shed_exit(cell).second; };

    std::vector<int> drop_units(TURNS_PER_DAY, 0);
    std::vector<std::array<int, NUM_PRODUCTS>> sell_at(TURNS_PER_DAY);
    for (auto& a : sell_at) a.fill(0);

    std::vector<ipos> crew = live.farmers;          // exactly one this morning
    std::vector<std::vector<move>> script;
    std::vector<int> start_hour;

    // THE CREW CAP IS 12 HANDS TOTAL here: past about a dozen the extra hands
    // walk further for less and compete for the same per-hour shed room.
    const int max_crew = std::min(12, MAX_UNITS);
    int hires = 0;
    double hire_spend = 0.0;

    auto work_left = [&]() {
        for (const Job& j : jobs) if (!j.taken) return true;
        return false;
    };

    std::vector<int> trip_jobs;                     // taken on the current trip

    // THE SHED IS 100 UNITS PER HOUR, NOT PER DAY. A drop and the sell of the
    // same units happen in one turn (apply_farmer_moves, then process_market),
    // so the shed empties every hour -- but whatever one hour's drops bring in
    // beyond `room` is discarded by dump_inventory before the sell can see it.
    // A trip is only worth taking if SOME hour from its homecoming to the end
    // of the day can still absorb its whole load. Checked while the trip is
    // being filled, so a job this hand cannot bring home stays on the board
    // for another hand whose drop lands on a quieter hour.
    auto drop_fits = [&](int home_hour, int load) {
        for (int h = std::max(home_hour, 1); h < TURNS_PER_DAY; ++h)
            if (drop_units[h] + load <= room) return true;
        return false;
    };

    for (int hi = 0; ; ++hi) {
        if (hi > 0) {
            if (!work_left()) break;
            if ((int)crew.size() >= max_crew) break;
            const double c = FARM_HAND_COST_MULT
                           * (double)macro_fib(live.hires_today + hires);
            if (hire_spend + c > live.money) break;
            hire_spend += c;
            ++hires;
            crew.push_back(macro_spawn(crew));
        }

        // THE REFEREE: the hires go out first, one hand per entry, ten a
        // turn, so hand hi (hire number hi-1) spawns after hour (hi-1)/10's
        // market and walks from the hour after.
        const int sh = (hi > 0) ? 1 + (hi - 1) / MAX_MARKET_ORDERS_PER_TURN : 1;
        const int budget = TURNS_PER_DAY - sh;
        std::vector<move> s;
        int pos = crew[hi].y * BOARD_SIZE + crew[hi].x;
        bool any_trip = false;

        while (true) {                              // one iteration == one trip
            int carried = 0;
            std::array<int, NUM_ITEMS> bag{};
            trip_jobs.clear();
            const size_t trip_start = s.size();

            while (true) {                          // fill the hands
                int best = -1;
                double best_score = 0.0;
                const int here_back = back_from(pos);

                for (size_t k = 0; k < jobs.size(); ++k) {
                    const Job& j = jobs[k];
                    if (j.taken) continue;
                    const int d1 = geo.D(pos, j.cell);
                    if (d1 >= MacroGeo::INF) continue;
                    if ((int)s.size() + d1 + j.n_ops + j.back + 1 > budget)
                        continue;

                    const int arrive = sh + (int)s.size() + d1 + j.n_ops;
                    const int u = units_at(j, arrive);
                    if (u <= 0 && !j.fert) continue;
                    if (carried + u + j.fert > room) continue;
                    // Home straight after this job: is there an hour left that
                    // takes the whole load?
                    if (!drop_fits(arrive + j.back, carried + u + j.fert)) continue;

                    // THE SCORE IS A MARGINAL INSERTION COST.
                    const int detour = d1 + j.back - here_back;
                    const double cost = (double)std::max(1, detour + j.n_ops);
                    double score = value_at(j, arrive) / cost;
                    if (is_rotting(j)) score *= 1.5;

                    if (best < 0 || score > best_score) {
                        best = (int)k; best_score = score;
                    }
                }
                if (best < 0) break;

                Job& j = jobs[best];
                const int arrive = sh + (int)s.size() + geo.D(pos, j.cell) + j.n_ops;
                pos = walk(s, hi, pos, j.cell);
                if (pos != j.cell) break;           // walk gave up: leave it
                for (int o = 0; o < j.n_ops; ++o) s.push_back(act(hi, j.ops[o]));

                const int u = units_at(j, arrive);
                if (u > 0 && j.item >= 0) bag[j.item] += u;
                if (j.fert)               bag[FERTILIZER] += 1;
                carried += u + j.fert;
                j.taken = true;
                trip_jobs.push_back(best);
            }

            if (carried <= 0) break;

            // ---- home, and drop on an hour the shed can take -------------
            const auto ex = shed_exit(pos);
            pos = walk(s, hi, pos, ex.first);
            int hour = sh + (int)s.size();
            while (hour + 1 < TURNS_PER_DAY &&
                   drop_units[hour] + carried > room &&
                   (int)s.size() + 1 < budget) {
                s.push_back(act(hi, A_PASS));
                ++hour;
            }
            // Out of hours, or out of room at every hour that is left: never
            // drop into a full hour. drop_fits makes this rare -- the waiting
            // above can only run out of script budget, not of hours -- and the
            // jobs go back on the board rather than into the overflow bin.
            if (hour >= TURNS_PER_DAY || drop_units[hour] + carried > room) {
                for (const int k : trip_jobs) jobs[k].taken = false;
                s.resize(trip_start);
                break;
            }
            s.push_back(act_drop_all(hi));
            drop_units[hour] += carried;
            for (int k = 0; k < NUM_PRODUCTS; ++k)
                if (bag[k] > 0) sell_at[hour][k] += bag[k];
            any_trip = true;

            if ((int)s.size() >= budget) break;
        }

        if (!any_trip) {
            if (hi > 0) { --hires; crew.pop_back(); }
            break;
        }
        script.push_back(std::move(s));
        start_hour.push_back(sh);
        if (!work_left()) break;
    }

    // ---- the day, hour by hour -------------------------------------------
    d.order_chunks.assign(TURNS_PER_DAY, {});

    if (hires > 0)
        d.order_chunks[0].push_back({ MARKET_ACTOR, A_HIRE, NO_ITEM, hires });
    for (int k = 0; k < NUM_PRODUCTS; ++k) {
        const int n = live.shed[k] + sell_at[0][k];
        if (n > 0) {
            d.order_chunks[0].push_back(order_sell(k, n));
            d.stats.raised += sell_proceeds(k, n, game.market.inventory[k]);
        }
    }
    for (int h = 1; h < TURNS_PER_DAY; ++h)
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            if (sell_at[h][k] > 0) d.order_chunks[h].push_back(order_sell(k, sell_at[h][k]));

    for (size_t hh = 0; hh < script.size(); ++hh)
        for (size_t i = 0; i < script[hh].size(); ++i) {
            const size_t hour = (size_t)start_hour[hh] + i;
            if (hour < (size_t)TURNS_PER_DAY)
                d.order_chunks[hour].push_back(script[hh][i]);
        }

    // THE REFEREE: one hand per HIRE entry, at most ten entries a turn. The
    // hire goes first (above), so its hands fill hour 0 and then hour 1 --
    // matching the start hours the scripts were built with -- and every hour
    // that ends up over ten entries carries its tail to the front of the next.
    // NOTE: the liquidation's order_chunks also carry every hand's SCRIPT
    // steps (farmer_idx >= 0). Those are not market entries: they neither
    // count against the cap nor ever move. Only MARKET_ACTOR entries do.
    std::vector<move> carry;                       // market entries owed to hour h
    for (int h = 0; h < TURNS_PER_DAY; ++h) {
        std::vector<move> units, market = carry;
        carry.clear();
        for (const move& m : d.order_chunks[h]) {
            if (m.farmer_idx != MARKET_ACTOR) { units.push_back(m); continue; }
            // A carried SELL folds into this hour's SELL of the same product
            // (the units sit in the shed either way), so the backlog shrinks
            // instead of snowballing into hour 23, where it would be lost.
            bool folded = false;
            if (m.type == A_SELL)
                for (move& c : market)
                    if (c.type == A_SELL && c.item == m.item) {
                        c.count += m.count;
                        folded = true;
                        break;
                    }
            if (!folded) market.push_back(m);
        }
        referee_expand(market);
        if ((int)market.size() > MAX_MARKET_ORDERS_PER_TURN) {
            carry.assign(market.begin() + MAX_MARKET_ORDERS_PER_TURN, market.end());
            market.resize(MAX_MARKET_ORDERS_PER_TURN);
        }
        std::vector<move>& ch = d.order_chunks[h];
        ch = market;
        ch.insert(ch.end(), units.begin(), units.end());
    }
    d.stats.liq_orders_lost += (int)carry.size();
    d.n_order_hours = TURNS_PER_DAY;

    d.stats.drawn  = 0;
    d.stats.hired  = hires;
    for (const Job& j : jobs) {
        if (!j.taken) { ++d.stats.must_dropped; continue; }
        ++d.stats.accepted_cells;
        d.stats.accepted_ops += j.n_ops;
        d.stats.must_ops     += j.n_ops;
        d.stats.must_placed  += j.n_ops;
    }
}


// Seeds THIS translation unit's `gen` -- the one multinomial() draws from.
// (`gen` is `static` in the header, so every .cpp has its own; seeding it
// from another file would seed a different engine.)
void macro_seed_rng(uint32_t seed) { gen.seed(seed); }

// ===========================================================================
// TRAINING-SIDE REGROUPING
// ===========================================================================
torch::Tensor policy_to_jointlog(
    torch::Tensor output,
    torch::Tensor actions_indices,
    torch::Tensor output_indices,
    torch::Tensor valid_action_indices,
    torch::Tensor chosen_move_indices,
    int num_groups_in, torch::Tensor& entropy,
    int batch_size)
{
    (void)batch_size;
    auto selected_logits = output.index({output_indices, actions_indices});
    const int64_t num_groups = num_groups_in;

    // --- stable softmax per group ---
    auto neg_inf = -std::numeric_limits<float>::infinity();
    auto max_init = torch::full({num_groups}, neg_inf, selected_logits.options());
    auto max_per_group = max_init.scatter_reduce(
        0, valid_action_indices, selected_logits, "amax");
    auto max_per_element = max_per_group.index_select(0, valid_action_indices);
    auto values_stable = selected_logits - max_per_element;
    auto exp_values = torch::exp(values_stable);

    auto sum_per_group = torch::zeros({num_groups}, selected_logits.options())
                            .scatter_add(0, valid_action_indices, exp_values);
    auto denom = sum_per_group.index_select(0, valid_action_indices);

    auto log_softmax = values_stable - torch::log(denom + 1e-8);
    auto softmax = torch::exp(log_softmax);

    auto chosen_log_probs = log_softmax.index_select(0, chosen_move_indices);

    // RAW ENTROPY per set, in nats: not divided by log(n), so a set with more
    // candidates can carry more entropy than a small one.
    auto entropy_per_entry = -softmax * log_softmax;
    entropy = torch::zeros({num_groups}, entropy_per_entry.options());
    entropy.scatter_add_(0, valid_action_indices, entropy_per_entry);

    return chosen_log_probs;
}