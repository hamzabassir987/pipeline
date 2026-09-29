#ifndef MACRO_PLANNER_HPP
#define MACRO_PLANNER_HPP
// ===========================================================================
// MACRO_PLANNER -- the executor.
//
//  1. INCREMENTAL CAPACITY. The filter asks "does this one extra op still fit"
//     hundreds of times per day, so the plan is built by cheapest insertion
//     against LIVE routes.
//
//  2. NO RUIN PHASE. The filter has already accepted every cell and recorded a
//     gradient saying so; optimize() is pure descent and never drops a task.
//
//  3. GEOMETRY IS CACHED. Four reachable boards, BFS once per process.
//
//  4. MID-DAY DROPS ARE DERIVED, NOT PLANNED. A route is still a sequence of
//     cells; the shed visits an HB_SELL harvest needs are inserted by the
//     route walker itself, so every move operator stays a sequence operator.
//     See the MacroPlanner header.
//
// A hand picks up every input for its whole route at the shed, walks it, and
// keeps every HB_KEEP harvest and every collected fertilizer in hand for the
// nightly sweep. HB_SELL harvests go back to the shed by their sell hour and
// are sold there and then. (The last day is scripted separately by
// macro_liquidate_day.)
// ===========================================================================
#include "game.hpp"
#include <climits>
#include <vector>
#include <array>
#include <queue>
#include <algorithm>
#include <cstdio>

// ---------------------------------------------------------------------------
// Cached board geometry
// ---------------------------------------------------------------------------
struct MacroGeo {
    static constexpr int N   = CELLS;
    static constexpr int INF = 1 << 28;
    std::vector<int> dist;   // N*N step counts
    std::vector<int> dir;    // N*N first step from a to b, or -1

    int D(int a, int b)   const { return dist[(size_t)a * N + b]; }
    int DIR(int a, int b) const { return dir[(size_t)a * N + b]; }
};

inline bool macro_owned(int n_unlocked, int x, int y) {
    const int q = quadrant_of(x, y);
    if (q == Q_NW) return true;
    for (int i = 0; i < n_unlocked - 1 && i < 3; ++i)
        if (LAND_ORDER[i] == q) return true;
    return false;
}

inline bool macro_walkable(int n_unlocked, int x, int y) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    const int d = shed_distance(x, y);
    if (d >= MAX_SHED_DIST) return false;
    if (!macro_owned(n_unlocked, x, y)) return d < LOCKED_STEP_DIST;
    return true;
}

namespace macro_detail {

inline void build_geo(MacroGeo& g, int n_unlocked) {
    static const int DX[4] = { 0, 0, 1, -1 };
    static const int DY[4] = { -1, 1, 0, 0 };
    const int N = MacroGeo::N;
    g.dist.assign((size_t)N * N, MacroGeo::INF);
    g.dir.assign((size_t)N * N, -1);

    std::vector<int> prev(N);
    for (int sy = 0; sy < BOARD_SIZE; ++sy)
    for (int sx = 0; sx < BOARD_SIZE; ++sx) {
        if (!macro_walkable(n_unlocked, sx, sy)) continue;
        const int s = sy * BOARD_SIZE + sx;
        std::fill(prev.begin(), prev.end(), -1);
        std::queue<int> q;
        g.dist[(size_t)s * N + s] = 0;
        q.push(s);
        while (!q.empty()) {
            const int c = q.front(); q.pop();
            const int cx = c % BOARD_SIZE, cy = c / BOARD_SIZE;
            for (int d = 0; d < 4; ++d) {
                const int nx = cx + DX[d], ny = cy + DY[d];
                if (!macro_walkable(n_unlocked, nx, ny)) continue;
                const int nb = ny * BOARD_SIZE + nx;
                if (g.dist[(size_t)s * N + nb] != MacroGeo::INF) continue;
                g.dist[(size_t)s * N + nb] = g.dist[(size_t)s * N + c] + 1;
                prev[nb] = c;
                q.push(nb);
            }
        }
        for (int b = 0; b < N; ++b) {
            if (b == s || g.dist[(size_t)s * N + b] == MacroGeo::INF) continue;
            int cur = b;
            while (prev[cur] != s) cur = prev[cur];
            const int cx = cur % BOARD_SIZE, cy = cur / BOARD_SIZE;
            for (int d = 0; d < 4; ++d)
                if (sx + DX[d] == cx && sy + DY[d] == cy)
                    g.dir[(size_t)s * N + b] = d;
        }
    }
}

inline const std::array<MacroGeo, 4>& geos() {
    static const std::array<MacroGeo, 4> G = [] {
        std::array<MacroGeo, 4> a;
        for (int n = 1; n <= 4; ++n) build_geo(a[n - 1], n);
        return a;
    }();
    return G;
}

}  // namespace macro_detail

inline const MacroGeo& macro_geo(int n_unlocked) {
    return macro_detail::geos()[std::clamp(n_unlocked, 1, 4) - 1];
}

// ===========================================================================
// MacroPlanner
// ===========================================================================
// A route is a plain sequence of CELLS. Everything else -- the pickups at the
// start, the walks, the ops, and every MID-DAY DROP -- is DERIVED from that
// sequence by one function, run(), which both the evaluator and the script
// writer call. They cannot disagree about a single hour.
//
// ---------------------------------------------------------------------------
// MID-DAY DROPS  (the HB_SELL harvest bin)
// ---------------------------------------------------------------------------
// A task whose harvest is SOLD carries a deadline: the units it banks must be
// at the shed no later than `sell_hour`, where a sell order of exactly those
// units goes out. Every other harvest (HB_KEEP, and every collect) stays in
// hand until the nightly sweep, as before.
//
// THE DROP IS NOT A NODE. The hand collects sell units into a BATCH and drops
// the batch LAZILY: it goes on to the next cell as long as it could still
// reach the nearest shed tile and finish the drop by the batch's EARLIEST
// deadline afterwards; the moment it could not, it drops first, taking the
// shed tile that is cheapest on the way to that next cell among the ones that
// still make the deadline. So a route's drops are a function of its cell order
// alone, 2-opt / or-opt / relocate / insertion all keep working on plain
// sequences, and every candidate they try is checked against every deadline.
//
// DROP ALL vs DROP ITEM. A_DROP with no item dumps the whole hand (one hour);
// with an item it stores that many of that one item (one hour per item).
//   * The hand carries ONLY the batch -- no input still to be spent later on
//     the route, no HB_KEEP harvest, no collected fertilizer -- so dropping
//     everything is dropping the batch: ONE A_DROP-all.
//   * Otherwise one item-drop per product in the batch, with the exact count.
//     A dump would put tomorrow's feed wheat, or a kept harvest, into the shed
//     mid-day -- the first breaks a later feed, the second breaks the shed
//     ledger the filter kept.
// run() prices whichever of the two applies, so the evaluator charges the
// hours flatten() will emit.
//
// THE COUNTS ARE EXACT. Within a day nothing but this player's own hands
// touches this player's board, so the units a harvest banks are a function of
// the hour it happens at: the hour-0 stack, minus what decay_plants eats on
// the turns before it (every other turn from max_lifespan_step), plus the
// pre-water's unit. The script knows that hour, so each drop carries the
// exact count and each sell order sells exactly what was dropped for it --
// never a unit of stock the morning sell head chose to hold.
//
// ---------------------------------------------------------------------------
// USING THE CREW  --  what reserve() tries before it says no
// ---------------------------------------------------------------------------
//   1. cheapest insertion over every hand and position; ties go to the
//      lighter hand, so a crew the passes hired is actually spread over
//   2. (deep) EJECTION: put the cell on a hand by pushing one of that hand's
//      cells onto another hand
// and the filter adds, where a refusal would cost money or a forced op:
//   3. tighten(): re-optimise every route (2-opt, or-opt, relocate), then
//      1 and 2 again on the shorter routes
// Only when all of that fails does the filter hire.
//
// SPEED. Every candidate position is first checked against a lower bound
// (insert_lb: the ops, plus the walking detour on routes without drops) and
// skipped without walking the route if it cannot fit the clock or beat the
// best so far; and optimize() does not re-run on routes that have not
// changed since a pass that found nothing.
// ===========================================================================
class MacroPlanner {
public:
    static constexpr int INF = MacroGeo::INF;

    // One task per cell that still needs work. `ops` is in execution order.
    struct Task {
        int x = 0, y = 0;
        std::vector<int> ops;      // action types
        std::vector<int> items;    // carryables consumed (WHEAT / FERT / animal)
        int n_asked  = 0;
        int type_ops = 0;
        int crop_id  = -1;         // operand for A_PLANT
        int animal_item_id = -1;   // operand for A_PLACE (an ITEM id)
        double value = 0.0;

        // ---- the harvest this task makes, if any ------------------------
        int h_op      = -1;        // index into ops of the A_HARVEST, or -1
        int h_product = -1;
        int h_units0  = 0;         // the stack at hour 0
        int h_bonus   = 0;         // what the pre-water adds (0 if none)
        int h_max     = 0;
        int h_mls     = -1;        // max_lifespan_step, -1 if it cannot decay
        int sell_hour = -1;        // >= 0: dropped by, and sold AT, this hour
        // Leaves something in hand that is NOT for sale today (an HB_KEEP
        // harvest, a collected fertilizer). Rules out the one-hour drop-all.
        bool carries_other = false;

        bool sells() const { return sell_hour >= 0 && h_op >= 0 && h_product >= 0; }
    };

    // One cell's sold units, as flatten() scheduled them.
    struct SellRec {
        int hour;        // the sell order's hour
        int product;
        int units;
        int drop_hour;   // the hour the last A_DROP of its batch executes
        int cell;
    };

    // ---- THE task extractor --------------------------------------------
    // OP ORDER IS LOAD-BEARING: the pre-water and the pre-harvest run before
    // the type fix (in that order: the water may add the unit the harvest
    // banks), and A_FERTILIZE runs before A_WATER.
    static bool build_task(const cell_goal& g, const tile& t, int day, Task& out) {
        if (!t.bought || !g.in_denominator) return false;

        out.ops.clear();
        out.items.clear();
        out.n_asked = goal_asked_count(g);
        out.crop_id = goal_is_plant(g) ? g.target_crop : -1;
        out.animal_item_id = goal_is_animal(g) ? animal_item(g.target_animal) : -1;
        out.h_op = -1; out.h_product = -1; out.h_units0 = 0; out.h_bonus = 0;
        out.h_max = 0; out.h_mls = -1; out.sell_hour = -1; out.carries_other = false;

        // Every A_HARVEST a task can hold acts on the HOUR-0 contents of the
        // cell (the pre-harvest, the ripe clearing harvest, the in-place
        // GB_HARVEST), so its units are read straight off `t`.
        bool prewatered = false;
        auto push_harvest = [&]() {
            out.h_op      = (int)out.ops.size();
            out.h_product = tile_product(t);
            out.h_units0  = t.yield_units;
            out.h_max     = tile_max_yield(t);
            out.h_mls     = (t.type == T_PLANT) ? t.max_lifespan_step : -1;
            out.h_bonus   = (prewatered && water_yields_now(t, day))
                          ? ((t.fertilized >= day) ? 2 : 1) : 0;
            out.ops.push_back(A_HARVEST);
        };

        const bool gate = goal_gate_passed(g, t,day);

        int pre = 0;
        if (g.want_pre_water && !gate &&
            t.type == T_PLANT && t.crop >= 0 && !t.watered_today) {
            out.ops.push_back(A_WATER);
            out.n_asked += 1;
            pre = 1;
            prewatered = true;
        }
        // HARVEST, DIG, PLANT: bank an ongoing crop's standing units before
        // the dig clears the tile for a replant. A non-ongoing crop needs no
        // flag -- its clearing op below IS the harvest (ripe_clear).
        if (g.want_pre_harvest && !gate &&
            t.type == T_PLANT && t.crop >= 0 && CROPS[t.crop].ongoing &&
            harvest_ready(t, day)) {
            push_harvest();
            out.n_asked += 1;
            pre += 1;
        }

        if (!gate) {
            if (goal_is_animal(g) && t.type == g.target_type && t.animal == -1) {
                out.ops.push_back(A_PLACE);
                out.items.push_back(animal_item(g.target_animal));
            } else if (t.animal >= 0) {
                return false;   // a live animal cannot be evicted inside a day
            } else {
                if (t.type != T_EMPTY) {
                    const bool ripe_clear =
                        t.type == T_PLANT && t.crop >= 0 &&
                        !CROPS[t.crop].ongoing && harvest_ready(t, day);
                    if (ripe_clear) push_harvest();
                    else            out.ops.push_back(A_DIG);
                }
                if (goal_is_plant(g)) {
                    out.ops.push_back(A_PLANT);
                } else if (goal_is_structure(g)) {
                    out.ops.push_back(g.target_type == T_COOP ? A_BUILD_COOP
                                                              : A_BUILD_PASTURE);
                    if (g.target_animal >= 0) {
                        out.ops.push_back(A_PLACE);
                        out.items.push_back(animal_item(g.target_animal));
                    }
                }
            }
        }
        out.type_ops = (int)out.ops.size() - pre;

        auto asked = [&](int b) {
            return goal_bool_applicable(g, b) && goal_wants(g, b)
                && !goal_bool_satisfied(b, t, day);
        };
        bool collects = false;
        if (asked(GB_HARVEST) && out.h_op < 0) push_harvest();
        if (asked(GB_FERTILIZED)){ out.ops.push_back(A_FERTILIZE); out.items.push_back(FERTILIZER); }
        if (asked(GB_WATERED))     out.ops.push_back(A_WATER);
        if (asked(GB_COLLECT))   { out.ops.push_back(A_COLLECT_FERTILIZER); collects = true; }
        if (asked(GB_FED))       { out.ops.push_back(A_FEED);      out.items.push_back(WHEAT); }
        if (asked(GB_CARED))       out.ops.push_back(A_CARE);

        if (out.ops.empty()) return false;

        if (out.h_op >= 0 && out.h_product >= 0 && out.h_product < NUM_PRODUCTS &&
            g.sell_hour >= 0 && g.sell_hour < TURNS_PER_DAY)
            out.sell_hour = g.sell_hour;
        out.carries_other = collects || (out.h_op >= 0 && out.sell_hour < 0);

        const int open = (int)out.ops.size() - out.type_ops;
        out.value = out.n_asked ? (double)open / (double)out.n_asked : 1.0;
        return true;
    }

    // What decay_plants takes off a stack on the turns [turn0, turn_h), from
    // max_lifespan_step on, every other turn. The harvest at turn_h runs in
    // apply_farmer_moves, BEFORE that turn's decay.
    static int decay_before(int mls, int turn0, int turn_h) {
        if (mls < 0) return 0;
        const int lo = std::max(mls, turn0), hi = turn_h - 1;
        if (lo > hi) return 0;
        const int first = lo + ((lo - mls) & 1);
        return first > hi ? 0 : (hi - first) / 2 + 1;
    }
    // Units the task's A_HARVEST puts in the hand if it runs at `hour`.
    int harvest_units(const Task& T, int hour) const {
        if (T.h_op < 0) return 0;
        const int u = T.h_units0 - decay_before(T.h_mls, turn0_, turn0_ + hour);
        if (u <= 0) return 0;                  // rotted to a weed first
        return std::min(T.h_max, u + T.h_bonus);
    }

    // ---- setup ---------------------------------------------------------
    // `proj` is the PROJECTED player state. turn_budget is how many of the
    // day's 24 hours remain once the market orders have had theirs, so the
    // script is PLANNED to start at hour 24 - turn_budget. It may start
    // earlier (fewer order chunks than priced); every deadline then only
    // gets easier.
    void init(const player_state& proj, int day, int turn_budget) {
        day_    = day;
        turn0_  = day * TURNS_PER_DAY;
        budget_ = std::max(0, turn_budget);
        ss_plan_ = TURNS_PER_DAY - budget_;
        geo_    = &macro_geo((int)proj.unlocked_quadrants.size());
        dist_   = geo_->dist.data();

        for (int i = 0; i < 4; ++i) {
            const ipos& a = shed_access_tiles()[i];
            access_[i] = a.y * BOARD_SIZE + a.x;
        }
        dshed_.assign(CELLS, INF);
        for (int c = 0; c < CELLS; ++c)
            for (const int a : access_) dshed_[c] = std::min(dshed_[c], geo_->D(c, a));

        home_.assign(CELLS, -1);
        task_.assign(CELLS, Task{});
        ti_.assign(CELLS, TInfo{});
        settled_ = false;
        routes_.assign(proj.farmers.size(), Route{});
        for (size_t h = 0; h < proj.farmers.size(); ++h)
            routes_[h].start = proj.farmers[h].y * BOARD_SIZE + proj.farmers[h].x;
        script_.clear();
        sells_.clear();
        script_start_ = 1;
    }

    int  hands()      const { return (int)routes_.size(); }
    int  budget()     const { return budget_; }
    int  cost(int h)  const { return routes_[h].cost; }
    int  slack(int h) const { return budget_ - routes_[h].cost; }
    bool routed(int cell) const { return home_[cell] >= 0; }
    const Task& task_at(int cell) const { return task_[cell]; }
    int  planned_start() const { return ss_plan_; }

    // The EARLIEST hour a task's sold units could possibly be at the shed:
    // a hand doing nothing else, starting on the nearest shed tile at the
    // planned script start. A lower bound, used to mask sell-time bins that
    // no crew could ever make.
    int min_drop_hour(int cell, const Task& t) const {
        if (dshed_[cell] >= INF) return INF;
        bool need[NUM_ITEMS] = { false };
        int picks = 0;
        for (const int it : t.items) if (!need[it]) { need[it] = true; ++picks; }
        // pickups, out, ops, back, and the drop's own hour
        return ss_plan_ + picks + 2 * dshed_[cell] + (int)t.ops.size();
    }

    // ---- growing the crew mid-plan -------------------------------------
    bool add_hand(const ipos& p) {
        if ((int)routes_.size() >= MAX_UNITS) return false;
        Route r;
        r.start = p.y * BOARD_SIZE + p.x;
        routes_.push_back(r);
        script_.clear();
        settled_ = false;
        return true;
    }

    int placed() const {
        int n = 0;
        for (const Route& r : routes_) n += (int)r.seq.size();
        return n;
    }
    int total_ops() const {
        int n = 0;
        for (const Route& r : routes_)
            for (const int v : r.seq) n += (int)task_[v].ops.size();
        return n;
    }
    int busy_hands() const {
        int n = 0;
        for (const Route& r : routes_) n += r.seq.empty() ? 0 : 1;
        return n;
    }
    std::array<int, NUM_ITEMS> needs() const {
        std::array<int, NUM_ITEMS> n{};
        for (const Route& r : routes_)
            for (const int v : r.seq)
                for (const int it : task_[v].items) n[it]++;
        return n;
    }

    // ---- THE capacity question -----------------------------------------
    // Install `t` as this cell's task. Returns false and leaves every task
    // and every route as feasible as it found them if the cell cannot be
    // served. `deep` spends more search before giving up (see the header);
    // it may shorten OTHER routes on the way, which is only ever an
    // improvement.
    bool reserve(int cell, const Task& t, bool deep = false) {
        if (t.ops.empty()) return true;
        if (routes_.empty()) return false;

        const int h = home_[cell];
        if (h >= 0) {
            Task old = task_[cell];
            set_task(cell, t);
            {
                const Eval e = eval(routes_[h].start, routes_[h].seq);
                if (e.ok) { routes_[h].cost = e.cost; settled_ = false; return true; }
            }
            // Take it out and place it anew; restore EXACTLY on failure.
            const std::vector<Route> saved_r = routes_;
            const std::vector<int>   saved_h = home_;
            if (detach(cell) &&
                (insert_best(cell) || (deep && eject_insert(cell)))) return true;
            routes_ = saved_r;
            home_   = saved_h;
            task_[cell] = std::move(old);
            refresh_ti(cell);
            settled_ = false;      // restored, but optimize may have run in between
            return false;
        }

        set_task(cell, t);
        if (insert_best(cell)) return true;
        if (deep && eject_insert(cell)) return true;
        set_task(cell, Task{});
        return false;
    }

    // Shorten every route (2-opt, or-opt, relocate) so a refused cell may fit
    // after all. The filter calls this at the two moments a refusal is
    // EXPENSIVE -- right before it would pay for a new hand, and before it
    // drops a forced op -- and nowhere else: a full pass costs far more than
    // an insertion. Returns whether anything got shorter.
    bool tighten() { return optimize(3); }

    // Removing a cell. False (and NOTHING changed -- the cell keeps its task
    // and its place) when no crew arrangement can run the rest without it;
    // see detach().
    bool release(int cell) {
        if (home_[cell] < 0) return true;
        if (!detach(cell)) return false;
        set_task(cell, Task{});
        return true;
    }

    // Every route evaluates and fits the clock. A plan that fails this must
    // never be flattened: run() stops emitting at the step it fails on, so
    // the hand would simply stop -- often right after a drop -- and every
    // cell after that point would be silently skipped.
    bool all_routes_ok() const {
        for (const Route& r : routes_) {
            if (r.seq.empty()) continue;
            const Eval e = run(r.start, r.seq, nullptr);
            if (!e.ok || e.cost > budget_) return false;
        }
        return true;
    }
    int broken_routes() const {
        int n = 0;
        for (const Route& r : routes_)
            if (!r.seq.empty() && (r.cost >= INF || r.cost > budget_)) ++n;
        return n;
    }

    // ---- pure descent: shorten routes, never drop a task ---------------
    // Returns whether anything improved.
    bool optimize(int max_rounds = 4) {
        // A pass that found nothing on routes nobody has touched since would
        // find nothing again. This is what makes a run of refusals on a full
        // crew cheap: only the first one pays for the search.
        if (settled_) return false;
        bool any = false;
        for (int round = 0; round < max_rounds; ++round) {
            bool improved = false;
            for (Route& r : routes_) { improved |= two_opt(r); improved |= or_opt(r); }
            improved |= relocate();
            any |= improved;
            if (!improved) break;
        }
        settled_ = true;
        return any;
    }

    // ---- route -> literal per-hour script ------------------------------
    void flatten(int start_hour) {
        script_start_ = std::max(1, start_hour);
        script_.assign(routes_.size(), {});
        sells_.clear();
        const int avail = std::max(0, TURNS_PER_DAY - script_start_);

        for (size_t h = 0; h < routes_.size(); ++h) {
            std::vector<SellRec> recs;
            Emitter em{ &script_[h], &recs, (int)h, script_start_ };
            run(routes_[h].start, routes_[h].seq, &em);
            std::vector<move>& s = script_[h];
            if ((int)s.size() > avail) s.resize(avail);
            // A sell whose drop fell off the end of the day would sell stock
            // the morning chose to hold. run() never plans one; this is the
            // belt to its braces.
            for (const SellRec& r : recs)
                if (r.drop_hour < script_start_ + (int)s.size() && r.units > 0)
                    sells_.push_back(r);
        }
    }

    // Unit moves for one hour of the day.
    void emit(int hour_in_day, std::vector<move>& moves) const {
        const int i = hour_in_day - script_start_;
        if (i < 0) return;
        for (size_t h = 0; h < script_.size(); ++h)
            if (i < (int)script_[h].size()) moves.push_back(script_[h][i]);
    }

    // Every mid-day sale the flattened script sets up, one record per cell.
    const std::vector<SellRec>& sells() const { return sells_; }

    int script_len() const {
        size_t n = 0;
        for (const auto& s : script_) n = std::max(n, s.size());
        return (int)n;
    }

    void dump(FILE* f = stdout) const {
        std::fprintf(f, "plan day %d: %zu hands, %d cells, %d ops, budget %d, start h%d, %zu mid-day sells\n",
                     day_, routes_.size(), placed(), total_ops(), budget_, script_start_,
                     sells_.size());
        for (size_t h = 0; h < routes_.size(); ++h)
            std::fprintf(f, "  hand %2zu cost %2d/%d  (%zu nodes, %zu moves)\n",
                         h, routes_[h].cost, budget_, routes_[h].seq.size(),
                         h < script_.size() ? script_[h].size() : 0);
    }

private:
    struct Route {
        int start = 0;
        std::vector<int> seq;      // cell ids, in visiting order
        int cost = 0;              // script length the route needs (INF: broken)
    };
    struct Eval {
        bool ok = false;
        int  cost = INF;
    };
    struct Emitter {
        std::vector<move>* s = nullptr;
        std::vector<SellRec>* sells = nullptr;
        int hi = 0;
        int start_hour = 1;
    };
    struct Pend { int cell, product, units, sell_hour; };

    const MacroGeo* geo_ = nullptr;
    int day_ = 0, turn0_ = 0, budget_ = 0, ss_plan_ = TURNS_PER_DAY, script_start_ = 1;
    int access_[4] = { 0, 0, 0, 0 };
    std::vector<int>   dshed_;     // per cell: steps to the nearest shed tile
    std::vector<Route> routes_;
    std::vector<Task>  task_;      // per cell
    std::vector<int>   home_;      // per cell -> hand, or -1
    std::vector<std::vector<move>> script_;
    std::vector<SellRec> sells_;
    std::vector<int> scratch_a_;
    bool settled_ = false;         // optimize() found nothing and no route changed since
    mutable std::vector<Pend> scratch_pend_;

    // ---- THE PRICING CACHE ----------------------------------------------
    // Everything run() reads off a Task when it only PRICES a route, flattened
    // into one small record per cell. Kept in step with task_ by set_task /
    // refresh_ti, the only places a task changes. Pure speed: the fast walker
    // below makes exactly the decisions run() makes, from the same numbers.
    struct TInfo {
        int      ops   = 0;        // T.ops.size()
        int      vdl   = INF;      // dl_step(sell_hour) if the task sells, else INF
        unsigned vbit  = 0;        // 1 << h_product if the task sells, else 0
        unsigned imask = 0;        // union of 1 << item over T.items
        bool     vs    = false;    // T.sells()
        bool     co    = false;    // T.carries_other
        bool     items = false;    // !T.items.empty()
    };
    std::vector<TInfo> ti_;        // per cell, parallel to task_
    // geo_->dist, read directly in the hot loops. The geometries are static
    // for the life of the process, so a copied planner may share it.
    const int* dist_ = nullptr;
    int DD(int a, int b) const { return dist_[a * MacroGeo::N + b]; }

    void refresh_ti(int cell) {
        const Task& T = task_[cell];
        TInfo& I = ti_[cell];
        I.ops   = (int)T.ops.size();
        I.vs    = T.sells();
        I.vdl   = I.vs ? dl_step(T.sell_hour) : INF;
        I.vbit  = I.vs ? (1u << T.h_product) : 0u;
        I.co    = T.carries_other;
        I.items = !T.items.empty();
        I.imask = 0;
        for (const int it : T.items) I.imask |= (1u << it);
    }
    void set_task(int cell, const Task& t) {
        task_[cell] = t;
        refresh_ti(cell);
    }

    // The walker's state between two nodes of a route. POD on purpose: the
    // prefix tables below hold a hundred of these and must not be zeroed.
    struct WS {
        int t, cur, pdl;
        unsigned pmask;
        bool other;
    };
    static int drop_cost_of(unsigned mask, bool dirty) {
        return dirty ? __builtin_popcount(mask) : 1;
    }
    WS ws_begin(int start, unsigned imask) const {
        WS s;
        s.t = __builtin_popcount(imask);   // one pickup per distinct item
        s.cur = start;
        s.pdl = INF;
        s.pmask = 0;
        s.other = false;
        return s;
    }
    // One node of run() with `em` null. in_i / in_n are run()'s in_from[i]
    // and in_from[i + 1]. False exactly where run() returns a failed Eval.
    bool ws_step(WS& s, int v, bool in_i, bool in_n) const {
        const TInfo& T = ti_[v];
        int d = DD(s.cur, v);
        if (d >= INF) return false;

        bool go_on = true;
        if (s.pmask | T.vbit) {
            const bool dirty = s.other || T.co || in_n;
            go_on = s.t + d + T.ops + dshed_[v] + drop_cost_of(s.pmask | T.vbit, dirty) - 1
                    <= std::min(s.pdl, T.vdl);
        }
        if (!go_on) {
            if (!s.pmask) return false;
            const bool dirty = s.other || in_i;
            const int dc = drop_cost_of(s.pmask, dirty);
            int best_a = -1, best_c = INF;
            for (const int a : access_) {
                const int d1 = DD(s.cur, a), d2 = DD(a, v);
                if (d1 >= INF || d2 >= INF) continue;
                if (s.t + d1 + dc - 1 > s.pdl) continue;
                if (d1 + d2 < best_c) { best_c = d1 + d2; best_a = a; }
            }
            if (best_a < 0) return false;
            s.t += DD(s.cur, best_a);
            s.cur = best_a;
            s.t += dc;
            s.pmask = 0; s.pdl = INF;
            if (s.t > budget_) return false;
            d = DD(s.cur, v);
            if (T.vs) {
                const bool dirty2 = s.other || T.co || in_n;
                if (s.t + d + T.ops + dshed_[v] + drop_cost_of(T.vbit, dirty2) - 1 > T.vdl)
                    return false;
            }
        }
        s.t += d + T.ops;
        s.cur = v;
        if (s.t > budget_) return false;
        if (T.vs) { s.pmask |= T.vbit; s.pdl = std::min(s.pdl, T.vdl); }
        s.other = s.other || T.co;
        return true;
    }
    // run()'s tail: home with the last batch, then the clock check.
    Eval ws_finish(WS s) const {
        Eval r;
        if (s.pmask) {
            const bool dirty = s.other;
            const int dc = drop_cost_of(s.pmask, dirty);
            int best_a = -1, best_d = INF;
            for (const int a : access_) {
                const int d1 = DD(s.cur, a);
                if (d1 < best_d) { best_d = d1; best_a = a; }
            }
            if (best_a < 0 || s.t + best_d + dc - 1 > s.pdl) return r;
            s.t += best_d;
            s.t += dc;
        }
        r.ok = s.t <= budget_;
        r.cost = s.t;
        return r;
    }
    // run(start, seq, nullptr) on a raw array.
    Eval eval_arr(int start, const int* seq, int n) const {
        char in_from[CELLS + 2];
        unsigned m = 0;
        in_from[n] = 0;
        for (int i = n - 1; i >= 0; --i) {
            const TInfo& T = ti_[seq[i]];
            in_from[i] = in_from[i + 1] || T.items;
            m |= T.imask;
        }
        WS s = ws_begin(start, m);
        for (int i = 0; i < n; ++i)
            if (!ws_step(s, seq[i], in_from[i], in_from[i + 1])) return Eval{};
        return ws_finish(s);
    }

    // ---- PRICING ONE INSERTION, MANY POSITIONS --------------------------
    // Every search operator asks the same question over and over: base route
    // B with one extra cell c put in at position p, for p = 0 .. |B|. The walk
    // over B[0 .. p-1] is the same for every p (the pickup count and every
    // in_from flag before p already include c), so those states are computed
    // once, lazily, and each position only walks c and B[p ..].
    // With c == -1 it is B itself: the prefix states two_opt resumes from.
    struct InsEval {
        const MacroPlanner* P = nullptr;
        const int* B = nullptr;
        int n = 0, cell = -1;
        bool hc = false;           // c carries an input
        int built = 0;             // st[0 .. built] are valid
        int fail_at = INT_MAX;     // the prefix step that failed
        char inB[CELLS + 2];       // run()'s in_from over B alone
        WS st[CELLS + 2];

        void init(const MacroPlanner* planner, int start, const int* base, int nb, int c) {
            P = planner; B = base; n = nb; cell = c;
            unsigned m = 0;
            inB[n] = 0;
            for (int i = n - 1; i >= 0; --i) {
                const TInfo& T = P->ti_[B[i]];
                inB[i] = inB[i + 1] || T.items;
                m |= T.imask;
            }
            hc = false;
            if (c >= 0) { m |= P->ti_[c].imask; hc = P->ti_[c].items; }
            st[0] = P->ws_begin(start, m);
            built = 0;
            fail_at = INT_MAX;
        }
        // The state before B[p] (and, with a cell, before c at p). False if
        // the walk already failed on the way there.
        bool prefix(int p) {
            if (p > fail_at) return false;
            while (built < p) {
                WS s = st[built];
                if (!P->ws_step(s, B[built], inB[built] || hc, inB[built + 1] || hc)) {
                    fail_at = built;
                    return false;
                }
                st[++built] = s;
            }
            return true;
        }
        // B with `cell` inserted at p.
        Eval at(int p) {
            if (!prefix(p)) return Eval{};
            WS s = st[p];
            if (!P->ws_step(s, cell, inB[p] || hc, inB[p])) return Eval{};
            for (int m = p; m < n; ++m)
                if (!P->ws_step(s, B[m], inB[m], inB[m + 1])) return Eval{};
            return P->ws_finish(s);
        }
        // B with B[i .. j] reversed (cell must be -1).
        Eval reversed(int i, int j) {
            if (!prefix(i)) return Eval{};
            int c[CELLS + 2];
            char in_c[CELLS + 2];
            for (int k = i; k < n; ++k) c[k] = (k <= j) ? B[i + j - k] : B[k];
            in_c[n] = 0;
            for (int k = n - 1; k >= i; --k) in_c[k] = in_c[k + 1] || P->ti_[c[k]].items;
            WS s = st[i];
            for (int k = i; k < n; ++k)
                if (!P->ws_step(s, c[k], in_c[k], in_c[k + 1])) return Eval{};
            return P->ws_finish(s);
        }
    };

#ifdef MACRO_PLANNER_VERIFY
    // Debug: every fast answer checked against the reference walker.
    void verify(int start, const std::vector<int>& seq, const Eval& fast) const {
        const Eval ref = run(start, seq, nullptr);
        if (ref.ok != fast.ok || ref.cost != fast.cost) {
            std::fprintf(stderr, "MacroPlanner fast eval mismatch: ref %d/%d fast %d/%d\n",
                         (int)ref.ok, ref.cost, (int)fast.ok, fast.cost);
            std::abort();
        }
    }
#endif

    // Last step index a drop for an order at `hour` may take, on the PLANNED
    // clock (step k executes at hour ss_plan_ + k).
    int dl_step(int hour) const { return hour - ss_plan_; }

    // ---- THE route walker ----------------------------------------------
    // One definition of what a sequence costs and what it does, hour by
    // hour. With `em` null it only prices; with `em` it also writes the
    // moves and the sell records. Fails on an unreachable node, a clock
    // overrun, or a batch that cannot make its deadline.
    Eval run(int start, const std::vector<int>& seq, const Emitter* em) const {
        static const int DX[4] = { 0, 0, 1, -1 };
        static const int DY[4] = { -1, 1, 0, 0 };
        Eval r;
        const int n = (int)seq.size();

        // in_from[i]: some task at index >= i still spends an input, so the
        // hand is carrying something a drop-all would throw into the shed.
        char in_from[CELLS + 1];              // a route never holds a cell twice
        in_from[n] = 0;
        for (int i = n - 1; i >= 0; --i)
            in_from[i] = in_from[i + 1] || !task_[seq[i]].items.empty();

        int need[NUM_ITEMS] = { 0 };
        for (const int v : seq)
            for (const int it : task_[v].items) need[it]++;
        int t = 0;
        for (int k = 0; k < NUM_ITEMS; ++k)
            if (need[k] > 0) {
                if (em) em->s->push_back(act_pickup(em->hi, k, need[k]));
                ++t;
            }

        int cur = start;
        int pdl = INF;               // latest step the batch's LAST drop may take
        unsigned pmask = 0;          // products in the batch
        bool other = false;          // carrying anything that is not the batch
        std::vector<Pend>& pend = scratch_pend_;
        pend.clear();

        auto drop_cost = [](unsigned mask, bool dirty) -> int {
            return dirty ? __builtin_popcount(mask) : 1;
        };
        auto walk = [&](int from, int to) {
            int c = from;
            while (c != to) {
                const int d = geo_->DIR(c, to);
                if (d < 0) break;
                em->s->push_back(act(em->hi, d));     // d == A_MOVE_*
                c = (c / BOARD_SIZE + DY[d]) * BOARD_SIZE + (c % BOARD_SIZE + DX[d]);
            }
        };
        // The batch's drop actions start at step t.
        auto emit_drop = [&](bool dirty, int dc) {
            if (!dirty) {
                em->s->push_back(act_drop_all(em->hi));
            } else {
                for (int k = 0; k < NUM_PRODUCTS; ++k) {
                    if (!((pmask >> k) & 1u)) continue;
                    int u = 0;
                    for (const Pend& p : pend) if (p.product == k) u += p.units;
                    em->s->push_back(u > 0 ? act_drop(em->hi, k, u) : act(em->hi, A_PASS));
                }
            }
            const int last = em->start_hour + t + dc - 1;
            for (const Pend& p : pend)
                em->sells->push_back({ std::max(p.sell_hour, last), p.product,
                                       p.units, last, p.cell });
            pend.clear();
        };

        for (int i = 0; i < n; ++i) {
            const int v = seq[i];
            const Task& T = task_[v];
            const int ops = (int)T.ops.size();
            const bool vs = T.sells();
            const int vdl = vs ? dl_step(T.sell_hour) : INF;
            const unsigned vbit = vs ? (1u << T.h_product) : 0u;

            int d = geo_->D(cur, v);
            if (d >= INF) return r;

            // Going on to v: can the batch, plus v's units, still make it?
            bool go_on = true;
            if (pmask | vbit) {
                const bool dirty = other || T.carries_other || in_from[i + 1];
                go_on = t + d + ops + dshed_[v] + drop_cost(pmask | vbit, dirty) - 1
                        <= std::min(pdl, vdl);
            }
            if (!go_on) {
                if (!pmask) return r;           // v alone cannot make its hour
                // Drop the batch on the way to v.
                const bool dirty = other || in_from[i];
                const int dc = drop_cost(pmask, dirty);
                int best_a = -1, best_c = INF;
                for (const int a : access_) {
                    const int d1 = geo_->D(cur, a), d2 = geo_->D(a, v);
                    if (d1 >= INF || d2 >= INF) continue;
                    if (t + d1 + dc - 1 > pdl) continue;
                    if (d1 + d2 < best_c) { best_c = d1 + d2; best_a = a; }
                }
                if (best_a < 0) return r;
                if (em) walk(cur, best_a);
                t += geo_->D(cur, best_a);
                cur = best_a;
                if (em) emit_drop(dirty, dc);
                t += dc;
                pmask = 0; pdl = INF;
                if (t > budget_) return r;
                d = geo_->D(cur, v);
                if (vs) {
                    const bool dirty2 = other || T.carries_other || in_from[i + 1];
                    if (t + d + ops + dshed_[v] + drop_cost(vbit, dirty2) - 1 > vdl)
                        return r;
                }
            }

            if (em) {
                walk(cur, v);
                for (const int op : T.ops) {
                    if (op == A_PLANT)      em->s->push_back(act_plant(em->hi, T.crop_id));
                    else if (op == A_PLACE) em->s->push_back(act_place_item(em->hi, T.animal_item_id));
                    else                    em->s->push_back(act(em->hi, op));
                }
                if (vs) {
                    const int hh = em->start_hour + t + d + T.h_op;
                    pend.push_back({ v, T.h_product, harvest_units(T, hh), T.sell_hour });
                }
            }
            t += d + ops;
            cur = v;
            if (t > budget_) return r;
            if (vs) { pmask |= vbit; pdl = std::min(pdl, vdl); }
            other = other || T.carries_other;
        }

        if (pmask) {                            // home with the last batch
            const bool dirty = other;           // no input is left to spend
            const int dc = drop_cost(pmask, dirty);
            int best_a = -1, best_d = INF;
            for (const int a : access_) {
                const int d1 = geo_->D(cur, a);
                if (d1 < best_d) { best_d = d1; best_a = a; }
            }
            if (best_a < 0 || t + best_d + dc - 1 > pdl) return r;
            if (em) walk(cur, best_a);
            t += best_d;
            if (em) emit_drop(dirty, dc);
            t += dc;
        }
        r.ok = t <= budget_;
        r.cost = t;
        return r;
    }

    Eval eval(int start, const std::vector<int>& seq) const {
        const Eval e = eval_arr(start, seq.data(), (int)seq.size());
#ifdef MACRO_PLANNER_VERIFY
        verify(start, seq, e);
#endif
        return e;
    }

#ifdef MACRO_PLANNER_VERIFY
    Eval checked_at(InsEval& ie, int start, int p) const {
        const Eval e = ie.at(p);
        std::vector<int> c(ie.B, ie.B + ie.n);
        c.insert(c.begin() + p, ie.cell);
        verify(start, c, e);
        return e;
    }
    Eval checked_rev(InsEval& ie, int start, int i, int j) const {
        const Eval e = ie.reversed(i, j);
        std::vector<int> c(ie.B, ie.B + ie.n);
        std::reverse(c.begin() + i, c.begin() + j + 1);
        verify(start, c, e);
        return e;
    }
#define MP_AT(ie, start, p)       checked_at((ie), (start), (p))
#define MP_REV(ie, start, i, j)   checked_rev((ie), (start), (i), (j))
#else
#define MP_AT(ie, start, p)       (ie).at(p)
#define MP_REV(ie, start, i, j)   (ie).reversed((i), (j))
#endif

    // Does any task on this sequence sell? Only then can a drop trip make the
    // route's cost anything other than pickups + walking + ops.
    bool has_sell(const std::vector<int>& seq) const {
        for (const int v : seq) if (ti_[v].vs) return true;
        return false;
    }
    // A LOWER BOUND on what inserting `cell` at position p adds to a route:
    // its own ops plus the walking detour. Used to SKIP positions that cannot
    // fit the clock or beat the best one found -- never to accept one.
    //
    // `strict` claims only what is certain. On a route with no drops the
    // detour bound IS certain (board distance is a metric). With drops in play
    // the lazy drop can re-plan around the new cell and occasionally save a
    // step, so the detour is a HEURISTIC there: measured wrong on ~0.05% of
    // the positions it skips. insert_best therefore screens with the full
    // bound first and, only if that leaves no position at all, re-checks the
    // drop routes strictly -- so the speed costs at most a marginally better
    // position, never a cell that would have fit.
    int insert_lb(int start, const std::vector<int>& seq, int p, int cell,
                  bool sells_on_route, bool strict = false) const {
        const TInfo& T = ti_[cell];
        int lb = T.ops;
        if (strict && (sells_on_route || T.vs)) return lb;
        const int prev = p == 0 ? start : seq[p - 1];
        const int a = DD(prev, cell);
        if (a >= INF) return INF;
        lb += a;
        if (p < (int)seq.size()) {
            const int next = seq[p];
            lb += DD(cell, next) - DD(prev, next);
        }
        return lb;
    }

    // Recompute a route's cost; a route that no longer evaluates (possible in
    // principle after a removal, because the lazy drop is a heuristic) is
    // re-ordered until it does.
    void refresh(Route& r) {
        const Eval e = eval(r.start, r.seq);
        if (e.ok) { r.cost = e.cost; return; }
        r.cost = INF;
        two_opt(r);
        or_opt(r);
    }

    // Take `cell` off its route. The rest of the route must still run.
    //
    // BUGFIX. It used to be assumed that a SHORTER route always runs, and a
    // route that did not was left in the plan at cost INF. It does not
    // always run: the lazy drop is greedy, so arriving earlier can make the
    // hand carry a sell batch one cell further and then miss the next
    // cell's deadline. A route left at INF was (a) skipped by every
    // insertion and relocation for the rest of the day and (b) flattened by
    // run(), which stops emitting at the step that fails -- so the hand
    // stopped dead, typically right after a drop, and every cell behind that
    // point (feeds and harvests included) was never done.
    //
    // Now: re-order (2-opt, or-opt); failing that, RESEAT the route's cells
    // across the whole crew; failing that, restore everything exactly and
    // return false, so the caller keeps the cell as it was.
    bool detach(int cell) {
        const int h = home_[cell];
        if (h < 0) return true;
        const std::vector<Route> saved_r = routes_;
        const std::vector<int>   saved_h = home_;
        settled_ = false;
        Route& r = routes_[h];
        r.seq.erase(std::find(r.seq.begin(), r.seq.end(), cell));
        home_[cell] = -1;
        refresh(r);
        if (routes_[h].cost < INF) return true;

        std::vector<int> cells = routes_[h].seq;
        routes_[h].seq.clear();
        routes_[h].cost = 0;
        for (const int v : cells) home_[v] = -1;
        bool ok = true;
        for (const int v : cells)
            if (!insert_best(v) && !eject_insert(v)) { ok = false; break; }
        if (ok) return true;
        routes_ = saved_r;
        home_   = saved_h;
        settled_ = false;
        return false;
    }

    // Cheapest insertion over every hand and position. Ties go to the
    // lighter resulting route, which is what spreads a paid-for crew.
    bool insert_best(int cell) {
        if (insert_scan(cell, false)) return true;
        return insert_scan(cell, true);        // strict: see insert_lb
    }
    bool insert_scan(int cell, bool strict) {
        int best_h = -1, best_p = -1, best_cost = 0;
        long long best_key = LLONG_MAX;
        InsEval ie;
        for (int hh = 0; hh < (int)routes_.size(); ++hh) {
            const Route& r = routes_[hh];
            if (geo_->D(r.start, cell) >= INF || r.cost >= INF) continue;
            const int sz = (int)r.seq.size();
            const bool sl = has_sell(r.seq);
            // The strict pass only needs the routes the heuristic may have
            // misjudged: the ones with drops (or a selling cell).
            if (strict && !sl && !ti_[cell].vs) continue;
            // Every position's bound is at least the cell's own ops (also in
            // the strict pass), so a route that cannot take even those is
            // skipped whole. Exact: the loop below would reject every p.
            if (r.cost + ti_[cell].ops > budget_) continue;
            bool ready = false;
            for (int p = 0; p <= sz; ++p) {
                const int lb = insert_lb(r.start, r.seq, p, cell, sl, strict);
                if (lb >= INF || r.cost + lb > budget_) continue;
                if ((long long)lb * 64 + r.cost + lb >= best_key) continue;
                if (!ready) { ie.init(this, r.start, r.seq.data(), sz, cell); ready = true; }
                const Eval e = MP_AT(ie, r.start, p);
                if (!e.ok) continue;
                const long long delta = (long long)e.cost - (long long)r.cost;
                const long long key = delta * 64 + e.cost;
                if (key < best_key) {
                    best_key = key; best_h = hh; best_p = p; best_cost = e.cost;
                }
            }
        }
        if (best_h < 0) return false;
        Route& r = routes_[best_h];
        r.seq.insert(r.seq.begin() + best_p, cell);
        r.cost = best_cost;
        home_[cell] = best_h;
        settled_ = false;
        return true;
    }

    // EJECTION: put `cell` on hand h in place of one of h's cells, u, and
    // put u on another hand. The cheapest such pair wins. Changes nothing on
    // failure.
    bool eject_insert(int cell) {
        long long best_total = LLONG_MAX;
        int bh = -1, bh2 = -1, bu = -1;
        std::vector<int> best_r, best_r2;
        std::vector<int> base;
        InsEval ieb, ie2;
        for (int h = 0; h < (int)routes_.size(); ++h) {
            const Route& R = routes_[h];
            if (R.seq.empty() || geo_->D(R.start, cell) >= INF) continue;
            for (size_t i = 0; i < R.seq.size(); ++i) {
                const int u = R.seq[i];
                base = R.seq;
                base.erase(base.begin() + i);
                // cell into h without u. The shortened route is walked once;
                // every position is then screened by its lower bound first.
                const Eval eb = eval(R.start, base);
                if (!eb.ok) continue;
                if (eb.cost + ti_[cell].ops > budget_) continue;
                // Same bound as in relocate(): lb >= ops at every position.
                if ((long long)eb.cost + ti_[cell].ops - R.cost >= best_total) continue;
                const bool slb = has_sell(base);
                int ch = INF, ch_p = -1;
                bool ready = false;
                for (size_t p = 0; p <= base.size(); ++p) {
                    const int lb = insert_lb(R.start, base, (int)p, cell, slb);
                    if (lb >= INF || eb.cost + lb > budget_ || eb.cost + lb >= ch) continue;
                    if ((long long)eb.cost + lb - R.cost >= best_total) continue;
                    if (!ready) {
                        ieb.init(this, R.start, base.data(), (int)base.size(), cell);
                        ready = true;
                    }
                    const Eval e = MP_AT(ieb, R.start, (int)p);
                    if (e.ok && e.cost < ch) { ch = e.cost; ch_p = (int)p; }
                }
                if (ch >= INF) continue;
                const long long dh = (long long)ch - (long long)R.cost;
                if (dh >= best_total) continue;
                if (dh + ti_[u].ops >= best_total) continue;   // lb >= ops, as above
                // u onto another hand
                for (int h2 = 0; h2 < (int)routes_.size(); ++h2) {
                    if (h2 == h) continue;
                    const Route& R2 = routes_[h2];
                    if (geo_->D(R2.start, u) >= INF || R2.cost >= INF) continue;
                    if (R2.cost + ti_[u].ops > budget_) continue;   // lb >= ops
                    const bool sl = has_sell(R2.seq);
                    bool ready2 = false;
                    for (size_t p = 0; p <= R2.seq.size(); ++p) {
                        const int lb = insert_lb(R2.start, R2.seq, (int)p, u, sl);
                        if (lb >= INF || R2.cost + lb > budget_) continue;
                        if (dh + lb >= best_total) continue;
                        if (!ready2) {
                            ie2.init(this, R2.start, R2.seq.data(), (int)R2.seq.size(), u);
                            ready2 = true;
                        }
                        const Eval e2 = MP_AT(ie2, R2.start, (int)p);
                        if (!e2.ok) continue;
                        const long long tot = dh + (long long)e2.cost - (long long)R2.cost;
                        if (tot < best_total) {
                            best_total = tot; bh = h; bh2 = h2; bu = u;
                            best_r = base;
                            best_r.insert(best_r.begin() + ch_p, cell);
                            best_r2 = R2.seq;
                            best_r2.insert(best_r2.begin() + p, u);
                        }
                    }
                }
            }
        }
        if (bh < 0) return false;
        routes_[bh].seq  = best_r;
        routes_[bh2].seq = best_r2;
        routes_[bh].cost  = eval(routes_[bh].start,  routes_[bh].seq).cost;
        routes_[bh2].cost = eval(routes_[bh2].start, routes_[bh2].seq).cost;
        home_[cell] = bh;
        home_[bu]   = bh2;
        settled_ = false;
        return true;
    }

    bool two_opt(Route& r) {
        if (r.seq.size() < 2) return false;
        bool any = false, improved = true;
        InsEval ie;
        while (improved) {
            improved = false;
            // Prefix states over the CURRENT r.seq; rebuilt after every
            // accepted reversal, since the sequence changed under them.
            ie.init(this, r.start, r.seq.data(), (int)r.seq.size(), -1);
            for (size_t i = 0; i + 1 < r.seq.size(); ++i)
            for (size_t j = i + 1; j < r.seq.size(); ++j) {
                const Eval e = MP_REV(ie, r.start, (int)i, (int)j);
                if (e.ok && e.cost < r.cost) {
                    std::reverse(r.seq.begin() + i, r.seq.begin() + j + 1);
                    r.cost = e.cost;
                    improved = any = true;
                    ie.init(this, r.start, r.seq.data(), (int)r.seq.size(), -1);
                }
            }
        }
        return any;
    }

    // Move one node to another position of the same route.
    bool or_opt(Route& r) {
        if (r.seq.size() < 2) return false;
        bool any = false, improved = true;
        std::vector<int> base;
        InsEval ie;
        while (improved) {
            improved = false;
            for (size_t i = 0; i < r.seq.size() && !improved; ++i) {
                base = r.seq;
                const int v = base[i];
                base.erase(base.begin() + i);
                ie.init(this, r.start, base.data(), (int)base.size(), v);
                for (size_t p = 0; p <= base.size(); ++p) {
                    if (p == i) continue;
                    const Eval e = MP_AT(ie, r.start, (int)p);
                    if (e.ok && e.cost < r.cost) {
                        r.seq = base;
                        r.seq.insert(r.seq.begin() + p, v);
                        r.cost = e.cost;
                        improved = any = true;
                        break;
                    }
                }
            }
        }
        return any;
    }

    // Move one cell to another hand when the pair gets cheaper.
    bool relocate() {
        bool any = false;
        std::vector<int> r1;
        InsEval ie;
        for (int h1 = 0; h1 < (int)routes_.size(); ++h1) {
            for (size_t i = 0; i < routes_[h1].seq.size(); ++i) {
                const int cell = routes_[h1].seq[i];
                r1 = routes_[h1].seq;
                r1.erase(r1.begin() + i);
                const Eval e1 = eval(routes_[h1].start, r1);
                if (!e1.ok) continue;
                const long long gain = (long long)routes_[h1].cost - e1.cost;
                // Every position's bound is at least the cell's own ops (the
                // detour is never negative: board distance is a metric), and
                // best_delta starts at 0, so the scan below cannot accept
                // anything once ops - gain >= 0. Skipping it is exact.
                if ((long long)ti_[cell].ops - gain >= 0) continue;

                int best_h2 = -1, best_cost = 0, best_p = -1;
                long long best_delta = 0;
                for (int h2 = 0; h2 < (int)routes_.size(); ++h2) {
                    if (h2 == h1) continue;
                    const Route& base2 = routes_[h2];
                    if (geo_->D(base2.start, cell) >= INF || base2.cost >= INF) continue;
                    if (base2.cost + ti_[cell].ops > budget_) continue;   // lb >= ops
                    const bool sl = has_sell(base2.seq);
                    bool ready = false;
                    for (size_t p = 0; p <= base2.seq.size(); ++p) {
                        const int lb = insert_lb(base2.start, base2.seq, (int)p, cell, sl);
                        if (lb >= INF || base2.cost + lb > budget_) continue;
                        if ((long long)lb - gain >= best_delta) continue;
                        if (!ready) {
                            ie.init(this, base2.start, base2.seq.data(),
                                    (int)base2.seq.size(), cell);
                            ready = true;
                        }
                        const Eval e2 = MP_AT(ie, base2.start, (int)p);
                        if (!e2.ok) continue;
                        const long long delta = ((long long)e2.cost - base2.cost) - gain;
                        if (delta < best_delta) {
                            best_delta = delta; best_h2 = h2;
                            best_cost = e2.cost; best_p = (int)p;
                        }
                    }
                }
                if (best_h2 >= 0) {
                    routes_[h1].seq  = r1;
                    routes_[h1].cost = e1.cost;
                    std::vector<int>& s2 = routes_[best_h2].seq;
                    s2.insert(s2.begin() + best_p, cell);
                    routes_[best_h2].cost = best_cost;
                    home_[cell] = best_h2;
                    any = true;
                    break;                      // indices of h1 just moved
                }
            }
        }
        return any;
    }
#undef MP_AT
#undef MP_REV
};


// ===========================================================================
// One day's accepted plan
// ===========================================================================
// WHERE THE DAY'S 24 HOURS GO
//   hour 0 .. n_order_hours-1   morning market orders (stock sells, hires,
//                               land, derived buys)
//   hour n_order_hours .. 23    the planner's script, one move per hand per
//                               hour. HB_KEEP harvests ride in hand to the
//                               nightly sweep and are sold by tomorrow's sell
//                               head.
//   any hour >= the script      TIMED SELLS: the HB_SELL harvests, dropped by
//                               the hands no later than that hour and sold at
//                               it. apply_turn runs the hands' moves BEFORE
//                               process_market, so a drop and its sell may
//                               share an hour. One entry per product per hour,
//                               so never more than NUM_PRODUCTS (9) of the
//                               MAX_MARKET_ORDERS_PER_TURN (10) per turn.
// ===========================================================================
struct MacroStats {
    // THE REFEREE'S HIRE RULE (one hand per HIRE entry, ten entries a turn).
    // How many times the day was planned again because its hires needed more
    // order hours; 1 if even the last attempt did not fit; entries the
    // liquidation could not send by hour 23.
    int replanned = 0;
    int replan_overflow = 0;
    int liq_orders_lost = 0;
    int drawn = 0;
    int accepted_cells = 0;
    int accepted_ops = 0;
    int rej_illegal = 0;
    int rej_budget = 0;
    int rej_capacity = 0;
    int hired = 0;
    bool bought_land = false;
    double spent = 0.0;
    double raised = 0.0;

    // ---- the forced tier ----
    int must_ops = 0;
    int must_placed = 0;
    int must_dropped = 0;

    // ---- the derived ops ----
    int fertilized = 0;
    int cared = 0;
    int collected = 0;       // the collect head's accepted collects
    int escape_harvests = 0; // animals left unfed tonight, harvested so the
                             // units are not lost with the escape
    // ---- hires, by the pass that made them (there is no hire head) ----
    int must_hires = 0;      // the forced pass
    int plant_hires = 0;     // a drawn planting
    int fert_hires = 0;      // an accepted fertilize
    int feed_hires = 0;      // an accepted feed (or care on a forced feed)
    int collect_hires = 0;   // an accepted collect, or an escape harvest
    int broken_routes = 0;   // routes that do not run at flatten time (must stay 0)

    // ---- the melon mask ----
    int melon_cap = -1;      // new melons that still pay today (-1: not computed)
    int rej_melon = 0;       // melon had room and money but no market

    // ---- the harvest bins ----
    int harvest_keep = 0;    // HB_KEEP harvests (drawn or forced)
    int harvest_sell = 0;    // HB_SELL harvests routed
    int sell_fallback = 0;   // HB_SELL left with ONE option after a remask (no choice made)
    int sell_keep = 0;       // the sell-time head chose ST_KEEP (first draw or remask)
    int sell_skip = 0;       // the sell-time head chose ST_NO_HARVEST (first draw or remask)
    int sell_redrawn = 0;    // a sale routed at an hour drawn on a REMASK
    int sell_units = 0;      // units sold mid-day, as scheduled
    int harvest_hires = 0;   // hands hired mid-plan to fit a harvest or a mid-day drop
    int busy_hands = 0;      // hands with at least one cell today
    int night_ledger = 0;    // the filter's forecast of the shed after tonight's sweep
};

struct MacroDay {
    cell_goal grid[BOARD_SIZE][BOARD_SIZE];
    MacroPlanner planner;
    std::vector<std::vector<move>> order_chunks;   // chunk i goes out at hour i
    // The HB_SELL harvests' sell orders, by hour. Kept apart from
    // order_chunks because n_order_hours is the SCRIPT START and must not
    // grow with them.
    std::array<std::vector<move>, TURNS_PER_DAY> timed_orders;
    int n_order_hours = 1;
    MacroStats stats;

    void clear() {
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x) grid[y][x] = cell_goal{};
        order_chunks.clear();
        for (auto& v : timed_orders) v.clear();
        n_order_hours = 1;
        stats = MacroStats{};
    }

    void emit(int hour_in_day, std::vector<move>& moves) const {
        if (hour_in_day >= 0 && hour_in_day < (int)order_chunks.size())
            for (const move& m : order_chunks[hour_in_day]) moves.push_back(m);
        planner.emit(hour_in_day, moves);
        if (hour_in_day >= 0 && hour_in_day < TURNS_PER_DAY)
            for (const move& m : timed_orders[hour_in_day]) moves.push_back(m);
    }
};

#endif  // MACRO_PLANNER_HPP