#ifndef PROJECTION_HPP
#define PROJECTION_HPP
// ===========================================================================
// PROJECTED MONEY AND MARKET  --  critic-only features
// ===========================================================================
// What each player's `money` will be -- tomorrow morning, the morning after,
// ... ten mornings out, and on the last night -- if nobody plants, places,
// buys land or changes course from here on: the RUN-OFF VALUE of the book as
// it stands, and the market prices that run-off produces along the way.
//
// THREE STAGES
//
//   1. TILE RUN-OFF. Every plant and animal is played forward day by day
//      under a CANONICAL POLICY, replicating the referee's own arithmetic
//      (A_WATER's yield window, daily_refresh_plants, daily_refresh_animals,
//      the fertilizer cover window, decay). Output per player: units of each
//      product that become SELLABLE in each slot, plus the wheat and
//      fertilizer each day consumes and the ops each day costs.
//
//   2. ONE SHARED MARKET, IN TIME ORDER. Both players' streams are replayed
//      against a single market inventory, day by day, with the town drain in
//      between. WHOEVER'S UNITS ARRIVE FIRST SELLS FIRST and the other sells
//      into the tanked price.
//
//      BOTH PLAYERS SELL ON THE SAME MORNING whenever their units arrive the
//      same day, and that case is reproduced exactly: process_market pairs
//      p0's i-th order with p1's i-th order and walks each pair unit by unit,
//      both quoted before either commits. On the same product at the same
//      index the two walk the price down together and split it; at DIFFERENT
//      indices (one also sold wheat and carrots that morning, so its melons
//      sit one slot later) the earlier index clears first and the later one
//      sells into what is left -- even though it is the same hour.
//
//      The replay SNAPSHOTS every morning: each player's cash straight after
//      the hour-0 market, the money that morning's sells brought in, and all
//      nine prices. Those are the per-day horizon features.
//
//   3. SOLO REPLAYS. Each player's stream replayed with the opponent absent.
//      solo - joint is exactly how much the opponent's timing costs.
//
// NOTHING HIDDEN IS READ
//   There is no committed fertilizer plan on a tile any more -- the fertilize
//   head decides day by day -- so both players run the same canonical
//   fertilize schedule (fert_day_for, gated by fertilize_pays) off public tile
//   state. The streams are whole-unit and deterministic.
//
// THE CANONICAL POLICY (both players, identical)
//   plants   watered whenever the must tier would (a future, or a yield
//            window); fertilized on the canonical schedule where it pays;
//            ongoing crops harvested every
//            morning there is something on them; non-ongoing crops harvested
//            the day they fill up or on the must-harvest day, whichever first.
//   animals  fed and cared every day, harvested and collected every morning.
//   selling  everything on arrival, the morning after harvest; wheat keeps
//            MACRO_WHEAT_RESERVE for feed, fertilizer keeps what the next
//            FERT_LOOKAHEAD days' applications need; the last day sells all.
//   inputs   feed wheat and schedule fertilizer come out of stock first and are
//            bought at the walked price when short.
//   labour   the daily crew is priced off the ops the run-off needs.
//   cash     never checked: a buy the referee would refuse for want of money
//            is still counted, so a broke player's cash rows can dip below 0.
// ===========================================================================
#include "game.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace proj {

constexpr int LAST  = NUM_DAYS - 1;   // the liquidation day
constexpr int LATE  = NUM_DAYS;       // slot: sold on LAST, after the hour-0 market
constexpr int SLOTS = NUM_DAYS + 1;
constexpr int HORIZON = 10;           // per-day snapshots: tomorrow .. 10 mornings out

// ---- knobs ----------------------------------------------------------------
constexpr int    ARRIVE_HOUR    = 4;    // when a hand reaches an already-decaying cell
constexpr double HOURS_PER_OP   = 1.8;  // one op plus its share of the walking
constexpr double WORK_HOURS     = 22.0; // hours a hand has after the morning orders
constexpr int    FERT_LOOKAHEAD = 3;    // days of schedule fertilizer kept back from sale
constexpr int    LIQ_CREW_MAX   = 12;   // macro_liquidate_day's cap
constexpr double EPS = 1e-6;

// ===========================================================================
// 1. TILE RUN-OFF
// ===========================================================================
// `w` scales every emission and need. Every tile runs at w = 1 now; the
// weight is kept so the replay's fractional arithmetic is unchanged.
struct Stream {
    double arrive[SLOTS][NUM_PRODUCTS] = {};
    double wheat_need[NUM_DAYS] = {};
    double fert_need[NUM_DAYS]  = {};
    double ops[NUM_DAYS]        = {};
    double w = 1.0;

    // Harvested on day h -> in the shed at the nightly sweep -> sold at hour 0
    // of h+1. Harvested on the last day -> sold by the liquidation that day.
    void emit(int item, int n, int harvest_day) {
        if (n <= 0 || item < 0 || item >= NUM_PRODUCTS) return;
        if (harvest_day < 0 || harvest_day > LAST) return;
        arrive[harvest_day < LAST ? harvest_day + 1 : LATE][item] += w * n;
    }
    void op(int d)   { ops[d] += w; }
};

// Units decay_plants takes between hour 0 and the hand's arrival. Only a cell
// ALREADY decaying at hour 0 can lose anything.
inline int decay_before(const tile& t, int turn0, int hour) {
    const int mls = t.max_lifespan_step;
    if (mls < 0) return 0;
    const int lo = std::max(mls, turn0), hi = turn0 + hour;
    int n = 0;
    for (int T = lo; T < hi; ++T) if ((T - mls) % 2 == 0) ++n;
    return n;
}

inline void run_plant(tile t, int day, int turn0, Stream& s) {
    for (int d = day; d <= LAST; ++d) {
        if (t.type != T_PLANT || t.crop < 0 || t.crop >= NUM_CROPS) return;
        const CropDef& cd = CROPS[t.crop];
        const int age = d - t.phase;

        if (d == day) {
            const int lost = decay_before(t, turn0, ARRIVE_HOUR);
            if (lost > 0) {
                t.yield_units -= lost;
                if (t.yield_units <= 0) return;      // a weed before anyone arrives
            }
        }

        // ---- ongoing: bank whatever last night produced -------------------
        if (cd.ongoing) {
            if (t.yield_units > 0 && age >= cd.first_yield_day) {
                s.emit(t.crop, t.yield_units, d);
                s.op(d);
                t.yield_units = 0;
            }
            if (t.max_lifespan_step >= 0) return;   // last production night is past
        }

        // ---- the canonical fertilize schedule, where it pays ---------------
        if (fert_day_for(t.crop, t, d) && fertilize_pays(t, d)) {
            t.fertilized = std::max(t.fertilized, d + 2);
            s.fert_need[d] += s.w;
            s.op(d);
        }

        // ---- water: life support, the yield water, the fertilizer bank ----
        bool water;
        if (d == LAST)   // the liquidation only waters what it can harvest
            water = !cd.ongoing && age >= cd.first_yield_day && water_yields_now(t, d);
        else
            water = plant_has_future(t, d) || water_yields_now(t, d);
        if (water && !t.watered_today) {
            t.watered_today = 1;
            s.op(d);
            if (!cd.ongoing && age >= (cd.max_yield_day + 1) / 2 && age <= cd.max_yield_day) {
                const int bonus = (t.fertilized >= d) ? 2 : 1;
                t.yield_units = std::min(cd.max_yield, t.yield_units + bonus);
            }
        }

        // ---- non-ongoing: harvest when full or due; the harvest wipes -----
        if (!cd.ongoing) {
            const bool full = t.yield_units >= cd.max_yield;
            const bool due  = age >= cd.max_yield_day || d == LAST;
            if ((full || due) && age >= cd.first_yield_day) {
                s.emit(t.crop, t.yield_units, d);
                s.op(d);
                return;
            }
            if (due) return;                        // too young on the last day
        }
        if (d == LAST) return;

        // ---- the night: daily_refresh_plants ------------------------------
        const bool was = t.watered_today != 0;
        t.consecutive_unwatered = was ? 0 : t.consecutive_unwatered + 1;
        t.watered_today = 0;
        if (t.consecutive_unwatered >= 2) return;   // weeds
        if (!cd.ongoing) continue;

        const int next  = d + 1;
        const int since = next - t.phase - cd.first_yield_day;
        if (since < 0 || cd.interval <= 0 || since % cd.interval != 0) continue;
        const int count = since / cd.interval + 1;
        if (count > cd.max_yield) continue;
        const bool fert = was && t.fertilized >= d;
        t.yield_units = std::min(cd.max_yield, t.yield_units + (fert ? 2 : 1));
        if (count == cd.max_yield) t.max_lifespan_step = (next + 1) * TURNS_PER_DAY;
    }
}

inline void run_animal(tile t, int day, Stream& s) {
    for (int d = day; d <= LAST; ++d) {
        if (t.animal < 0 || t.animal >= NUM_ANIMALS) return;
        const AnimalDef& a = ANIMALS[t.animal];

        if (t.fertilizer_available) {
            s.emit(FERTILIZER, 1, d);
            s.op(d);
            t.fertilizer_available = 0;
        }
        if (t.yield_units > 0) {
            s.emit(a.product, t.yield_units, d);
            s.op(d);
            t.yield_units = 0;
        }
        if (d == LAST) return;

        if (!t.fed_today)   { t.fed_today = 1; s.wheat_need[d] += s.w; s.op(d); }
        if (!t.cared_today) { t.cared_today = 1; s.op(d); }

        // ---- the night: daily_refresh_animals, always fed -----------------
        t.consecutive_unfed = 0;
        const int since = (d + 1) - t.phase - a.first_yield_day;
        if (since >= 0 && a.interval > 0 && since % a.interval == 0) {
            t.yield_units = std::min(a.max_held, t.yield_units + 1 + t.pending_care_bonus);
            t.pending_care_bonus = 0;
        }
        t.pending_care_bonus += 1;                  // cared and fed tonight
        t.fertilizer_available = 1;
        t.fed_today = t.cared_today = 0;
    }
}

inline void build_stream(const player_state& ps, int day, int turn0, Stream& s) {
    s = Stream{};
    if (day < 0 || day > LAST) return;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const tile& t = ps.board[y][x];
            if (!t.bought) continue;
            if (t.type == T_PLANT) {
                s.w = 1.0;
                run_plant(t, day, turn0, s);
            } else if (t.animal >= 0) {
                s.w = 1.0;
                run_animal(t, day, s);
            }
        }
}

// ===========================================================================
// 2. THE SHARED MARKET
// ===========================================================================
enum OrderKind { O_NOP = 0, O_SELL, O_BUY };   // O_NOP: a hire, holds an index

struct Order { int kind = O_NOP; int item = -1; int n = 0; };

// Expected extra drain per shop tick from ONE future unlock.
inline const std::array<double, NUM_PRODUCTS>& mean_unlock_tick() {
    static const std::array<double, NUM_PRODUCTS> T = [] {
        std::array<double, NUM_PRODUCTS> a{};
        for (int s = 0; s < NUM_SHOPS; ++s) {
            const double mult = (SHOP_PRODUCTS[s].size() == 1) ? 2.0 : 1.0;
            for (int item : SHOP_PRODUCTS[s]) a[item] += mult / NUM_SHOPS;
        }
        return a;
    }();
    return T;
}

struct Ledger {
    double sold_money[2][NUM_PRODUCTS] = {};
    int    sold_units[2][NUM_PRODUCTS] = {};
    double buy_cost[2]   = {};
    double labor_cost[2] = {};

    // ---- per-morning snapshots, indexed by day; LATE = after liquidation --
    bool   snap = false;
    double cash[SLOTS][2]  = {};       // money straight after that market
    double sales[SLOTS][2] = {};       // money that market's sells brought in
    int    price[SLOTS][NUM_PRODUCTS] = {};

    double sold_total(int p) const {
        double s = 0.0;
        for (int k = 0; k < NUM_PRODUCTS; ++k) s += sold_money[p][k];
        return s;
    }
};

struct Market {
    double inv[NUM_PRODUCTS];
    double tick[NUM_PRODUCTS];   // units drained per shop tick (every 4 turns)
    int    shops = 0;

    int inv_i(int k) const { return (int)std::lround(inv[k]); }
    void shop_ticks(int n) {
        for (int k = 0; k < NUM_PRODUCTS; ++k) inv[k] -= tick[k] * n;
    }
    void center_tick() {
        for (int k = 0; k < NUM_PRODUCTS; ++k) if (k != FERTILIZER) inv[k] -= 1.0;
    }
};

// process_market for one hour: order i of p0 against order i of p1, each pair
// walked unit by unit, BOTH QUOTED BEFORE EITHER COMMITS. This is the same-day
// sale: two sells of one product at one index split the walk between them.
inline void run_hour(Market& m, const std::vector<Order> ord[2],
                     size_t lo, size_t hi, Ledger& L) {
    for (size_t i = lo; i < hi; ++i) {
        Order o[2];
        int left[2] = { 0, 0 };
        for (int p = 0; p < 2; ++p)
            if (i < ord[p].size() && ord[p][i].kind != O_NOP) {
                o[p] = ord[p][i];
                left[p] = o[p].n;
            }
        while (left[0] > 0 || left[1] > 0) {
            int price[2] = { 0, 0 };
            for (int p = 0; p < 2; ++p) {
                if (left[p] <= 0) continue;
                price[p] = (o[p].kind == O_SELL)
                    ? market_price(o[p].item, m.inv_i(o[p].item))
                    : market_price(o[p].item, m.inv_i(o[p].item) - 1);
            }
            for (int p = 0; p < 2; ++p) {
                if (left[p] <= 0) continue;
                const int k = o[p].item;
                if (o[p].kind == O_SELL) {
                    L.sold_money[p][k] += price[p];
                    L.sold_units[p][k] += 1;
                    if (price[p] > PRICE_FLOOR) m.inv[k] += 1.0;
                } else {
                    L.buy_cost[p] += price[p];
                    m.inv[k] -= 1.0;
                }
                --left[p];
            }
        }
    }
}

// One hour-0 market with its order chunking: chunk c goes out at hour c, and
// the hour-0 town tick runs between chunk 0 and chunk 1. Returns shop ticks used.
inline int run_morning(Market& m, const std::vector<Order> ord[2], Ledger& L) {
    const size_t n = std::max(ord[0].size(), ord[1].size());
    const size_t C = MAX_MARKET_ORDERS_PER_TURN;
    run_hour(m, ord, 0, std::min(n, C), L);
    m.shop_ticks(1);
    m.center_tick();
    for (size_t lo = C; lo < n; lo += C) run_hour(m, ord, lo, std::min(n, lo + C), L);
    return 1;
}

inline int fib_cost(int hires) {             // do_hire's fib(0) + ... + fib(hires-1)
    int a = 1, b = 1, sum = 0;
    for (int i = 0; i < hires; ++i) { sum += a; const int t = a + b; a = b; b = t; }
    return FARM_HAND_COST_MULT * sum;
}

inline int crew_for(double ops, bool last_day) {
    const int need = (int)std::ceil(ops * HOURS_PER_OP / WORK_HOURS - EPS);
    return last_day ? std::clamp(need, 1, LIQ_CREW_MAX)
                    : std::clamp(need, 1, MACRO_HIRE_HARD_MAX);   // no hire-head floor any more
}

// Replay the streams of the ACTIVE players against one shared market.
inline void replay(const Stream s[2], const bool active[2],
                   const player_state* ps, const market_state& mk,
                   const std::vector<int>& shops, int day, bool snap, Ledger& L) {
    L = Ledger{};
    L.snap = snap;
    Market m;
    for (int k = 0; k < NUM_PRODUCTS; ++k) { m.inv[k] = mk.inventory[k]; m.tick[k] = 0.0; }
    for (int sh : shops) {
        const double mult = (SHOP_PRODUCTS[sh].size() == 1) ? 2.0 : 1.0;
        for (int item : SHOP_PRODUCTS[sh]) m.tick[item] += mult;
    }
    m.shops = (int)shops.size();
    const auto& unlock = mean_unlock_tick();
    const int ticks_per_day = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL;

    // Stock is kept as double (the replay was written for weighted streams);
    // whole units are sold, any remainder waits for the next morning.
    double stock[2][NUM_PRODUCTS];
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            stock[p][k] = active[p] ? ps[p].shed[k] : 0.0;

    auto take_snap = [&](int slot, const double before[2]) {
        if (!L.snap) return;
        for (int p = 0; p < 2; ++p) {
            const double sold = L.sold_total(p);
            L.cash[slot][p]  = ps[p].money + sold - L.buy_cost[p] - L.labor_cost[p];
            L.sales[slot][p] = sold - before[p];
        }
        for (int k = 0; k < NUM_PRODUCTS; ++k) L.price[slot][k] = market_price(k, m.inv_i(k));
    };

    std::vector<Order> ord[2];
    for (int D = day; D <= LAST; ++D) {
        const bool last = (D == LAST);

        // ---- hour 0: sells (product order), hires, then input buys --------
        for (int p = 0; p < 2; ++p) {
            ord[p].clear();
            if (!active[p]) continue;
            const Stream& st = s[p];

            double fert_ahead = 0.0;
            for (int e = D; e <= std::min(LAST, D + FERT_LOOKAHEAD); ++e)
                fert_ahead += st.fert_need[e];

            for (int k = 0; k < NUM_PRODUCTS; ++k) {
                stock[p][k] += st.arrive[D][k];
                double reserve = 0.0;
                if (!last && k == WHEAT)      reserve = MACRO_WHEAT_RESERVE;
                if (!last && k == FERTILIZER) reserve = fert_ahead;
                const int n = (int)std::floor(stock[p][k] - reserve + EPS);
                if (n > 0) { ord[p].push_back({ O_SELL, k, n }); stock[p][k] -= n; }
            }

            const int crew = crew_for(st.ops[D], last);
            L.labor_cost[p] += fib_cost(crew - 1);
            // One market entry PER HAND, every day: the referee takes one hand
            // per HIRE (the liquidation used to be priced as one A_HIRE x n).
            for (int h = 1; h < crew; ++h) ord[p].push_back({});

            const double need[2] = { st.wheat_need[D], st.fert_need[D] };
            const int  items[2] = { WHEAT, FERTILIZER };
            for (int j = 0; j < 2; ++j) {
                const int k = items[j];
                const int nb = (int)std::ceil(need[j] - stock[p][k] - EPS);
                if (nb > 0) { ord[p].push_back({ O_BUY, k, nb }); stock[p][k] += nb; }
                stock[p][k] -= need[j];
            }
        }
        const double before[2] = { L.sold_total(0), L.sold_total(1) };
        const int ticked = run_morning(m, ord, L);
        take_snap(D, before);

        if (!last) {
            m.shop_ticks(ticks_per_day - ticked);
            // end_of_day of D unlocks a shop that is live from D+1.
            if ((D + 1) % TOWN_SHOP_UNLOCK_INTERVAL == 0 && m.shops < MAX_SHOP_INSTANCES) {
                for (int k = 0; k < NUM_PRODUCTS; ++k) m.tick[k] += unlock[k];
                ++m.shops;
            }
            continue;
        }

        // ---- the liquidation's own harvests, sold mid-day -----------------
        m.shop_ticks(ticks_per_day / 2 - ticked);
        for (int p = 0; p < 2; ++p) {
            ord[p].clear();
            if (!active[p]) continue;
            for (int k = 0; k < NUM_PRODUCTS; ++k) {
                const int n = (int)std::lround(stock[p][k] + s[p].arrive[LATE][k]);
                if (n > 0) ord[p].push_back({ O_SELL, k, n });
                stock[p][k] = 0.0;
            }
        }
        const double before_late[2] = { L.sold_total(0), L.sold_total(1) };
        run_hour(m, ord, 0, std::max(ord[0].size(), ord[1].size()), L);
        take_snap(LATE, before_late);
    }
}

// ===========================================================================
// 3. THE RESULT
// ===========================================================================
struct Result {
    int    viewer = 0;                       // whose knowledge the projection uses
    double money[2]    = {};                 // now
    double terminal[2] = {};                 // projected money on the last night
    double proceeds[2][NUM_PRODUCTS] = {};   // joint-market sale money
    int    units[2][NUM_PRODUCTS]    = {};
    double input_cost[2] = {};               // wheat and fertilizer bought
    double labor_cost[2] = {};
    double joint_sales[2] = {};
    double solo_sales[2]  = {};              // same stream, opponent absent

    // ---- the horizon: h = 0 is TOMORROW morning, h = 9 ten mornings out ---
    // Past the last day every row holds the final, post-liquidation state.
    double h_cash[HORIZON][2]  = {};         // money straight after that market
    double h_sales[HORIZON][2] = {};         // what that morning's sells raised
    int    h_price[HORIZON][NUM_PRODUCTS] = {};
};

inline Result project(const player_state* ps, const market_state& mk,
                      const std::vector<int>& shops, int day, int turn0,
                      int viewer, bool with_solo = true) {
    Result r;
    r.viewer = viewer;
    Stream s[2];
    for (int p = 0; p < 2; ++p)
        build_stream(ps[p], day, turn0, s[p]);

    Ledger J;
    const bool both[2] = { true, true };
    replay(s, both, ps, mk, shops, day, /*snap=*/true, J);

    for (int p = 0; p < 2; ++p) {
        r.money[p] = ps[p].money;
        for (int k = 0; k < NUM_PRODUCTS; ++k) {
            r.proceeds[p][k] = J.sold_money[p][k];
            r.units[p][k]    = J.sold_units[p][k];
        }
        r.joint_sales[p] = J.sold_total(p);
        r.input_cost[p]  = J.buy_cost[p];
        r.labor_cost[p]  = J.labor_cost[p];
        r.terminal[p]    = ps[p].money + r.joint_sales[p] - J.buy_cost[p] - J.labor_cost[p];
        r.solo_sales[p]  = r.joint_sales[p];
    }
    for (int h = 0; h < HORIZON; ++h) {
        const int D = day + 1 + h;
        const int slot = (D <= LAST) ? D : LATE;
        for (int p = 0; p < 2; ++p) {
            r.h_cash[h][p]  = (D <= LAST) ? J.cash[slot][p] : r.terminal[p];
            r.h_sales[h][p] = (D <= LAST) ? J.sales[slot][p] : 0.0;
        }
        for (int k = 0; k < NUM_PRODUCTS; ++k) r.h_price[h][k] = J.price[slot][k];
    }
    if (with_solo) {
        for (int p = 0; p < 2; ++p) {
            Ledger S;
            const bool only[2] = { p == 0, p == 1 };
            replay(s, only, ps, mk, shops, day, /*snap=*/false, S);
            r.solo_sales[p] = S.sold_total(p);
        }
    }
    return r;
}

// ===========================================================================
// FEATURES  --  PROJ_FEATURES floats, from the viewer's point of view
// ===========================================================================
// Differences are OWN MINUS OPPONENT here (the older critic features use
// opponent minus own; the sign is learnable either way, but do not mix them up
// when reading the numbers).
constexpr int PROJ_SUMMARY  = 50;
constexpr int PROJ_PER_DAY  = 5 + NUM_PRODUCTS;                  // 14
constexpr int PROJ_FEATURES = PROJ_SUMMARY + HORIZON * PROJ_PER_DAY;   // 190
static_assert(PROJ_FEATURES == PROJ_DIM,
              "game.hpp's PROJ_DIM must match the projection feature count");

template <class WriteFn>
inline void write_features(const Result& r, WriteFn&& W) {
    const int me = r.viewer, op = 1 - me;
    const double tm = r.terminal[me], to = r.terminal[op], diff = tm - to;

    // ---- summary (50) ------------------------------------------------------
    W((float)std::clamp(tm / 20000.0, -0.5, 5.0));                     // 8
    W((float)(tm / 100000.0));
    W((float)std::clamp(to / 20000.0, -0.5, 5.0));
    W((float)(to / 100000.0));
    W((float)std::clamp(diff /  1000.0, -1.0, 1.0));
    W((float)std::clamp(diff / 10000.0, -1.0, 1.0));
    W((float)(diff / 100000.0));
    W((float)std::tanh(diff / 3000.0));

    for (int side : { me, op })                                          // 18
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W((float)std::clamp(r.proceeds[side][k] / 10000.0, 0.0, 4.0));

    // Realised price per unit over base: being SECOND to market shows here.
    for (int side : { me, op })                                          // 18
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W(r.units[side][k] > 0
                ? (float)std::clamp(r.proceeds[side][k] / r.units[side][k]
                                    / MARKET[k].base, 0.0, 3.0)
                : 0.0f);

    for (int side : { me, op })                                          // 2
        W((float)std::clamp((r.solo_sales[side] - r.joint_sales[side]) / 2000.0, -1.0, 2.0));
    for (int side : { me, op })                                          // 2
        W((float)std::clamp((r.input_cost[side] + r.labor_cost[side]) / 10000.0, 0.0, 4.0));
    for (int side : { me, op })                                          // 2
        W((float)std::clamp((r.terminal[side] - r.money[side]) / 20000.0, -1.0, 5.0));

    // ---- per day, tomorrow .. ten mornings out (10 x 14) ------------------
    // Cash is written as the GAIN over today's money, so an early-game 3000
    // and a late-game 40000 both land in range and the rows stay comparable.
    // Scales were set from a real episode, where money runs to ~80k and a
    // late morning's sells alone can raise 30k.
    for (int h = 0; h < HORIZON; ++h) {
        const double gm = r.h_cash[h][me] - r.money[me];
        const double go = r.h_cash[h][op] - r.money[op];
        W((float)std::clamp(gm / 10000.0, -1.0, 6.0));
        W((float)std::clamp(go / 10000.0, -1.0, 6.0));
        W((float)std::tanh((r.h_cash[h][me] - r.h_cash[h][op]) / 3000.0));
        W((float)std::clamp(r.h_sales[h][me] / 10000.0, 0.0, 4.0));
        W((float)std::clamp(r.h_sales[h][op] / 10000.0, 0.0, 4.0));
        for (int k = 0; k < NUM_PRODUCTS; ++k)
            W((float)std::clamp(r.h_price[h][k] / MARKET[k].base, 0.0, 3.0));
    }
}

}  // namespace proj

#endif  // PROJECTION_HPP