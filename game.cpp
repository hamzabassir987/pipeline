#include "game.hpp"

// ===========================================================================
// THE REFEREE
// ===========================================================================
// Unchanged in substance from the two-player ladder game: the 10-entry market
// cap, the lockstep market, apply_farmer_moves running before process_market,
// the atomic PLANT validation, the nightly refresh. That is deliberate -- the
// macro rework only changes WHO chooses the moves and how often, never which
// moves the referee accepts.
//
// (The path rework's one exception -- A_PLANT recording a fertilizer path from
// `count` and A_FERTILIZE decrementing it -- is gone again with the path
// slots. A_PLANT ignores `count`.)
//
// What was removed:
//   * MODE_LADDER / MODE_DAILY_GOAL and every `mode` read. There is one game.
//   * the six ladder criterion bitmasks and check_winner's rung ladder. The
//     win condition is money on the last night, full stop.
//   * the goal grid, its generator, its scorer and its log render.
//   * action costs and the per-unit cooldown vector. The planner's cost model
//     is one hour per op; a 2-turn action would make that wrong, and the cost
//     model is what capacity checking is priced on.
//   * to_flat / from_flat. Nothing indexes a flat action space any more.
// ===========================================================================

static float randFloat(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(gen);
}

// ===========================================================================
// Static tables
// ===========================================================================
const CropDef CROPS[NUM_CROPS] = {
    //         seed  first  maxday  interval  maxyield  ongoing
    /*WHEAT*/      { 10,  2,  4,  0, 6, false },
    /*CARROT*/     { 20,  2,  3,  0, 4, false },
    /*TOMATO*/     { 50,  8,  8,  1, 4, true  },
    /*STRAWBERRY*/ {100, 10, 10,  2, 4, true  },
    /*MELON*/      { 80, 10, 12,  0, 6, false },
};

const AnimalDef ANIMALS[NUM_ANIMALS] = {
    //        cost  structure    first  interval  maxheld  product
    /*GOOSE*/ { 300, T_COOP,    4, 1, 4, EGG  },
    /*COW*/   { 400, T_PASTURE, 8, 2, 6, MILK },
    /*SHEEP*/ { 500, T_PASTURE, 6, 3, 6, WOOL },
};

const MarketDef MARKET[NUM_PRODUCTS] = {
    /*WHEAT*/      {  25, MARKET_I0, 400, SH_SQRT,   0.80, SH_LOG,    0.20 },
    /*CARROT*/     {  35, MARKET_I0, 450, SH_HINGE,  1.00, SH_SQRT,   0.70 },
    /*TOMATO*/     {  60, MARKET_I0, 200, SH_HINGE,  0.40, SH_SQRT,   0.60 },
    /*STRAWBERRY*/ { 120, MARKET_I0, 100, SH_SQRT,   0.70, SH_LINEAR, 1.60 },
    /*MELON*/      { 250, MARKET_I0, 300, SH_LOG,    0.20, SH_SQ,     3.60 },
    /*EGG*/        {  50, MARKET_I0, 332, SH_HINGE,  0.40, SH_LOG,    0.20 },
    /*MILK*/       { 160, MARKET_I0, 122, SH_SQRT,   0.60, SH_LINEAR, 1.60 },
    /*WOOL*/       { 200, MARKET_I0, 105, SH_LOG,    0.20, SH_SQ,     3.20 },
    /*FERTILIZER*/ { 100, MARKET_I0, 200, SH_LINEAR, 0.40, SH_LINEAR, 0.40 },
};

// Shops, in the same alphabetical order the reference rng draws from.
const std::vector<int> SHOP_PRODUCTS[NUM_SHOPS] = {
    /*BAKERY*/         { EGG, WHEAT },
    /*BRUNCH_SPOT*/    { EGG, WHEAT, STRAWBERRY },
    /*FARMERS_MARKET*/ { WHEAT, CARROT, TOMATO, STRAWBERRY },
    /*ICE_CREAM_SHOP*/ { STRAWBERRY, MILK, WHEAT },
    /*PET_CAFE*/       { CARROT },
    /*PIZZA_SHOP*/     { MILK, TOMATO, WHEAT },
    /*SMOOTHIE_SHOP*/  { STRAWBERRY, MILK },
    /*YARN_STORE*/     { WOOL },
};

// ===========================================================================
// Geometry
// ===========================================================================
int quadrant_of(int x, int y) {
    const int half = BOARD_SIZE / 2;
    if (y < half) return (x < half) ? Q_NW : Q_NE;
    return (x < half) ? Q_SW : Q_SE;
}

const std::array<ipos, 4>& shed_access_tiles() {
    static const int h = BOARD_SIZE / 2;
    static const std::array<ipos, 4> tiles = {
        ipos{h - 1, h - 1}, ipos{h, h - 1}, ipos{h - 1, h}, ipos{h, h}
    };
    return tiles;
}

bool is_shed_adjacent(const ipos& p) {
    for (const ipos& t : shed_access_tiles())
        if (t == p) return true;
    return false;
}

ipos default_spawn() {
    for (const ipos& t : shed_access_tiles())
        if (quadrant_of(t.x, t.y) == Q_NW) return t;
    return ipos{0, 0};
}

// Chebyshev distance to the 2x2 shed block.
int shed_distance(int x, int y) {
    const int h = BOARD_SIZE / 2;
    const int dx = (x < h - 1) ? (h - 1 - x) : (x > h ? x - h : 0);
    const int dy = (y < h - 1) ? (h - 1 - y) : (y > h ? y - h : 0);
    return std::max(dx, dy);
}

// ===========================================================================
// Pricing
// ===========================================================================
double shape_f(Shape f, double x, double T) {
    if (x < 0.0) x = 0.0;
    switch (f) {
        case SH_LINEAR: return x;
        case SH_SQ:     return x * x;
        case SH_SQRT:   return std::sqrt(x);
        case SH_LOG:    return std::log1p(x);
        case SH_LOG10:  return std::log10(1.0 + x);
        case SH_HINGE: {
            if (T <= 0.0) return x;                 // degenerate -> linear
            const double u = x / T;
            const double over = (u > 1.0) ? (u - 1.0) : 0.0;
            return u + HINGE_GAIN * over * over;
        }
    }
    return x;
}

static int market_price_calc(int item, int inventory) {
    const MarketDef& p = MARKET[item];
    double price;
    if (inventory < p.I0) {
        const double amp = p.below_target * p.base / shape_f(p.below, p.T, p.T);
        price = p.base + amp * shape_f(p.below, p.I0 - inventory, p.T);
    } else {
        const double amp = p.above_target * p.base / shape_f(p.above, p.T, p.T);
        price = p.base - amp * shape_f(p.above, inventory - p.I0, p.T);
    }
    // nearbyint under the default rounding mode is half-to-even, matching
    // Python's round().
    const int rounded = static_cast<int>(std::nearbyint(price));
    return std::max(PRICE_FLOOR, rounded);
}

// market_price is a pure function of (item, inventory) and is called millions
// of times a day (every unit of every sell / buy walk, the projection replay,
// net_worth). Inventories stay in a band around I0, so the prices there are
// tabulated once, by market_price_calc itself; anything outside the band, or
// an unknown item, is computed exactly as before. Same numbers either way.
namespace {
constexpr int PRICE_LUT_HALF = 4096;
constexpr int PRICE_LUT_N    = 2 * PRICE_LUT_HALF + 1;
struct PriceLut {
    int lo[NUM_PRODUCTS];
    std::vector<int> v;       // NUM_PRODUCTS * PRICE_LUT_N
    PriceLut() : v((size_t)NUM_PRODUCTS * PRICE_LUT_N) {
        for (int k = 0; k < NUM_PRODUCTS; ++k) {
            lo[k] = (int)MARKET[k].I0 - PRICE_LUT_HALF;
            for (int j = 0; j < PRICE_LUT_N; ++j)
                v[(size_t)k * PRICE_LUT_N + j] = market_price_calc(k, lo[k] + j);
        }
    }
};
const PriceLut& price_lut() {
    static const PriceLut L;   // built on first use; thread-safe (C++11)
    return L;
}
}  // namespace

int market_price(int item, int inventory) {
    if ((unsigned)item < (unsigned)NUM_PRODUCTS) {
        const PriceLut& L = price_lut();
        const unsigned j = (unsigned)(inventory - L.lo[item]);
        if (j < (unsigned)PRICE_LUT_N) return L.v[(size_t)item * PRICE_LUT_N + j];
    }
    return market_price_calc(item, inventory);
}

// A BUY_PRODUCT is quoted at post-buy inventory and commit_unit then decrements
// it, so n units are charged at inv-1, inv-2, ... inv-n. The filter prices
// wheat and fertilizer through here so a budget is the real walked-up cost and
// not n * spot.
double buy_cost_walk(int item, int n, int inventory) {
    double total = 0.0;
    for (int j = 0; j < n; ++j)
        total += (double)market_price(item, inventory - 1 - j);
    return total;
}

// The mirror image: n units sold walk the price DOWN, and a sale at the floor
// adds no supply, so the walk stalls there.
double sell_proceeds(int item, int n, int inventory) {
    double total = 0.0;
    int inv = inventory;
    for (int j = 0; j < n; ++j) {
        const int price = market_price(item, inv);
        total += price;
        if (price > PRICE_FLOOR) inv += 1;
    }
    return total;
}

// ===========================================================================
// player_state
// ===========================================================================
int player_state::shed_total() const {
    return std::accumulate(shed.begin(), shed.end(), 0);
}

bool player_state::quadrant_unlocked(int q) const {
    return std::find(unlocked_quadrants.begin(), unlocked_quadrants.end(), q)
           != unlocked_quadrants.end();
}

// ===========================================================================
// Construction
// ===========================================================================
simulation::simulation() { reset(0); }

simulation::simulation(std::string fname, bool log_game, uint64_t seed)
    : log(log_game), file_name(std::move(fname)) {
    reset(seed);
    if (log) {
        outFile.open(file_name);
        outFile << "# kaggriculture log  board=" << BOARD_SIZE
                << " turns_per_day=" << TURNS_PER_DAY
                << " days=" << NUM_DAYS << " seed=" << seed
                << " tile_fields=16\n";
        write_log();
    }
}

void simulation::reset(uint64_t seed) {
    turn  = 0;
    seed_ = seed;
    unlocked_shops.clear();
    ghost[0] = ghost[1] = false;
    dropped_no_unit = plant_groups_blocked = shed_overflows = 0;

    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0xDEADBEEF));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < NUM_PRODUCTS; ++i) {
        market.inventory[i] = static_cast<int>(MARKET[i].I0);
        market.prices[i]    = static_cast<int>(MARKET[i].base);
    }

    // ---- one loadout, drawn ONCE, applied to both players ----------------
    // Independent draws per player hand one side a material edge before the
    // first decision. The critic can read both sheds, so it would predict the
    // winner from the draw and every advantage in the episode collapses to
    // zero.
    std::array<int, NUM_CROPS> start_seeds{};
    std::array<int, NUM_ITEMS> start_shed{};


    for (int p = 0; p < NUM_PLAYERS; ++p) {
        player_state& ps = players[p];
        ps = player_state{};
        ps.money = STARTING_MONEY;
        ps.shed  = start_shed;
        ps.seeds = start_seeds;
        ps.unlocked_quadrants = { Q_NW };
        ps.farmers.assign(1, default_spawn());
        ps.carried.assign(1, unit_inv{});
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x) {
                ps.board[y][x] = tile{};
                ps.board[y][x].bought = (quadrant_of(x, y) == Q_NW) ? 1 : 0;
            }
    }

    refresh_farmer_marks();
}

double simulation::net_worth(int p) const {
    const player_state& ps = players[p];
    double total = ps.money;
    for (int k = 0; k < NUM_PRODUCTS; ++k) {   // livestock is not sellable
        int n = ps.shed[k];
        for (const unit_inv& u : ps.carried) n += u[k];
        total += sell_proceeds(k, n, market.inventory[k]);
    }
    return total;
}

// ===========================================================================
// Shed deposits
// ===========================================================================
// Moves up to `n` of one item from a unit's hands into the shed, honouring
// SHED_CAPACITY. Shared by DROP-with-item and the PLACE fallback.
void simulation::store_to_shed(int p, int idx, int item, int n) {
    player_state& ps = players[p];
    if (item < 0 || item >= NUM_ITEMS || n <= 0) return;
    unit_inv& inv = ps.carried[idx];
    const int want = std::min(n, inv[item]);
    if (want <= 0) return;
    const int room = std::max(0, SHED_CAPACITY - ps.shed_total());
    const int take = std::min(want, room);
    if (take < want) shed_overflows++;
    if (take <= 0) return;
    inv.sub_to(item, take);
    ps.shed[item] += take;
}

// The dump-everything path, shared by DROP-with-no-item and the end-of-day
// sweep. Walks `inv.keys` -- insertion order -- because when the shed is full
// the iteration order decides which goods survive. Overflow is discarded.
void simulation::dump_inventory(int p, unit_inv& inv) {
    player_state& ps = players[p];
    for (const int item : inv.keys) {
        const int n = inv.n[item];
        if (n <= 0) continue;
        const int room = std::max(0, SHED_CAPACITY - ps.shed_total());
        const int take = std::min(n, room);
        if (take < n) shed_overflows++;
        if (take > 0) ps.shed[item] += take;
    }
    inv.clear();
}

// ===========================================================================
// Unit actions
// ===========================================================================
void simulation::apply_unit_action(int p, int idx, const move& m) {
    player_state& ps = players[p];
    if (idx < 0 || idx >= static_cast<int>(ps.farmers.size())) return;
    if (is_market_action(m.type)) return;          // handled by process_market

    ipos& pos = ps.farmers[idx];
    unit_inv& inv = ps.carried[idx];

    const bool item_ok = (m.item >= 0 && m.item < NUM_ITEMS);

    // ---- movement -------------------------------------------------------
    if (is_move_action(m.type)) {
        static const int DX[4] = { 0, 0, 1, -1 };
        static const int DY[4] = { -1, 1, 0, 0 };
        const int nx = pos.x + DX[m.type];
        const int ny = pos.y + DY[m.type];
        // Movement onto LOCKED tiles is legal: a hand can stand on a locked
        // shed-access tile and blocking it would strand the hand there.
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
            pos.x = nx;
            pos.y = ny;
        }
        return;
    }
    if (m.type == A_PASS) return;

    tile& t = ps.board[pos.y][pos.x];

    // ---- shed ops: resolve before the LOCKED guard ----------------------
    // The shed itself is always owned, and three of its four access tiles
    // start locked, so guarding first would make it unreachable.
    if (m.type == A_DROP) {
        if (!is_shed_adjacent(pos)) return;
        if (m.item < 0) { dump_inventory(p, inv); return; }
        if (!item_ok || m.count <= 0) return;
        store_to_shed(p, idx, m.item, m.count);
        return;
    }

    if (m.type == A_PICKUP) {
        if (!is_shed_adjacent(pos) || !item_ok || m.count <= 0) return;
        // Seeds live in ps.seeds and are consumed straight by PLANT; they never
        // pass through a carried inventory or the shed.
        const int n = std::min(m.count, ps.shed[m.item]);
        if (n <= 0) return;
        ps.shed[m.item] -= n;
        inv.add(m.item, n);
        return;
    }

    if (m.type == A_PLACE) {
        // Animal placement first, otherwise fall through to a shed deposit, so
        // PLACE off a matching structure is a store rather than a no-op.
        const int a_id = (m.item >= GOOSE && m.item < GOOSE + NUM_ANIMALS)
                       ? m.item - GOOSE : -1;
        const bool on_structure = a_id >= 0 &&
            t.type == ANIMALS[a_id].structure && t.animal == -1;
        if (!on_structure) {
            if (item_ok && is_shed_adjacent(pos))
                store_to_shed(p, idx, m.item, std::max(1, m.count));
            return;
        }
        if (!inv.take(m.item, 1)) return;
        ps.stat_plants++;
        report(LE_COST, p, -1, pos.y * BOARD_SIZE + pos.x, -1, 1, ANIMALS[a_id].cost);
        t.animal = a_id;
        t.phase = day();
        t.yield_units = 0;
        t.consecutive_unfed = 0;
        t.fed_today = t.cared_today = 0;
        t.fertilizer_available = 0;
        t.pending_care_bonus = 0;
        return;
    }

    // ---- everything below mutates the tile, so it must be owned ---------
    if (!t.bought) return;

    switch (m.type) {
        case A_WATER: {
            if (t.type != T_PLANT || t.watered_today) return;
            t.watered_today = 1;
            ps.stat_water_feed++;
            const CropDef& cd = CROPS[t.crop];
            if (!cd.ongoing) {
                const int age = day() - t.phase;
                const int window_start = (cd.max_yield_day + 1) / 2;
                if (age >= window_start && age <= cd.max_yield_day) {
                    const int bonus = (t.fertilized >= day()) ? 2 : 1;
                    t.yield_units = std::min(cd.max_yield, t.yield_units + bonus);
                }
            }
            break;
        }
        case A_HARVEST: {
            if (t.yield_units <= 0) return;
            if (t.type == T_PLANT) {
                const CropDef& cd = CROPS[t.crop];
                if (day() - t.phase < cd.first_yield_day) return;
                report(LE_HARVEST, p, t.crop, pos.y * BOARD_SIZE + pos.x, t.phase,
                       t.yield_units, 0);
                inv.add(t.crop, t.yield_units);
                t.yield_units = 0;
                ps.stat_harvests++;
                if (!cd.ongoing) t = tile{ T_EMPTY, 1 };
            } else if (t.animal >= 0) {
                report(LE_HARVEST, p, ANIMALS[t.animal].product,
                       pos.y * BOARD_SIZE + pos.x, t.phase, t.yield_units, 0);
                inv.add(ANIMALS[t.animal].product, t.yield_units);
                t.yield_units = 0;
                ps.stat_harvests++;
            } else {
                return;
            }
            break;
        }
        case A_FERTILIZE: {
            if (t.type != T_PLANT) return;
            if (!inv.take(FERTILIZER, 1)) return;
            report(LE_CONSUME, p, FERTILIZER, pos.y * BOARD_SIZE + pos.x, -1, 1,
                   market_price(FERTILIZER, market.inventory[FERTILIZER]));
            t.fertilized = std::max(t.fertilized, day() + 2);   // day..day+2
            ps.stat_fertilized++;
            break;
        }
        case A_DIG: {
            // Clears plants, weeds and empty structures. Never evicts an animal.
            if (t.type == T_EMPTY) return;
            if (t.animal >= 0) return;
            t = tile{ T_EMPTY, 1 };
            break;
        }
        case A_BUILD_COOP:
        case A_BUILD_PASTURE: {
            if (t.type != T_EMPTY) return;
            t.type = (m.type == A_BUILD_COOP) ? T_COOP : T_PASTURE;
            t.animal = -1;
            break;
        }
        case A_FEED: {
            if (t.animal < 0 || t.fed_today) return;
            if (!inv.take(WHEAT, 1)) return;
            report(LE_CONSUME, p, WHEAT, pos.y * BOARD_SIZE + pos.x, -1, 1,
                   market_price(WHEAT, market.inventory[WHEAT]));
            t.fed_today = 1;
            ps.stat_water_feed++;
            break;
        }
        case A_COLLECT_FERTILIZER: {
            if (t.animal < 0 || !t.fertilizer_available) return;
            t.fertilizer_available = 0;
            report(LE_COLLECT, p, FERTILIZER, pos.y * BOARD_SIZE + pos.x, t.phase, 1, 0);
            inv.add(FERTILIZER, 1);
            break;
        }
        case A_CARE: {
            if (t.animal < 0 || t.cared_today) return;
            t.cared_today = 1;
            ps.stat_water_feed++;
            break;
        }
        default:
            return;   // PLANT is handled by the caller (atomic seed check)
    }
}

void simulation::apply_farmer_moves(std::vector<move> moves[NUM_PLAYERS]) {
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        player_state& ps = players[p];
        const int n_units = static_cast<int>(ps.farmers.size());

        // One action per unit per turn: first request wins.
        std::vector<move> chosen(n_units);
        std::vector<char> taken(n_units, 0);
        for (int i = 0; i < n_units; ++i) chosen[i] = act(i, A_PASS);
        for (const move& m : moves[p]) {
            if (is_market_action(m.type)) continue;
            if (m.farmer_idx < 0 || m.farmer_idx >= n_units) {
                // Addressed to a hand that does not exist here. In macro mode
                // this should stay at zero: the planner only scripts hands it
                // projected, and the hires that create them resolve at hour 0
                // while the script starts at hour 1 or later. A non-zero count
                // means the projection and the referee disagree.
                dropped_no_unit++;
                continue;
            }
            if (taken[m.farmer_idx]) continue;
            taken[m.farmer_idx] = 1;
            chosen[m.farmer_idx] = m;
        }

        // Atomic PLANT validation: if the turn's PLANT requests for a crop
        // exceed the seeds on hand, every PLANT of that crop is dropped.
        // One PLANT is always one seed; `count` is not read.
        int demand[NUM_CROPS] = { 0 };
        {
            std::vector<char> counted(n_units, 0);
            for (const move& m : moves[p]) {
                if (is_market_action(m.type)) continue;
                if (m.type != A_PLANT) continue;
                if (m.item < 0 || m.item >= NUM_CROPS) continue;
                if (m.farmer_idx >= 0 && m.farmer_idx < n_units) {
                    if (counted[m.farmer_idx]) continue;   // one action per unit
                    counted[m.farmer_idx] = 1;
                }
                demand[m.item] += 1;
            }
        }

        // Decided ONCE, against the seed counts at the start of the turn.
        // Evaluating it inside the loop would be wrong: each plant decrements
        // the pool, so the second of two requests against two seeds would
        // compare 2 > 1 and block a group that should have gone through.
        bool blocked[NUM_CROPS];
        int seeds_at_start[NUM_CROPS];
        for (int c = 0; c < NUM_CROPS; ++c) {
            seeds_at_start[c] = ps.seeds[c];
            blocked[c] = demand[c] > seeds_at_start[c];
            if (blocked[c] && demand[c] > 0) plant_groups_blocked++;
        }

        for (int i = 0; i < n_units; ++i) {
            const move& m = chosen[i];
            if (m.type == A_PLANT) {
                const int c = m.item;
                if (c < 0 || c >= NUM_CROPS) continue;
                if (blocked[c]) continue;                  // whole group dropped
                tile& t = ps.board[ps.farmers[i].y][ps.farmers[i].x];
                if (!t.bought || t.type != T_EMPTY) continue;
                if (ps.seeds[c] <= 0) continue;
                ps.seeds[c] -= 1;
                report(LE_COST, p, -1, ps.farmers[i].y * BOARD_SIZE + ps.farmers[i].x,
                       -1, 1, CROPS[c].seed_cost);
                t = tile{ T_PLANT, 1 };
                t.crop = c;
                t.phase = day();
                ps.stat_plants++;
                t.consecutive_unwatered = 1;   // planting day counts as unwatered
                t.yield_units = CROPS[c].ongoing ? 0 : 1;
                t.max_lifespan_step = CROPS[c].ongoing
                    ? -1
                    : (day() + CROPS[c].max_yield_day + 1) * TURNS_PER_DAY;
                continue;
            }
            apply_unit_action(p, i, m);
        }
    }
}

// ===========================================================================
// Market
// ===========================================================================
namespace {
struct OrderState {
    int  kind = A_PASS;
    int  item = -1;
    int  remaining = 0;
    bool active = false;
};

// Validates the operand and seeds the unit counter. Malformed orders never
// become active, so they are silent no-ops.
OrderState open_order(const move& m) {
    OrderState o;
    if (m.count <= 0) return o;
    switch (m.type) {
        case A_SELL:
            if (m.item < 0 || m.item >= NUM_PRODUCTS) return o;
            break;
        case A_BUY_PRODUCT:
            if (m.item != WHEAT && m.item != FERTILIZER) return o;
            break;
        case A_BUY_SEED:
            if (m.item < 0 || m.item >= NUM_CROPS) return o;
            break;
        case A_BUY_ANIMAL:
            if (m.item < 0 || m.item >= NUM_ANIMALS) return o;
            break;
        default:
            return o;
    }
    o.kind = m.type;
    o.item = m.item;
    o.remaining = m.count;
    o.active = true;
    return o;
}
}  // namespace

bool simulation::commit_unit(int p, int kind, int item, int price) {
    player_state& ps = players[p];
    // A ghost (a replayed tape) buys whatever it recorded buying.
    const bool broke = !ghost[p] && ps.money < price;
    switch (kind) {
        case A_SELL:
            if (ps.shed[item] <= 0) return false;
            ps.shed[item] -= 1;
            ps.money += price;
            report(LE_SELL, p, item, -1, -1, 1, price);
            // Sales at the floor price do not add to market supply.
            if (price > PRICE_FLOOR) market.inventory[item] += 1;
            return true;
        case A_BUY_PRODUCT:
            if (broke) return false;
            if (ps.shed_total() >= SHED_CAPACITY) return false;
            ps.money -= price;
            ps.shed[item] += 1;
            market.inventory[item] -= 1;
            report(LE_BUY, p, item, -1, -1, 1, price);
            return true;
        case A_BUY_SEED:
            if (broke) return false;
            ps.money -= price;
            ps.seeds[item] += 1;
            return true;
        case A_BUY_ANIMAL:
            if (broke) return false;
            if (ps.shed_total() >= SHED_CAPACITY) return false;
            ps.money -= price;
            ps.shed[animal_item(item)] += 1;
            return true;
        default:
            return false;
    }
}

int simulation::quote_unit(int kind, int item) const {
    switch (kind) {
        case A_SELL:
            return market_price(item, market.inventory[item]);
        case A_BUY_PRODUCT:
            // Quoted at post-buy inventory, so a buy/sell round trip against
            // an unchanged market nets zero.
            return market_price(item, market.inventory[item] - 1);
        case A_BUY_SEED:   return CROPS[item].seed_cost;
        case A_BUY_ANIMAL: return ANIMALS[item].cost;
        default:           return 0;
    }
}

void simulation::process_market(std::vector<move> moves[NUM_PLAYERS]) {
    std::vector<move> queue[NUM_PLAYERS];
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        // THE REFEREE'S RULE: one HIRE / BUY_LAND per entry, and every entry
        // is one of the MAX_MARKET_ORDERS_PER_TURN slots. A counted A_HIRE
        // (or A_BUY_LAND) is expanded in place into single entries; whatever
        // falls past the cap is dropped, as the referee's q[:max_orders] does.
        bool full = false;
        for (const move& m : moves[p]) {
            if (m.farmer_idx != MARKET_ACTOR || !is_market_action(m.type)) continue;
            const bool atomic = m.type == A_HIRE || m.type == A_BUY_LAND;
            const int reps = atomic ? std::max(1, m.count) : 1;
            for (int r = 0; r < reps; ++r) {
                if (static_cast<int>(queue[p].size()) >= MAX_MARKET_ORDERS_PER_TURN) {
                    full = true;
                    break;
                }
                move e = m;
                if (atomic) e.count = 1;
                queue[p].push_back(e);
            }
            if (full) break;
        }
    }

    const size_t n_orders = std::max(queue[0].size(), queue[1].size());
    for (size_t i = 0; i < n_orders; ++i) {
        OrderState ord[NUM_PLAYERS];

        // Atomic orders resolve once, in player order, before anything is quoted.
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (i >= queue[p].size()) continue;
            const move& m = queue[p][i];
            if (m.type == A_HIRE) {
                for (int k = 0; k < std::max(1, m.count); ++k) do_hire(p);
            } else if (m.type == A_BUY_LAND) {
                for (int k = 0; k < std::max(1, m.count); ++k) do_buy_land(p);
            } else {
                ord[p] = open_order(m);
            }
        }

        // Per-unit lockstep. Both players are quoted against the same
        // pre-commit inventory, then both commit, so neither can front-run the
        // other inside a unit no matter how large their order is.
        for (int guard = 0; ; ++guard) {
            if (guard >= 100000) {
                std::fprintf(stderr,
                             "WARNING: market loop exceeded 100k units; aborting\n");
                break;
            }
            int price[NUM_PLAYERS] = { 0, 0 };
            bool quoted[NUM_PLAYERS] = { false, false };
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                if (!ord[p].active || ord[p].remaining <= 0) continue;
                price[p] = quote_unit(ord[p].kind, ord[p].item);
                quoted[p] = true;
            }
            if (!quoted[0] && !quoted[1]) break;

            bool committed_any = false;
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                if (!quoted[p]) continue;
                if (commit_unit(p, ord[p].kind, ord[p].item, price[p])) {
                    ord[p].remaining -= 1;
                    committed_any = true;
                } else {
                    ord[p].active = false;   // can't fund / can't fill: order ends
                }
            }
            if (!committed_any) break;
        }
        refresh_prices();
    }
}

static int fib(int n) {   // fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { const int t = a + b; a = b; b = t; }
    return a;
}

ipos simulation::spawn_hand(int p) const {
    const player_state& ps = players[p];
    const auto& tiles = shed_access_tiles();
    int occ[4] = { 0, 0, 0, 0 };
    for (const ipos& f : ps.farmers)
        for (int i = 0; i < 4; ++i)
            if (tiles[i] == f) occ[i] += 1;
    int best = 0;
    for (int i = 1; i < 4; ++i)
        if (occ[i] < occ[best]) best = i;   // ties keep the earlier NWSE index
    return tiles[best];
}

void simulation::do_hire(int p) {
    player_state& ps = players[p];
    if ((int)ps.farmers.size() >= 16) return;
    const double cost = FARM_HAND_COST_MULT * static_cast<double>(fib(ps.hires_today));
    if (!ghost[p] && ps.money < cost) return;
    ps.money -= cost;
    ps.hires_today += 1;
    ps.farmers.push_back(spawn_hand(p));
    ps.carried.push_back(unit_inv{});
}

void simulation::do_buy_land(int p) {
    player_state& ps = players[p];
    const int extra = static_cast<int>(ps.unlocked_quadrants.size()) - 1;  // NW is free
    if (extra >= 3) return;
    const double cost = LAND_PRICES[extra];
    if (!ghost[p] && ps.money < cost) return;
    ps.money -= cost;
    const int q = LAND_ORDER[extra];
    ps.unlocked_quadrants.push_back(q);
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x)
            if (quadrant_of(x, y) == q) ps.board[y][x].bought = 1;
}

void simulation::refresh_prices() {
    for (int i = 0; i < NUM_PRODUCTS; ++i)
        market.prices[i] = market_price(i, market.inventory[i]);
}

// ===========================================================================
// Town
// ===========================================================================
void simulation::town_consume() {
    if (turn % TOWN_SHOP_SELL_INTERVAL == 0) {
        // unlocked_shops may hold duplicates; each instance consumes on its own.
        for (int s : unlocked_shops) {
            const std::vector<int>& prods = SHOP_PRODUCTS[s];
            const int mult = (prods.size() == 1) ? 2 : 1;
            for (int item : prods) market.inventory[item] -= mult;
        }
    }

    if (turn % TOWN_CENTER_SELL_INTERVAL == 0) {
        for (int item = 0; item < NUM_PRODUCTS; ++item)
            if (item != FERTILIZER) market.inventory[item] -= 1;
    }
    refresh_prices();
}

// ===========================================================================
// Plant decay and the daily tick
// ===========================================================================
void simulation::decay_plants(int p) {
    player_state& ps = players[p];
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            tile& t = ps.board[y][x];
            if (t.type != T_PLANT) continue;
            const int mls = t.max_lifespan_step;
            if (mls < 0 || turn < mls) continue;
            if ((turn - mls) % 2 != 0) continue;
            t.yield_units -= 1;
            if (t.yield_units <= 0) t = tile{ T_WEED, 1 };
        }
}

void simulation::daily_refresh_plants(int p) {
    player_state& ps = players[p];
    const int cur = day();
    const int next = cur + 1;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            tile& t = ps.board[y][x];
            if (t.type != T_PLANT) continue;

            const bool was_watered = t.watered_today != 0;
            t.consecutive_unwatered = was_watered ? 0 : t.consecutive_unwatered + 1;
            t.watered_today = 0;
            // The plant dies: a T_WEED tile is a fresh tile{}.
            if (t.consecutive_unwatered >= 2) { t = tile{ T_WEED, 1 }; continue; }

            const CropDef& cd = CROPS[t.crop];
            if (!cd.ongoing) continue;

            const int since = next - t.phase - cd.first_yield_day;
            if (since < 0) continue;
            if (cd.interval <= 0 || since % cd.interval != 0) continue;
            const int production_count = since / cd.interval + 1;
            if (production_count > cd.max_yield) continue;

            // Fertilizer only pays out on a watered day: basic needs first.
            // THIS IS THE READ SITE must_bank_water exists to protect.
            const bool fertilized = was_watered && t.fertilized >= cur;
            t.yield_units = std::min(cd.max_yield, t.yield_units + (fertilized ? 2 : 1));
            if (production_count == cd.max_yield)
                t.max_lifespan_step = (next + 1) * TURNS_PER_DAY;
        }
}

void simulation::daily_refresh_animals(int p) {
    player_state& ps = players[p];
    const int next = day() + 1;
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            tile& t = ps.board[y][x];
            if (t.animal < 0) continue;

            t.consecutive_unfed = t.fed_today ? 0 : t.consecutive_unfed + 1;
            if (t.consecutive_unfed >= 2) {
                // The animal escapes; the structure survives.
                const int structure = ANIMALS[t.animal].structure;
                t = tile{ structure, 1 };
                t.animal = -1;
                continue;
            }

            const AnimalDef& a = ANIMALS[t.animal];
            const int since = next - t.phase - a.first_yield_day;
            if (since >= 0 && a.interval > 0 && since % a.interval == 0) {
                // A care bonus is only cashed in on a fed production day.
                const int bonus = t.fed_today ? t.pending_care_bonus : 0;
                t.yield_units = std::min(a.max_held, t.yield_units + 1 + bonus);
                t.pending_care_bonus = 0;
            }
            if (t.cared_today && t.fed_today) t.pending_care_bonus += 1;

            t.fertilizer_available = 1;
            t.fed_today = 0;
            t.cared_today = 0;
        }
}

void simulation::spawn_weeds(int p, std::mt19937& rng) {
    // A ghost's weeds come from its recording (tape::sync), not from here.
    if (ghost[p]) return;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    player_state& ps = players[p];
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            tile& t = ps.board[y][x];
            if (t.type == T_EMPTY && t.bought && uni(rng) < weed_chance)
                t = tile{ T_WEED, 1 };
        }
}

void simulation::drop_inventories_to_shed(int p) {
    player_state& ps = players[p];
    // Units in order, and within each unit, items in acquisition order.
    for (unit_inv& inv : players[p].carried) dump_inventory(p, inv);
    (void)ps;
}

void simulation::end_of_day() {
    const int d = day();
    // Seeded off (seed, day) so replays reproduce.
    std::mt19937 rng(static_cast<uint32_t>((seed_ * 1000003ULL) ^ static_cast<uint64_t>(d)));

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        daily_refresh_plants(p);
        daily_refresh_animals(p);
        spawn_weeds(p, rng);
        drop_inventories_to_shed(p);

        // The crew does not survive the night: every day starts with one hand
        // at the shed and hires_today back at zero, which is why the macro
        // hire head names a TARGET TOTAL and is drawn again every morning.
        // The board survives; the crew is rebuilt daily.
        player_state& ps = players[p];
        ps.farmers.assign(1, default_spawn());
        ps.carried.assign(1, unit_inv{});
        ps.hires_today = 0;
    }

    const int next_day = d + 1;
    if (auto_shops && next_day > 0 && next_day % TOWN_SHOP_UNLOCK_INTERVAL == 0 &&
        static_cast<int>(unlocked_shops.size()) < MAX_SHOP_INSTANCES) {
        // Drawn with replacement: variety is not guaranteed, only the cap.
        std::uniform_int_distribution<int> pick(0, NUM_SHOPS - 1);
        unlocked_shops.push_back(pick(rng));
    }
}

// ===========================================================================
// Bookkeeping
// ===========================================================================
void simulation::refresh_farmer_marks() {
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        player_state& ps = players[p];
        for (int y = 0; y < BOARD_SIZE; ++y)
            for (int x = 0; x < BOARD_SIZE; ++x)
                ps.board[y][x].has_farmer = 0;
        for (const ipos& f : ps.farmers)
            ps.board[f.y][f.x].has_farmer += 1;
    }
}

void simulation::write_player_block(const player_state& ps) {
    outFile << static_cast<long long>(ps.money) << " "
            << ps.farmers.size() << " " << ps.shed_total();
    for (int i = 0; i < NUM_ITEMS; ++i) outFile << " " << ps.shed[i];
    for (int c = 0; c < NUM_CROPS; ++c) outFile << " " << ps.seeds[c];
    outFile << ";";
    // 15 FIELDS PER TILE (fert_plan is gone again). fertilized stays the raw
    // deadline day, not a flag, so the viewer can show how much longer it
    // lasts.
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            const tile& t = ps.board[y][x];
            outFile << t.type << "," << t.bought << "," << t.crop << ","
                    << t.animal << "," << t.yield_units << ","
                    << t.has_farmer << "," << t.watered_today << ","
                    << t.fertilized << "," << t.phase << ","
                    << t.consecutive_unwatered << "," << t.consecutive_unfed << ","
                    << t.fed_today << "," << t.cared_today << ","
                    << t.fertilizer_available << "," << t.pending_care_bonus << " ";
        }
    outFile << ";";
    // one token per unit: x,y then the 12 carried counts
    for (size_t u = 0; u < ps.farmers.size(); ++u) {
        outFile << ps.farmers[u].x << "," << ps.farmers[u].y;
        for (int i = 0; i < NUM_ITEMS; ++i) outFile << "," << ps.carried[u][i];
        outFile << " ";
    }
    outFile << "|";
}

void simulation::write_log() {
    if (!log || !outFile.is_open()) return;
    outFile << turn << "|";
    for (int p = 0; p < NUM_PLAYERS; ++p) write_player_block(players[p]);
    for (int i = 0; i < NUM_PRODUCTS; ++i)
        outFile << market.inventory[i] << "," << market.prices[i] << " ";
    outFile << "|";
    for (int s : unlocked_shops) outFile << s << " ";
    outFile << "\n";
}

void simulation::enable_log(const std::string& path) {
    if (outFile.is_open()) outFile.close();
    outFile.open(path);
    log = outFile.is_open();
    if (!log) return;
    file_name = path;
    outFile << "# kaggriculture log  board=" << BOARD_SIZE
            << " turns_per_day=" << TURNS_PER_DAY
            << " days=" << NUM_DAYS << " seed=" << seed_
            << " tile_fields=16\n";
    write_log();
}

void simulation::log_frame() { write_log(); }

// ===========================================================================
// The turn
// ===========================================================================
int simulation::apply_turn(std::vector<move> moves[NUM_PLAYERS]) {
    if (turn >= MAX_TURNS) return check_winner();

    apply_farmer_moves(moves);
    process_market(moves);
    town_consume();
    for (int p = 0; p < NUM_PLAYERS; ++p) decay_plants(p);

    if ((turn + 1) % TURNS_PER_DAY == 0) end_of_day();

    turn++;
    refresh_farmer_marks();
    if (auto_log) write_log();

    if (turn >= MAX_TURNS) return check_winner();
    return -1;
}

// ===========================================================================
// THE WIN CONDITION
// ===========================================================================
// Most money on the last night. Nothing else -- no behavioural ladder, no
// graded grid score. The daily objective grid is an ACTION now, so scoring it
// would be scoring the policy on a target it chose for itself, and the way to
// maximise that is to ask for an easy day.
//
// Note this reads `money` and not net_worth(): unsold stock is not money, and
// making the policy convert it is part of the problem. net_worth() exists for
// the observation and for logging.
// ===========================================================================
int simulation::check_winner() const {
    if (players[0].money > players[1].money) return 0;
    if (players[1].money > players[0].money) return 1;
    return 2;
}