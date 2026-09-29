#include "game.hpp"
#include "macro_planner.hpp"
#include "tape.hpp"
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <deque>
#include <set>
#include <numeric>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

std::ofstream dbg_file;

// ===========================================================================
// TRAINING
// ===========================================================================
// One decision per day, self-play, two players. The reward is +-1 on who ends
// the 30th day with more money, paid once at the end, so:
//
//   * gamma is 1.0. There is one reward and it arrives at a fixed horizon.
//   * there is no bootstrapping. Every episode is exactly
//     MACRO_DECISION_DAYS decisions and terminates, so the GAE recursion
//     starts from next_value = 0.
//   * an episode contributes 2 * MACRO_DECISION_DAYS samples, not
//     2 * MAX_TURNS and not 2 * NUM_DAYS.
//
// THE LAST DAY IS NOT ONE OF THEM. Day NUM_DAYS-1 is run by
// macro_liquidate_day: harvest everything, carry it in, sell it. No forward
// pass, no draw, no experience. The terminal reward is therefore credited to
// day NUM_DAYS-2, the last state the policy actually chose in -- crediting it
// to the liquidation would ask the critic to regress a value that is already
// determined by arithmetic, and would spend a thirtieth of every rollout's
// gradient on a day with no choice in it.
// ===========================================================================
#define GAMES_AMOUNT 250

// ---- TAPES ----------------------------------------------------------------
// Recorded Kaggle players as opponents (tape.hpp). TAPE_DIR holds .tape files
// made by tapeconv from the zipped episode JSONs (raw .json works too, but
// costs a 32 MB parse each). Every iteration
// TAPES_PER_ITER episodes are drawn from TAPE_DIR and each one's side is
// replayed into p1 of one game. The rest of the games, if any, are self-play
// and pool games as before. TAPES_PER_ITER == GAMES_AMOUNT is all tapes.
// An empty or missing directory falls back to self-play.
static const char* TAPE_DIR       = "./tapes";
constexpr int      TAPES_PER_ITER = 0;
constexpr bool     TAPE_WINNERS_ONLY = false;   // only the side that won its episode

// TAPE ONLY. No opponent pool (no snapshots taken, no GPU slots allocated).
// SELFPLAY_SHARE of the games are live-vs-live self-play, both sides learning;
// the rest are tape games. SELFPLAY_SHARE = 0 is tapes and nothing else.
// Progress is then
// measured two ways, every EVAL_EVERY iterations, both greedy:
//   * vs `frozen`, an old checkpoint of itself, promoted when beaten at
//     >= 85% -- "is it still learning"
//   * vs THE BENCH, TAPE_BENCH_N tapes drawn ONCE at startup with a fixed
//     seed -- "is it getting better against real players". The same games
//     every time, so the curve is comparable across the whole run.
// With TAPE_BENCH_HOLDOUT the bench files are removed from training, so the
// bench measures generalisation rather than memorised tapes. It is only held
// out when the directory has at least 3x the bench size.
constexpr bool     TAPE_ONLY          = false;
constexpr float    SELFPLAY_SHARE     = 1.0f;
constexpr int      TAPE_BENCH_N       = 200;
constexpr bool     TAPE_BENCH_HOLDOUT = true;
constexpr uint32_t TAPE_BENCH_SEED    = 12345;
constexpr int      EVAL_EVERY         = 50;

// ===========================================================================
// THE REWARD
// ===========================================================================
// Every decision day t >= REWARD_START_DAY gets two terms, both measured on
// the morning of t+1 (the last decision day's "next morning" is the end of
// the game, after the liquidation):
//
//   LEAD   +LEAD_REWARD if my MONEY is higher than the opponent's,
//          -LEAD_REWARD if it is lower, 0 on a tie.
//   GAP    how much the day moved the net-worth gap,
//              clamp((GAP(t+1) - GAP(t)) * GAP_SCALE, -GAP_CLAMP, GAP_CLAMP)
//          with GAP = NW_me - NW_opp. Growing faster than the opponent, or
//          making its sales fetch less in the shared market, both count.
//
// Days before REWARD_START_DAY get nothing, so the opening can invest without
// being punished for it. The last decision day also gets END_REWARD: +1 win,
// -1 loss, 0 draw, on money exactly as check_winner decides it.
//
// NW = simulation::net_worth: money plus every shed / carried product at what
// selling it would actually fetch. Net worth for the GAP because buying goods
// and selling them later is neutral in it; LEAD reads money because money is
// what the game is won on.
constexpr int      REWARD_START_DAY = 56;
constexpr float    LEAD_REWARD      = 0.0f;
constexpr double   GAP_SCALE        = 1.0 / 6000000.0;
constexpr float    GAP_CLAMP        = 0.0f;
constexpr float    END_REWARD       = 1.0f;

// MARGIN: on the last decision day, on top of END_REWARD,
//     clamp(MARGIN_COEF * tanh((money_me - money_opp) / MARGIN_SCALE),
//           -MARGIN_CLIP, MARGIN_CLIP)
// on FINAL money (after the liquidation), the same quantity check_winner
// compares. It separates a close game from a blowout, so a lost game still
// says how badly and a won one how safely.
constexpr double   MARGIN_SCALE     = 5000.0;
constexpr float    MARGIN_COEF      = 0.3f;
constexpr float    MARGIN_CLIP      = 0.5f;

constexpr size_t OBS_ELEMS = (size_t)NUM_INPUT_CH * CELLS;

// Weight of the value loss. The critic has its own body now, so this only
// scales the critic's own gradient.
constexpr float VALUE_COEF = 0.5f;

// The per-set advantage is clamped to +-ADV_CLIP before the policy loss. The
// ppo log prints what share of sets the clamp changed.
constexpr double ADV_CLIP = 6.0;

// ---------------------------------------------------------------------------
// GRADIENT-NORM CLIPPING, ONE THRESHOLD PER MODEL
// ---------------------------------------------------------------------------
// The net is three models with three losses and no shared weights (checked by
// the grouping at start-up and by a backward test on the real net):
//
//   CELL    trunk_cell.*, spatial.*                      <- pl_cell - ent_cell
//   GLOBAL  global_critic.* policy body, glob_land,       <- pl_glob - ent_glob
//           sell_heads, count_heads, type_heads
//   VALUE   every parameter whose name contains "value"  <- value_loss
//
// Each group is clipped to its OWN global L2 norm, so a spike in one (the
// critic's MSE on large returns, most likely) never scales down the others.
//
// HOW TO SET THEM. Start at 1e9 (never clips) and read grad_norms.csv: it
// holds every iteration's mean and max PRE-CLIP norm per group. Set each
// threshold to about 1.5-2x that group's typical (median) norm. Clipping
// should fire on roughly 5-10% of steps; if clip_frac sits near 1, the
// threshold is acting as a hidden learning-rate cut -- raise it.
constexpr double CLIP_CELL = 1.0;
constexpr double CLIP_GLOB = 1.0;
constexpr double CLIP_VAL  = 1.0;

float gammaa  = 1.0f;
float lambdaa = 0.97f;

// One state = one day's decision for one player.
struct Experience {
    torch::Tensor obs;
    float value;
    float advantage;
    float return_val;
    std::vector<std::vector<drone_action>> joint_actions;
    std::vector<double> log_probs;
    std::vector<int> head_ids;      // parallel to log_probs: which head drew it
};

// ===========================================================================
// entropy_file
// ===========================================================================
// Named fields, value on the line below the name (or on the same line):
//
//     harvest
//     0.01
//     primary
//     0.02
//     lr
//     0.00003
//
// Parsed BY NAME, not by position. Order does not matter, a field may be left
// out (it keeps whatever it held last iteration), '#' starts a comment, and an
// unrecognised name is reported rather than absorbed.
//
// `collect` is the per-cell collect head. There is no care line: on a FORCED
// feed the care is the feed head's own draw, so `feed` covers it.
//
// THE FOUR TYPE SLOTS each have their own line now, plant_type_1 .. _4 (slot 1
// is drawn most, slot 4 least). A bare `plant_type` line -- what older files
// have -- still works: it sets all four at once, and a later per-slot line
// overrides it. `sell_time` is the per-cell hour of an HB_SELL harvest, or its
// NO_HARVEST / KEEP fallback. `hire` is dead: hands are hired on demand, so a
// `hire` line is accepted and has no effect.
static const char* HEAD_NAMES[NUM_HEADS] = {
    "harvest", "water", "primary", "fertilize", "feed",
    "land", "hire", "sell", "plant_n", "place_n",
    "plant_type_1", "plant_type_2", "plant_type_3", "plant_type_4",
    "sell_time", "collect"
};
static_assert(H_COLLECT == 15 && NUM_HEADS == 16, "HEAD_NAMES is out of step with HeadId");

// ===========================================================================
// LOG FORMATTING
// ===========================================================================
// Every console line is "label  value" so nothing has to be counted from a
// header row. Numbers keep three significant figures and switch to 1.2e-05
// style only when they are tiny, so a small entropy never prints as 0.0000.
namespace lfmt {
inline std::string num(double x, int w = 9) {
    std::ostringstream o;
    const double a = std::fabs(x);
    if (x != 0.0 && a < 1e-3) o << std::scientific << std::setprecision(2) << x;
    else if (a >= 1000.0)     o << std::fixed << std::setprecision(0) << x;
    else                      o << std::fixed << std::setprecision(a >= 100.0 ? 1 : a >= 10.0 ? 2 : 3) << x;
    std::string s = o.str();
    if ((int)s.size() < w) s.insert(0, w - s.size(), ' ');
    return s;
}
// "+1.23k" / "-0.40k" for a value already in thousands.
inline std::string kilo(double k) {
    std::ostringstream o;
    o << std::showpos << std::fixed << std::setprecision(2) << k << "k";
    return o.str();
}
// " 412/800  51.5%"
inline std::string ratio(int a, int b) {
    std::ostringstream o;
    o << std::setw(4) << a << "/" << std::left << std::setw(4) << b << std::right
      << std::fixed << std::setprecision(1) << std::setw(6)
      << (b > 0 ? 100.0 * a / b : 0.0) << "%";
    return o.str();
}
inline std::string label(const char* l, int w = 12) {
    std::string s = l;
    if ((int)s.size() < w) s.append(w - s.size(), ' ');
    return s;
}
}  // namespace lfmt

// Heads that no longer draw anything. They stay in HeadId so entropy_file
// and the CSV columns keep their names; the console skips them.
inline bool head_dead(int h) { return h == H_WATER || h == H_PRIMARY || h == H_HIRE; }

static void read_entropy_file(const char* path,
                              float head_entropy[NUM_HEADS], double& lr)
{
    std::ifstream f(path);
    if (!f) return;                      // no file -> keep the current values

    std::string line;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        for (char& c : line) if (c == '=' || c == ',' || c == ':') c = ' ';

        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;

        double v;
        if (!(ls >> v)) {
            std::cerr << "entropy_file: '" << key
                      << "' has no readable value, ignored\n";
            continue;
        }
        if (key == "lr") { lr = v; continue; }
        if (key == "plant_type") {               // legacy: all four slots
            for (int s = 0; s < MACRO_TYPE_SLOTS; ++s)
                head_entropy[head_plant_type(s)] = (float)v;
            continue;
        }

        int h = -1;
        for (int i = 0; i < NUM_HEADS; ++i)
            if (key == HEAD_NAMES[i]) { h = i; break; }
        if (h < 0) {
            std::cerr << "entropy_file: unknown field '" << key << "', ignored\n";
            continue;
        }
        head_entropy[h] = (float)v;
    }
}

// ---------------------------------------------------------------------------
// LOAD A CHECKPOINT ACROSS A LAYER-SHAPE CHANGE
// ---------------------------------------------------------------------------
// torch::load refuses a checkpoint outright if any one tensor changed shape.
// This walks the module tree against the archive instead: every tensor whose
// name and shape still match is copied, a tensor that changed shape keeps its
// fresh initialisation, and the counts are printed. The projection rework only
// widened trunk_value.mlp_fc1 (190 -> 380 inputs), so an older checkpoint
// loads with that one layer re-initialised and everything else intact.
// ---------------------------------------------------------------------------
// KNOWN LAYOUT CHANGES, remapped instead of re-initialised
// ---------------------------------------------------------------------------
// (a) `spatial` 40 -> 42 output channels: the sell-time block grew from 12 to
//     14 (ST_NO_HARVEST, ST_KEEP appended after the hours), which pushed the
//     fertilize and feed pairs down by two. Old channels land where their
//     meaning went; the two new ones start at ZERO (weight and bias), so every
//     old logit is unchanged and the fallbacks start level with a 0 logit.
// (b) global_critic.in_fc / value_in_fc 272 -> 284 inputs: the free-land
//     features were APPENDED to the policy scalars, so the old columns are
//     copied in place and the new ones start at ZERO -- the bodies compute
//     exactly what they did until the new inputs earn a weight.
// (c) `spatial` 42 -> 44 output channels: the collect pair was APPENDED after
//     the feed pair, so every old channel stays where it is. The new pair
//     starts at zero weight with bias [0, COLLECT_PRIOR_LOGIT], i.e. collecting
//     about 88% of the time -- close to the forced collect it replaces, so a
//     loaded policy does not suddenly stop collecting. A 40-channel checkpoint
//     goes through (a) and then (c).
// Returns true if `dst` was filled from `src`.
static bool remap_known(const std::string& name, const torch::Tensor& src, torch::Tensor dst) {
    constexpr int OLD_SPATIAL_A = 40;           // before the sell-time fallbacks
    constexpr int OLD_SPATIAL_C = 42;           // before the collect pair
    constexpr int OLD_SELL_TIME = 12;
    constexpr int OLD_POLICY_SCALARS = 272;
    constexpr float COLLECT_PRIOR_LOGIT = 2.0f;
    const int collect_ch = MACRO_SPATIAL_CH - MACRO_COLLECT_CH;        // 42
    if ((name == "spatial.weight" || name == "spatial.bias") &&
        (src.size(0) == OLD_SPATIAL_A || src.size(0) == OLD_SPATIAL_C) &&
        dst.size(0) == MACRO_SPATIAL_CH &&
        src.dim() == dst.dim() && (src.dim() == 1 || src.sizes().slice(1).equals(dst.sizes().slice(1)))) {
        dst.zero_();
        if (src.size(0) == OLD_SPATIAL_A) {
            const int old_sell_end = MACRO_HARVEST_CH + OLD_SELL_TIME;      // 36
            const int new_sell_end = MACRO_HARVEST_CH + MACRO_SELL_TIME_CH; // 38
            dst.slice(0, 0, old_sell_end).copy_(src.slice(0, 0, old_sell_end));
            dst.slice(0, new_sell_end, collect_ch)
               .copy_(src.slice(0, old_sell_end, OLD_SPATIAL_A));
        } else {
            dst.slice(0, 0, collect_ch).copy_(src);
        }
        if (dst.dim() == 1) dst[collect_ch + 1].fill_(COLLECT_PRIOR_LOGIT);
        return true;
    }
    if ((name == "global_critic.in_fc.weight" || name == "global_critic.value_in_fc.weight") &&
        src.dim() == 2 && dst.dim() == 2 && src.size(0) == dst.size(0) &&
        src.size(1) == OLD_POLICY_SCALARS && dst.size(1) == SCALAR_DIM_POLICY) {
        dst.zero_();
        dst.slice(1, 0, OLD_POLICY_SCALARS).copy_(src);
        return true;
    }
    return false;
}

static void load_compatible_rec(torch::nn::Module& m, torch::serialize::InputArchive& ar,
                                const std::string& prefix, int n[3]) {
    for (auto& p : m.named_parameters(/*recurse=*/false)) {
        torch::Tensor t;
        if (!ar.try_read(p.key(), t)) { ++n[2]; continue; }
        if (t.sizes() != p.value().sizes() && remap_known(prefix + p.key(), t, p.value())) {
            std::cout << "  load: " << prefix << p.key() << " " << t.sizes()
                      << " -> " << p.value().sizes() << ", remapped" << std::endl;
            ++n[0];
            continue;
        }
        if (t.sizes() != p.value().sizes()) {
            std::cout << "  load: " << prefix << p.key() << " " << t.sizes()
                      << " -> " << p.value().sizes() << ", re-initialised" << std::endl;
            ++n[1];
            continue;
        }
        p.value().copy_(t);
        ++n[0];
    }
    for (auto& b : m.named_buffers(/*recurse=*/false)) {
        torch::Tensor t;
        if (ar.try_read(b.key(), t, /*is_buffer=*/true) && t.sizes() == b.value().sizes())
            b.value().copy_(t);
    }
    for (auto& c : m.named_children()) {
        torch::serialize::InputArchive sub;
        if (!ar.try_read(c.key(), sub)) {
            n[2] += (int)c.value()->parameters().size();
            continue;
        }
        load_compatible_rec(*c.value(), sub, prefix + c.key() + ".", n);
    }
}

static void load_compatible(NetPolicy& net, const std::string& path) {
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive ar;
    ar.load_from(path, torch::kCPU);
    int n[3] = { 0, 0, 0 };            // copied, shape changed, missing
    load_compatible_rec(*net, ar, "", n);
    std::cout << "load " << path << ": " << n[0] << " tensors copied, "
              << n[1] << " re-initialised, " << n[2] << " missing" << std::endl;
}

static size_t countParameters(NetPolicy& module) {
    size_t count = 0;
    for (const auto& p : module->parameters(true))
        if (p.requires_grad()) count += p.numel();
    return count;
}

// Structurally identical modules enumerate their tensors in the same order, so
// a positional copy is exact. Buffers are copied too: there are none in this
// net today, but a norm layer with running statistics would otherwise be
// silently left at its initialisation and the clone would not be the model it
// claims to be. Works across devices -- the pool lives on the CPU and the three
// slots it is dealt into live on the GPU.
static void copy_model(NetPolicy& from, NetPolicy& to) {
    torch::NoGradGuard no_grad;
    auto a = from->parameters();
    auto b = to->parameters();
    for (size_t i = 0; i < a.size(); ++i) b[i].copy_(a[i]);
    auto ab = from->buffers();
    auto bb = to->buffers();
    for (size_t i = 0; i < ab.size(); ++i) bb[i].copy_(ab[i]);
}

// ===========================================================================
// FICTITIOUS SELF-PLAY  --  the opponent pool
// ===========================================================================
// Pure self-play trains the policy against exactly one opponent: itself, right
// now. That is a moving target with no memory, and the classic failure is a
// cycle -- the policy learns a counter to what it is currently doing, forgets
// why the previous thing worked, and drifts back to it a few hundred iterations
// later having learned nothing that holds up against anything else.
//
// The fix is to keep old versions of itself around and make some fraction of
// every rollout play against them. THE RULES HERE:
//
//   * A SNAPSHOT EVERY SNAP_EVERY (500) ITERATIONS, into a ring of
//     POOL_MAX (30). The oldest snapshot is evicted once the ring is full.
//   * EACH DRAWN SNAPSHOT PLAYS PER_MODEL_SHARE (1%) OF THE GAMES. Up to
//     N_SAMPLE (20) snapshots are drawn per rollout, uniformly and without
//     replacement, so the pool's share ramps by itself:
//         iter   500  -> 1 model  ->  1%
//         iter  1000  -> 2 models ->  2%
//         ...
//         iter 10000  -> 20 models -> 20%   (the cap, N_SAMPLE * 1%)
//     From then on the ring keeps filling to 30 (iter 15000) and slides,
//     and every rollout draws 20 of the 30 it holds.
//   * THE POOL SIDE PLAYS GREEDILY and ITS EXPERIENCE IS NOT COLLECTED. Those
//     trajectories are off-policy -- drawn from a different set of weights
//     entirely, so the importance ratio in the PPO objective is meaningless for
//     them. Only the live policy's own side of a pool game enters the buffer.
//     Its opponent being frozen changes nothing about its own log-probs.
//
// The pool is CPU-resident; the ones that are drawn are copied into N_SAMPLE
// reusable GPU slots once per iteration, so GPU memory is bounded by N_SAMPLE
// extra models whatever POOL_MAX is.
// ===========================================================================
struct OpponentPool {
    static constexpr size_t POOL_MAX        = 30;     // ring size
    static constexpr int    SNAP_EVERY      = 400;    // iterations per snapshot
    static constexpr int    N_SAMPLE        = 20;     // max models drawn per rollout
    static constexpr float  PER_MODEL_SHARE = 0.01f;  // of the games, per drawn model

    std::deque<NetPolicy> models;    // CPU-resident snapshots, oldest first
    std::deque<int>       born;      // the iteration each was taken at

    bool   empty()    const { return models.empty(); }
    size_t size()     const { return models.size(); }
    int    interval() const { return SNAP_EVERY; }
    // Games each drawn model gets out of G (at least one).
    static int games_per_model(int G) {
        return std::max(1, (int)std::lround(G * PER_MODEL_SHARE));
    }
    // Iteration 0 is not a snapshot: it is the random initialisation, and it is
    // not an opponent anybody learns anything from.
    bool due(int iteration) const {
        return iteration > 0 && iteration % interval() == 0;
    }

    void push(NetPolicy& live, int iteration) {
        NetPolicy snap = std::make_shared<NetPolicyImpl>();   // stays on the CPU
        copy_model(live, snap);
        snap->eval();
        models.push_back(snap);
        born.push_back(iteration);
        while (models.size() > POOL_MAX) { models.pop_front(); born.pop_front(); }
    }

    // Up to `n` distinct indices, uniform over the pool.
    std::vector<int> sample(int n, std::mt19937& rng) const {
        std::vector<int> all((int)models.size());
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        if ((int)all.size() > n) all.resize(n);
        return all;
    }
};

// ===========================================================================
// ROLLOUT
// ===========================================================================
// One encode and one forward per DECISION DAY covers both players of every
// game: cell_to_input fills player 0's observation in row i and player 1's in
// row G+i, so the batch is 2G wide. Day NUM_DAYS-1 skips all of that and runs
// the liquidation instead.
//
// ---------------------------------------------------------------------------
// THE ROW LAYOUT IS WHAT MAKES THE POOL FREE
// ---------------------------------------------------------------------------
// Games [0, G_self) are pure self-play; games [G_self, G) are played against a
// frozen pool model. Player 0 is ALWAYS the live policy, so:
//
//     rows [0, G)              p0 of every game          -> live
//     rows [G, G + G_self)     p1 of the self-play games -> live
//     rows [G + G_self, 2G)    p1 of the pool games      -> pool
//
// The live half is one contiguous slice, and each pool model's share is a
// contiguous slice inside the rest, so the whole thing is `1 + n_opp` forwards
// with no gather and no scatter.
//
// WHAT ENTERS THE BUFFER. Player 0 of every game, and player 1 of the
// self-play games. Player 1 of a pool game is a different set of weights
// playing greedily: its log-probs are not this policy's, so its ratio in the
// PPO objective would be nonsense. It is environment, not experience.
//
// Returns the mean margin (own money minus opponent money, in thousands),
// over the self-play games only, so the number stays comparable across
// iterations as the pool grows.
// ===========================================================================
static float rollouts_macro(NetPolicy& player, simulation games[GAMES_AMOUNT],
                            std::vector<Experience>& buffer, int G,
                            std::vector<NetPolicy>& opp,
                            const std::vector<tape::Pick>& tapes)
{
    // ---- how the games are split -----------------------------------------
    //   [0, G_self)                 self-play
    //   [G_self, G_tape0)           vs a frozen pool model
    //   [G_tape0, G)                vs a recorded tape, p1 is a GHOST seat
    // The tape rows are never forwarded and never planned: p1's moves come
    // straight off the recording, hour by hour.
    const int n_opp   = (int)opp.size();
    const int G_tape  = std::min(G, (int)tapes.size());
    const int G_rest  = G - G_tape;
    // PER_MODEL_SHARE of G per drawn model: 1 model 1%, ..., 20 models 20%.
    const int G_pool  = (n_opp > 0)
        ? std::min(G_rest, n_opp * OpponentPool::games_per_model(G)) : 0;
    const int G_self  = G_rest - G_pool;
    const int G_tape0 = G_self + G_pool;
    auto is_tape = [&](int i) { return i >= G_tape0; };
    auto tape_of = [&](int i) { return i - G_tape0; };
    // Contiguous game range per pool model: [opp_lo[j], opp_lo[j+1]).
    std::vector<int> opp_lo(n_opp + 1, G_self);
    for (int j = 0; j < n_opp; ++j)
        opp_lo[j + 1] = G_self + (int)((int64_t)G_pool * (j + 1) / n_opp);
    // p1 of a pool game is off-policy and contributes nothing.
    auto on_policy = [&](int p, int i) { return p == 0 || i < G_self; };

    torch::Device gpu_device(torch::kCUDA, 0);
    const int D = MACRO_DECISION_DAYS;          // days that produce a sample

    auto pin = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    torch::Tensor host_in  = torch::empty({G * 2, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE}, pin);
    torch::Tensor host_pol = torch::empty({G * 2, MACRO_DIM}, pin);
    torch::Tensor host_val = torch::empty({G * 2, 1}, pin);

    CustomTensor raw_input(1, G * 2, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE, 0);
    raw_input.data      = host_in.data_ptr<float>();
    raw_input.owns_data = 0;

    torch::Tensor obs_store = torch::empty(
        {D, 2 * G, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE},
        torch::TensorOptions().dtype(torch::kFloat32));
    float* obs_ptr = obs_store.data_ptr<float>();

    std::vector<std::vector<std::vector<drone_action>>> act_log[2];
    std::vector<std::vector<double>> log_log[2];
    std::vector<std::vector<int>>    head_log[2];
    std::vector<float>               val_log[2];
    for (int p = 0; p < 2; ++p) {
        act_log[p].assign((size_t)D * G, {});
        log_log[p].assign((size_t)D * G, {});
        head_log[p].assign((size_t)D * G, {});
        val_log[p].assign((size_t)D * G, 0.0f);
    }

    std::vector<MacroDay> plan0(G), plan1(G);
    std::vector<int> result(G, -1);

    player->eval();
    for (int i = 0; i < G; ++i) games[i] = simulation("", 0, rand());
    for (int i = G_tape0; i < G; ++i) tape::seat(games[i], tapes[tape_of(i)]);

    // Net worth each MORNING of a decision day, [t*G + i], plus the end of the
    // game at [D*G + i]. The reward is built from these.
    std::vector<double> nw[2], mo[2];     // net worth / money, same layout
    for (int p = 0; p < 2; ++p) {
        nw[p].assign((size_t)(D + 1) * G, 0.0);
        mo[p].assign((size_t)(D + 1) * G, 0.0);
    }

    for (int dy = 0; dy < NUM_DAYS; ++dy)
    {
        if (dy < D)
        {
            torch::NoGradGuard noGrad;
            raw_input.set_zero();

            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                cell_to_input(raw_input.to_3d(i), raw_input.to_3d(G + i), games[i]);
                for (int p = 0; p < 2; ++p) {
                    nw[p][(size_t)dy * G + i] = games[i].net_worth(p);
                    mo[p][(size_t)dy * G + i] = games[i].players[p].money;
                }
            }

            // Pinned upload; the synchronize below guarantees it has drained
            // before the next day's set_zero overwrites host_in.
            torch::Tensor inputs_gpu = host_in.to(gpu_device, /*non_blocking=*/true);

            // The live policy: every p0 row plus the self-play p1 rows, which
            // the layout above makes one contiguous slice.
            const int live_rows = G + G_self;
            value_policy out = player->forward(inputs_gpu.slice(0, 0, live_rows));
            host_pol.narrow(0, 0, live_rows).copy_(out.policy, /*non_blocking=*/true);
            host_val.narrow(0, 0, live_rows).copy_(out.value,  /*non_blocking=*/true);

            // Each pool model, over its own contiguous block of p1 rows. Only
            // the policy is kept: nothing regresses a value on these states.
            for (int j = 0; j < n_opp; ++j) {
                const int lo = G + opp_lo[j], hi = G + opp_lo[j + 1];
                if (hi <= lo) continue;
                auto po = opp[j]->forward(inputs_gpu.slice(0, lo, hi)).policy;
                host_pol.narrow(0, lo, hi - lo).copy_(po, /*non_blocking=*/true);
            }
            torch::cuda::synchronize();

            const float* pol_ptr = host_pol.data_ptr<float>();
            const float* val_ptr = host_val.data_ptr<float>();

            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i)
            {
                const size_t slot = (size_t)dy * G + i;
                for (int p = 0; p < 2; ++p) {
                    if (p == 1 && is_tape(i)) continue;     // the tape plays itself
                    MacroDay& d = (p == 0) ? plan0[i] : plan1[i];
                    // A frozen pool opponent draws nothing and is recorded
                    // nowhere: it plays the arg-max of its own logits and the
                    // three parallel lists stay untouched for that slot.
                    const bool learn = on_policy(p, i);
                    macro_plan_day(pol_ptr + (size_t)(p * G + i) * MACRO_DIM,
                                   games[i], p, d,
                                   learn ? &act_log[p][slot]  : nullptr,
                                   learn ? &log_log[p][slot]  : nullptr,
                                   learn ? &head_log[p][slot] : nullptr,
                                   /*temperature=*/1.0f, /*greedy=*/!learn, 0);
                    if (!learn) continue;
                    val_log[p][slot] = val_ptr[p * G + i];

                    std::memcpy(obs_ptr + ((size_t)dy * 2 * G + p * G + i) * OBS_ELEMS,
                                host_in.data_ptr<float>() + (size_t)(p * G + i) * OBS_ELEMS,
                                OBS_ELEMS * sizeof(float));
                }
            }
        }
        else
        {
            // ---- the liquidation: no encode, no forward, no experience ----
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                macro_liquidate_day(games[i], 0, plan0[i]);
                if (!is_tape(i)) macro_liquidate_day(games[i], 1, plan1[i]);
            }
        }

        // ---- 24 hours of execution, no network -----------------------
        // A tape seat's recorded weeds and shops are synced AFTER each turn,
        // so the next morning's observation already shows them.
        for (int h = 0; h < TURNS_PER_DAY; ++h) {
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                std::vector<move> mv[NUM_PLAYERS];
                plan0[i].emit(h, mv[0]);
                if (is_tape(i)) {
                    tape::moves(games[i], tapes[tape_of(i)], mv[1]);
                } else {
                    plan1[i].emit(h, mv[1]);
                }
                const int r = games[i].apply_turn(mv);
                if (r != -1) result[i] = r;
                if (is_tape(i)) tape::after(games[i], tapes[tape_of(i)]);
            }
        }

    }

    // ---- GAE over the daily decisions ------------------------------------
    // The terminal lands on t == D-1, the LAST DECISION day. The liquidation
    // that follows it is part of the environment, not part of the trajectory.
    double margin_sum = 0.0;
    int    wins = 0, draws = 0;

    for (int i = 0; i < G; ++i)
        for (int p = 0; p < 2; ++p) {
            nw[p][(size_t)D * G + i] = games[i].net_worth(p);
            mo[p][(size_t)D * G + i] = games[i].players[p].money;
        }

    std::vector<float> adv[2], ret[2];
    for (int p = 0; p < 2; ++p) {
        adv[p].assign((size_t)D * G, 0.0f);
        ret[p].assign((size_t)D * G, 0.0f);
    }

    // The per-day reward, see THE REWARD at the top of the file.
    auto reward = [&](int p, int i, int t) -> float {
        float r = 0.0f;
        if (t >= REWARD_START_DAY) {
            const size_t a = (size_t)t * G + i, b = (size_t)(t + 1) * G + i;
            // LEAD: who holds more money tomorrow morning.
            if (mo[p][b] > mo[1 - p][b])      r += LEAD_REWARD;
            else if (mo[p][b] < mo[1 - p][b]) r -= LEAD_REWARD;
            // GAP: how far today moved the net-worth gap, clamped.
            const double gap_now  = nw[p][a] - nw[1 - p][a];
            const double gap_next = nw[p][b] - nw[1 - p][b];
            r += std::clamp((float)((gap_next - gap_now) * GAP_SCALE),
                            -GAP_CLAMP, GAP_CLAMP);
        }
        if (t == D - 1) {
            if (result[i] == p)          r += END_REWARD;
            else if (result[i] == 1 - p) r -= END_REWARD;
            // MARGIN: final money, the row after the last decision day.
            const size_t fin = (size_t)D * G + i;
            const double margin = mo[p][fin] - mo[1 - p][fin];
            r += std::clamp(MARGIN_COEF * (float)std::tanh(margin / MARGIN_SCALE),
                            -MARGIN_CLIP, MARGIN_CLIP);
        }
        return r;
    };

    double growth_sum = 0.0;          // full return from day 0, live p0 only
    #pragma omp parallel for schedule(guided, 12) reduction(+:growth_sum)
    for (int i = 0; i < G; ++i) {
        for (int p = 0; p < 2; ++p) {
            if (!on_policy(p, i)) continue;     // the pool / tape side has no trajectory
            float gae = 0.0f, next_value = 0.0f, rtg = 0.0f;
            for (int t = D - 1; t >= 0; --t) {
                const size_t s = (size_t)t * G + i;
                const float r = reward(p, i, t);
                rtg += r;
                const float delta = r + gammaa * next_value - val_log[p][s];
                gae = delta + gammaa * lambdaa * gae;
                adv[p][s] = gae;
                ret[p][s] = gae + val_log[p][s];
                next_value = val_log[p][s];
            }
            if (p == 0) growth_sum += rtg;

        }
            {
        static int calls = 0;
        // if (calls++ % 50 == 0) {
        //     std::vector<int> show;
        //     if (G_self > 0) show.push_back(0);
        //     if (G_tape > 0) show.push_back(G_tape0);
        //     for (int i : show) {
        //         std::printf("  -- rewards, game %d (%s), result %d, money %.0f vs %.0f\n",
        //                     i, is_tape(i) ? "tape" : "self", result[i],
        //                     games[i].players[0].money, games[i].players[1].money);
        //         std::printf("   day      my_nw     opp_nw        gap   reward    value      adv   return\n");
        //         for (int t = 0; t < D; ++t) {
        //             const size_t s = (size_t)t * G + i;
        //             const double gap = nw[0][s] - nw[1][s];
        //             std::printf("  %4d %10.0f %10.0f %10.0f %8.4f %8.4f %8.4f %8.4f\n",
        //                         t, nw[0][s], nw[1][s], gap, reward(0, i, t),
        //                         val_log[0][s], adv[0][s], ret[0][s]);
        //         }
        //         std::printf("   end %10.0f %10.0f %10.0f\n",
        //                     nw[0][(size_t)D * G + i], nw[1][(size_t)D * G + i],
        //                     nw[0][(size_t)D * G + i] - nw[1][(size_t)D * G + i]);
        //     }
        // }
    }
    }

    int pool_wins = 0, pool_draws = 0;
    double pool_margin = 0.0;
    int tape_wins = 0, tape_draws = 0;
    double tape_margin = 0.0, tape_money = 0.0, tape_rec = 0.0, my_money = 0.0;
    long tape_dropped = 0;

    for (int i = 0; i < G; ++i) {
        const double m = (games[i].players[0].money - games[i].players[1].money) / 1000.0;
        if (i < G_self) {
            margin_sum += m;
            if (result[i] == 0) ++wins;
            else if (result[i] == 2) ++draws;
        } else if (!is_tape(i)) {
            pool_margin += m;
            if (result[i] == 0) ++pool_wins;
            else if (result[i] == 2) ++pool_draws;
        } else {
            const tape::Pick& tp = tapes[tape_of(i)];
            tape_margin += m;
            my_money    += games[i].players[0].money / 1000.0;
            tape_money  += games[i].players[1].money / 1000.0;
            tape_rec    += tp.ep->final_money[tp.side] / 1000.0;
            // Our hands never address a missing unit, so every drop here is
            // the tape's: its recorded crew and ours have drifted apart.
            tape_dropped += games[i].dropped_no_unit;
            if (result[i] == 0) ++tape_wins;
            else if (result[i] == 2) ++tape_draws;
        }
        for (int p = 0; p < 2; ++p) {
            if (!on_policy(p, i)) continue;
            for (int t = 0; t < D; ++t) {
                const size_t s = (size_t)t * G + i;
                // A day where every head was masked out has nothing to learn
                // from. Skipped silently; the day still happened.
                if (act_log[p][s].empty()) continue;
                buffer.push_back({
                    obs_store[t][p * G + i],
                    val_log[p][s],
                    adv[p][s],
                    ret[p][s],
                    act_log[p][s],
                    log_log[p][s],
                    head_log[p][s]
                });
            }
        }
    }

    // ---- critic explained variance, on-policy states only ----------------
    double sz = 0, sz2 = 0, se = 0, se2 = 0; long n = 0;
    for (int i = 0; i < G; ++i) for (int p = 0; p < 2; ++p) {
        if (!on_policy(p, i)) continue;
        for (int t = 0; t < D; ++t) {
            const float z = ret[p][(size_t)t * G + i];
            const float e = z - val_log[p][(size_t)t * G + i];
            sz += z; sz2 += z * z; se += e; se2 += e * e; ++n;
        }
    }
    if (n > 0) {
        const double var_z = sz2 / n - (sz / n) * (sz / n);
        const double var_e = se2 / n - (se / n) * (se / n);
        std::cout << "  " << lfmt::label("critic")
                  << "explained variance " << lfmt::num(1.0 - var_e / (var_z + 1e-8), 6)
                  << "    mean return (p0, from day 0) "
                  << lfmt::num(growth_sum / std::max(1, G), 7) << std::endl;
    }

    // Self-play and pool games are reported apart. Mixing them would make the
    // headline number move whenever the pool's strength moved, which is not
    // what it is there to measure: the self-play row is the frontier, and the
    // pool row is how the current policy fares against its own past.
    const int Gs = std::max(1, G_self);
    if (G_self > 0)
        std::cout << "  " << lfmt::label("self-play")
                  << "wins " << lfmt::ratio(wins, G_self)
                  << "   draws " << std::setw(3) << draws
                  << "   margin " << lfmt::kilo(margin_sum / Gs) << std::endl;
    if (G_pool > 0) {
        const std::string lab = "vs pool(" + std::to_string(n_opp) + ")";
        std::cout << "  " << lfmt::label(lab.c_str())
                  << "wins " << lfmt::ratio(pool_wins, G_pool)
                  << "   draws " << std::setw(3) << pool_draws
                  << "   margin " << lfmt::kilo(pool_margin / G_pool) << std::endl;
    }
    // THE TAPE ROW. `tape $` is what the recorded plan earned in OUR market,
    // `rec $` what it earned in its own game; the gap is how much our selling
    // moved its prices plus whatever replay drift there is. dropped_no_unit
    // should stay near zero -- if it climbs, the tapes' crews are drifting.
    if (G_tape > 0) {
        std::cout << "  " << lfmt::label("vs tapes")
                  << "wins " << lfmt::ratio(tape_wins, G_tape)
                  << "   draws " << std::setw(3) << tape_draws
                  << "   margin " << lfmt::kilo(tape_margin / G_tape) << std::endl;
        std::cout << "  " << lfmt::label("")
                  << "final money: mine " << std::fixed << std::setprecision(1)
                  << (my_money / G_tape) << "k   tape here "
                  << (tape_money / G_tape) << "k   tape recorded "
                  << (tape_rec / G_tape) << "k   replay drops/game "
                  << std::setprecision(2) << ((double)tape_dropped / G_tape)
                  << std::defaultfloat << std::endl;
    }

    player->train();
    return G_self > 0 ? (float)(margin_sum / Gs)
                      : (float)(tape_margin / std::max(1, G_tape));
}

// ===========================================================================
// EVALUATION -- greedy, against a frozen opponent
// ===========================================================================
static float eval_vs(NetPolicy& live, NetPolicy& frozen, int G, int index,
                     std::ofstream& log)
{
    torch::NoGradGuard no_grad;
    torch::Device gpu_device(torch::kCUDA, 0);
    live->eval();
    frozen->eval();

    std::vector<simulation> games(G);
    for (int i = 0; i < G; ++i)
        games[i] = simulation(i == 0 ? "game_log_" + std::to_string(index) : "",
                              i == 0, rand());

    CustomTensor raw_input(1, G * 2, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE, 1);
    std::vector<MacroDay> plan0(G), plan1(G);
    std::vector<int> result(G, -1);

    // The stats printed below are the LAST DECISION day's, snapshotted before
    // the liquidation overwrites the plans -- otherwise every number would be
    // the fixed last day's and would say nothing about the policy.
    std::vector<MacroStats> last_stats(G);

    for (int dy = 0; dy < NUM_DAYS; ++dy) {
        if (dy < MACRO_DECISION_DAYS) {
            raw_input.set_zero();
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i)
                cell_to_input(raw_input.to_3d(i), raw_input.to_3d(G + i), games[i]);

            torch::Tensor in = torch::from_blob(
                raw_input.data, {G * 2, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE},
                torch::kFloat32).to(gpu_device);

            auto p0 = live->forward(in.slice(0, 0, G)).policy.to(torch::kCPU).contiguous();
            auto p1 = frozen->forward(in.slice(0, G, G * 2)).policy.to(torch::kCPU).contiguous();
            const float* a = p0.data_ptr<float>();
            const float* b = p1.data_ptr<float>();

            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                macro_plan_day(a + (size_t)i * MACRO_DIM, games[i], 0, plan0[i],
                               nullptr, nullptr, nullptr, 1.0f, /*greedy=*/true, 0);
                macro_plan_day(b + (size_t)i * MACRO_DIM, games[i], 1, plan1[i],
                               nullptr, nullptr, nullptr, 1.0f, /*greedy=*/true, 0);
            }
            if (dy == MACRO_DECISION_DAYS - 1)
                for (int i = 0; i < G; ++i) last_stats[i] = plan0[i].stats;
        } else {
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                macro_liquidate_day(games[i], 0, plan0[i]);
                macro_liquidate_day(games[i], 1, plan1[i]);
            }
        }
        for (int h = 0; h < TURNS_PER_DAY; ++h) {
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                std::vector<move> mv[NUM_PLAYERS];
                plan0[i].emit(h, mv[0]);
                plan1[i].emit(h, mv[1]);
                const int r = games[i].apply_turn(mv);
                if (r != -1) result[i] = r;
            }
        }
    }

    int w = 0, l = 0, dr = 0;
    double margin = 0.0, ops = 0.0, cells = 0.0;
    double musts = 0.0, mdrop = 0.0, crew = 0.0;
    double h_must = 0.0, h_harv = 0.0, h_plant = 0.0, h_fert = 0.0, h_feed = 0.0;
    double s_sell = 0.0, s_keep = 0.0, s_skip = 0.0, s_redraw = 0.0, s_fall = 0.0;
    double fert = 0.0, care = 0.0, coll = 0.0, esc = 0.0, h_coll = 0.0;
    double liq_crew = 0.0, liq_cells = 0.0, liq_left = 0.0;
    long dropped = 0;
    for (int i = 0; i < G; ++i) {
        if (result[i] == 0) ++w; else if (result[i] == 1) ++l; else ++dr;
        margin += (games[i].players[0].money - games[i].players[1].money) / 1000.0;
        cells  += last_stats[i].accepted_cells;
        ops    += last_stats[i].accepted_ops;
        musts  += last_stats[i].must_ops;
        mdrop  += last_stats[i].must_dropped;
        crew   += last_stats[i].hired + 1;
        h_must  += last_stats[i].must_hires;
        h_harv  += last_stats[i].harvest_hires;
        h_plant += last_stats[i].plant_hires;
        h_fert  += last_stats[i].fert_hires;
        h_feed  += last_stats[i].feed_hires;
        s_sell   += last_stats[i].harvest_sell;
        s_keep   += last_stats[i].sell_keep;
        s_skip   += last_stats[i].sell_skip;
        s_redraw += last_stats[i].sell_redrawn;
        s_fall   += last_stats[i].sell_fallback;
        fert   += last_stats[i].fertilized;
        care   += last_stats[i].cared;
        coll   += last_stats[i].collected;
        esc    += last_stats[i].escape_harvests;
        h_coll += last_stats[i].collect_hires;
        // The liquidation: how big a crew it needed and what it could not
        // reach in time. liq_left should be at or near zero -- anything else
        // is money left standing on the board on the last night.
        liq_crew  += plan0[i].stats.hired + 1;
        liq_cells += plan0[i].stats.accepted_cells;
        liq_left  += plan0[i].stats.must_dropped;
        dropped += games[i].dropped_no_unit;
    }
    // dropped_no_unit should be exactly zero; anything else means the
    // projection and the referee disagree about the crew.
    //
    // must_dropped should be at or near zero. It now also counts cares shed
    // to make a feed fit, which is the first thing to go when hands run out.
    //
    // cared counts both the care that rides on a drawn feed and the feed head's
    // draws on forced feeds; collected is the collect head's.
    // Plan figures are per game, from the LAST decision day's plan.
    std::cout << "\n  EVAL vs frozen  wins " << lfmt::ratio(w, G)
              << "   losses " << l << "   draws " << dr
              << "   margin " << lfmt::kilo(margin / G) << "\n"
              << std::fixed << std::setprecision(1)
              << "    last-day plan   cells " << (cells / G)
              << "   ops " << (ops / G)
              << "   forced ops " << (musts / G)
              << " (dropped " << std::setprecision(2) << (mdrop / G) << std::setprecision(1) << ")"
              << "   crew " << (crew / G)
              << "   fertilized " << (fert / G)
              << "   cared " << (care / G)
              << "   collected " << (coll / G)
              << "   escape harvests " << (esc / G) << "\n"
              << "    hires by pass   must " << (h_must / G)
              << "   harvest+drop " << (h_harv / G)
              << "   plant " << (h_plant / G)
              << "   fert " << (h_fert / G)
              << "   feed+care " << (h_feed / G)
              << "   collect " << (h_coll / G) << "\n"
              << "    sell-time       sold " << (s_sell / G)
              << "   keep " << (s_keep / G)
              << "   no-harvest " << (s_skip / G)
              << "   redrawn " << (s_redraw / G)
              << "   forced keep " << (s_fall / G) << "\n"
              << "    liquidation     crew " << (liq_crew / G)
              << "   cells " << (liq_cells / G)
              << "   missed " << std::setprecision(2) << (liq_left / G)
              << "   desyncs " << dropped << " (should be 0)"
              << std::defaultfloat << std::endl;
    log << w << " " << l << " " << dr << " " << (margin / G) << std::endl;
    if (w >= 170)
        log << "0" << std::endl;
    live->train();

    return (float)w / (float)G;
}

// ===========================================================================
// EVALUATION -- greedy, against the fixed tape bench
// ===========================================================================
// The same tapes, in the same seats, every call. The live player's weeds are
// seeded per bench slot (TAPE_BENCH_SEED + i), so the ONLY thing that changes
// between two calls is the policy: the curve in tape_bench_logs.csv is a
// straight measurement of progress against recorded players.
//
// Columns: iteration, wins, losses, draws, margin(k), my money(k),
// tape money in our market(k), tape money as recorded(k).
static float eval_vs_tapes(NetPolicy& live, const std::vector<tape::Pick>& bench,
                           int index, std::ofstream& log, std::ofstream& winrate_log)
{
    const int G = (int)bench.size();
    if (G == 0) return 0.0f;

    torch::NoGradGuard no_grad;
    torch::Device gpu_device(torch::kCUDA, 0);
    live->eval();

    std::vector<simulation> games(G);
    for (int i = 0; i < G; ++i) {
        games[i] = simulation(i == 0 ? "tape_log_" + std::to_string(index) : "",
                              i == 0, TAPE_BENCH_SEED + (uint64_t)i);
        tape::seat(games[i], bench[i]);
    }

    CustomTensor raw_input(1, G * 2, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE, 1);
    std::vector<MacroDay> plan0(G);
    std::vector<int> result(G, -1);

    for (int dy = 0; dy < NUM_DAYS; ++dy) {
        if (dy < MACRO_DECISION_DAYS) {
            raw_input.set_zero();
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i)
                cell_to_input(raw_input.to_3d(i), raw_input.to_3d(G + i), games[i]);

            // Only the p0 half is ever read: the tape side plays itself.
            torch::Tensor in = torch::from_blob(
                raw_input.data, {G, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE},
                torch::kFloat32).to(gpu_device);
            auto p0 = live->forward(in).policy.to(torch::kCPU).contiguous();
            const float* a = p0.data_ptr<float>();

            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i)
                macro_plan_day(a + (size_t)i * MACRO_DIM, games[i], 0, plan0[i],
                               nullptr, nullptr, nullptr, 1.0f, /*greedy=*/true, 0);
        } else {
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) macro_liquidate_day(games[i], 0, plan0[i]);
        }
        for (int h = 0; h < TURNS_PER_DAY; ++h) {
            #pragma omp parallel for schedule(guided, 12)
            for (int i = 0; i < G; ++i) {
                std::vector<move> mv[NUM_PLAYERS];
                plan0[i].emit(h, mv[0]);
                tape::moves(games[i], bench[i], mv[1]);
                const int r = games[i].apply_turn(mv);
                if (r != -1) result[i] = r;
                tape::after(games[i], bench[i]);
            }
        }
    }

    int w = 0, l = 0, dr = 0;
    double margin = 0.0, mine = 0.0, theirs = 0.0, rec = 0.0;
    long dropped = 0;
    for (int i = 0; i < G; ++i) {
        if (result[i] == 0) ++w; else if (result[i] == 1) ++l; else ++dr;
        margin += (games[i].players[0].money - games[i].players[1].money) / 1000.0;
        mine   += games[i].players[0].money / 1000.0;
        theirs += games[i].players[1].money / 1000.0;
        rec    += bench[i].ep->final_money[bench[i].side] / 1000.0;
        dropped += games[i].dropped_no_unit;
    }
    std::cout << "  BENCH vs tapes  wins " << lfmt::ratio(w, G)
              << "   losses " << l << "   draws " << dr
              << "   margin " << lfmt::kilo(margin / G) << "\n"
              << std::fixed << std::setprecision(1)
              << "    final money     mine " << (mine / G)
              << "k   tape here " << (theirs / G)
              << "k   tape recorded " << (rec / G)
              << "k   replay drops " << dropped
              << std::defaultfloat << "\n" << std::endl;
    log << index << "," << w << "," << l << "," << dr << "," << (margin / G) << ","
        << (mine / G) << "," << (theirs / G) << "," << (rec / G) << std::endl;
    // Also into winrate_logs.csv, right under that eval's self-play line, in
    // the same "wins losses draws margin" order, tagged so the two can be
    // told apart: "tape w l dr margin my$ tape$ rec$".
    winrate_log << "tape " << w << " " << l << " " << dr << " " << (margin / G) << " "
                << (mine / G) << " " << (theirs / G) << " " << (rec / G) << std::endl;
    live->train();
    return (float)w / (float)G;
}

// ===========================================================================
int main()
{
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);

    torch::Device gpu_device(torch::kCUDA, 0);
    srand((unsigned)time(NULL));

    NetPolicy player = std::make_shared<NetPolicyImpl>();
    // load_compatible(player, "./model_macro_6000.pt");
    //
    // NOTE (no hire head, 14-bin sell-time head, free-land features): a
    // checkpoint from before this change loads with glob_hire ignored,
    // `spatial` and the two MLP input layers REMAPPED (remap_known), and
    // nothing else touched. The two new sell-time channels and the twelve new
    // scalar columns start at zero.
    //
    // NOTE (global/critic merge): trunk_value and trunk_global are GONE,
    // replaced by one conv-less MLP, `global_critic`, that reads the new
    // FCAST scalar block. An older checkpoint loads through load_compatible
    // with trunk_cell and `spatial` intact; global_critic, every global head
    // and the halved count heads start fresh.
    //
    // NOTE: the critic projection widened trunk_value.mlp_fc1 (190 -> 380
    // scalar inputs), so torch::load refuses every earlier checkpoint.
    // load_compatible takes one anyway: every tensor that kept its shape is
    // copied and that one layer starts fresh. Expect critic EV to dip for a
    // few hundred iterations while it relearns its input layer; the policy
    // is untouched. (A checkpoint from before the per-type harvest head,
    // MACRO_DIM 2042, also loads, with `spatial` re-initialised.)

    NetPolicy frozen = std::make_shared<NetPolicyImpl>();
    copy_model(player, frozen);

    player->to(gpu_device);
    frozen->to(gpu_device);
    std::cout << "params: " << countParameters(player) << std::endl;
    std::cout << "reward: net-worth gap / " << (int)(1.0 / GAP_SCALE)
              << " per day from day " << REWARD_START_DAY
              << ", lead +-" << LEAD_REWARD << " per day, +-" << END_REWARD
              << " at the end, margin " << MARGIN_COEF << "*tanh(d/"
              << (int)MARGIN_SCALE << ") clipped +-" << MARGIN_CLIP << std::endl;

    std::vector<torch::Tensor> params;
    for (auto& np : player->named_parameters()) {
        params.push_back(np.value());
    }
    auto make_adam_opts = [](double lr) {
        auto o = std::make_unique<torch::optim::AdamOptions>(lr);
        
        return o;
    };
    torch::optim::Adam optimizer(
        { torch::optim::OptimizerParamGroup(params,  make_adam_opts(1e-4))},
        torch::optim::AdamOptions(1e-4));

    // ---- the three clip groups ------------------------------------------
    // By NAME, and strictly: a parameter that matches no rule stops the run,
    // so a module added later cannot silently join the wrong group.
    enum ClipGroup { CG_CELL = 0, CG_GLOB, CG_VAL, NUM_CG };
    static const char* CG_NAME[NUM_CG] = { "cell", "global", "value" };
    const double CG_CLIP[NUM_CG] = { CLIP_CELL, CLIP_GLOB, CLIP_VAL };
    std::vector<torch::Tensor> clip_params[NUM_CG];
    {
        auto starts = [](const std::string& s, const char* pre) {
            return s.rfind(pre, 0) == 0;
        };
        long numel[NUM_CG] = { 0, 0, 0 };
        for (auto& np : player->named_parameters()) {
            const std::string& n = np.key();
            int g;
            if (n.find("value") != std::string::npos)                  g = CG_VAL;
            else if (starts(n, "trunk_cell.") || starts(n, "spatial.")) g = CG_CELL;
            else if (starts(n, "global_critic.") || starts(n, "glob_land.") ||
                     starts(n, "sell_heads.")   || starts(n, "count_heads.") ||
                     starts(n, "type_heads."))                          g = CG_GLOB;
            else {
                std::cerr << "clip groups: parameter '" << n
                          << "' matches no group -- add it to the rules" << std::endl;
                return 1;
            }
            clip_params[g].push_back(np.value());
            numel[g] += np.value().numel();
        }
        size_t total = 0;
        for (int g = 0; g < NUM_CG; ++g) total += clip_params[g].size();
        if (total != params.size()) {
            std::cerr << "clip groups: " << total << " grouped, "
                      << params.size() << " in the optimizer" << std::endl;
            return 1;
        }
        std::cout << "clip groups:";
        for (int g = 0; g < NUM_CG; ++g)
            std::cout << "  " << CG_NAME[g] << " " << clip_params[g].size()
                      << " tensors / " << numel[g] << " params (clip "
                      << CG_CLIP[g] << ")";
        std::cout << std::endl;
    }
    std::ofstream grad_log("grad_norms.csv");
    grad_log << "iteration,steps,skipped";
    for (int g = 0; g < NUM_CG; ++g)
        grad_log << "," << CG_NAME[g] << "_mean," << CG_NAME[g] << "_max,"
                 << CG_NAME[g] << "_clip_frac";
    grad_log << std::endl;

    std::ofstream log_file("ppo_logs.csv");
    std::ofstream winrate_log("winrate_logs.csv");
    // ppo_logs.csv is still plain CSV, but every column is padded to one
    // fixed width and every float has the same number of decimals, so the
    // numbers sit right under their names in a text editor. Readers that
    // mind the padding: pandas.read_csv(..., skipinitialspace=True).
    std::vector<std::string> csv_cols = {
        "iteration", "states", "sets_per_state",
        "policy_loss", "value_loss", "entropy", "entropy_loss" };
    for (int h = 0; h < NUM_HEADS; ++h) csv_cols.push_back(std::string("e_") + HEAD_NAMES[h]);
    // The advantage the policy loss actually sees: per choice set, AFTER the
    // +-ADV_CLIP clamp. clip_frac is the share of sets the clamp changed.
    for (const char* c : { "adv_mean", "adv_std", "adv_min", "adv_max", "adv_clip_frac" })
        csv_cols.push_back(c);
    std::vector<int> csv_w;
    for (const auto& c : csv_cols) csv_w.push_back(std::max<int>(12, (int)c.size()));
    auto csv_row = [&](const std::vector<std::string>& cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) log_file << ",";
            log_file << std::setw(csv_w[i]) << cells[i];
        }
        log_file << std::endl;
    };
    auto csv_f = [](double x) {                // every float: 6 decimals
        std::ostringstream o;
        o << std::fixed << std::setprecision(6) << x;
        return o.str();
    };
    csv_row(csv_cols);

    static simulation games[GAMES_AMOUNT];

    // ---- fictitious self-play ------------------------------------------
    // The pool itself lives on the CPU; these are the GPU slots the iteration's
    // draw is dealt into, reused every iteration, so the GPU cost of the scheme
    // is N_SAMPLE models no matter how big POOL_MAX gets.
    OpponentPool pool;
    tape::Library tape_lib(TAPE_DIR);
    tape_lib.winners_only = TAPE_WINNERS_ONLY;

    // ---- THE BENCH: fixed tapes, drawn once, fixed seed -----------------
    std::vector<tape::Pick> bench;
    {
        std::mt19937 bench_rng(TAPE_BENCH_SEED);
        bench = tape_lib.sample(TAPE_BENCH_N, bench_rng);
        std::set<std::string> uniq;
        for (const tape::Pick& b : bench) uniq.insert(b.ep->path);
        const bool hold = TAPE_BENCH_HOLDOUT &&
                          tape_lib.files.size() >= 3 * uniq.size();
        if (hold) for (const std::string& p : uniq) tape_lib.exclude.insert(p);
        std::cout << "tape bench: " << bench.size() << " games over " << uniq.size()
                  << " episodes, " << (hold ? "held out of training"
                                            : "ALSO used in training") << std::endl;
    }
    std::ofstream bench_log("tape_bench_logs.csv");
    bench_log << "iteration,wins,losses,draws,margin_k,my_k,tape_k,rec_k" << std::endl;

    std::vector<NetPolicy> opp_slots;
    for (int k = 0; k < (TAPE_ONLY ? 0 : OpponentPool::N_SAMPLE); ++k) {
        NetPolicy slot = std::make_shared<NetPolicyImpl>();
        slot->to(gpu_device);
        slot->eval();
        opp_slots.push_back(slot);
    }

    // One TARGET entropy per head, in HeadId order. Each head is normalised to
    // its own target and the results averaged over the heads present in the
    // minibatch.
    //
    // `fertilize` fires only where its payout window can open, so it will
    // print "--" on some minibatches. `feed` fires on every animal that is not
    // already force-fed, and now implicitly carries the care decision too.
    // `land` fires at most once a day. `hire` no longer draws: hands are
    // hired on demand.
    float head_entropy[NUM_HEADS];
    for (int h = 0; h < NUM_HEADS; ++h) head_entropy[h] = 0.008f;

    // Which model each head reads (head_is_global). A flag per head rather
    // than an index threshold: sell_time is a per-cell head that sits after
    // the global ones in HeadId.
    float head_glob_flags[NUM_HEADS];
    for (int h = 0; h < NUM_HEADS; ++h) head_glob_flags[h] = head_is_global(h) ? 1.0f : 0.0f;
    const torch::Tensor head_glob_t =
        torch::from_blob(head_glob_flags, {NUM_HEADS}, torch::kFloat32).clone().to(gpu_device);

    double initial_lr = 3e-5, final_lr = 1e-5;
    const int total_iters = 200000;
    const int GRAD_ACCUM_STEPS = 1;

    // ---- reusable minibatch staging, allocated once and grown on demand ---
    torch::Tensor pin_obs, pin_idx, pin_flt;
    auto ensure_pin = [](torch::Tensor& t, int64_t n, torch::ScalarType dt) {
        if (!t.defined() || t.numel() < n)
            t = torch::empty({n},
                torch::TensorOptions().dtype(dt).pinned_memory(true));
    };
    std::vector<int>     keep;
    std::vector<int64_t> off_set, off_cand;

    for (int iteration = 0; iteration < total_iters; ++iteration)
    {
        const float progress = (float)iteration / total_iters;
        double lr = initial_lr + (final_lr - initial_lr) * progress;
        read_entropy_file("entropy_file", head_entropy, lr);

        auto coeff_t = torch::from_blob(head_entropy, {NUM_HEADS}, torch::kFloat32)
                           .clone().to(gpu_device);
        for (auto& pg : optimizer.param_groups())
            static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        {
            std::ostringstream h;
            h << "---- iter " << iteration << "   lr " << std::scientific
              << std::setprecision(2) << lr << " ";
            std::string line = h.str();
            if (line.size() < 72) line.append(72 - line.size(), '-');
            std::cout << line << std::endl;
        }
        if (iteration % 400 == 0) {
            if (dbg_file.is_open()) dbg_file.close();
            dbg_file.open("dbg_policy_logs.txt", std::ios::out | std::ios::trunc);
        }

        // ---- the pool: snapshot on schedule, then deal this rollout ------
        // Taken BEFORE the rollout, so the snapshot at iteration N is already
        // an eligible opponent at iteration N. One every SNAP_EVERY; each
        // drawn model takes PER_MODEL_SHARE of the games, N_SAMPLE at most.
        if (!TAPE_ONLY && pool.due(iteration)) {
            pool.push(player, iteration);
            std::cout << "  " << lfmt::label("pool")
                      << "snapshot added   size " << pool.size() << "/" << OpponentPool::POOL_MAX
                      << "   playing " << std::min<size_t>(pool.size(), OpponentPool::N_SAMPLE)
                      << " x " << OpponentPool::games_per_model(GAMES_AMOUNT) << " games" << std::endl;
        }
        std::vector<NetPolicy> opp;
        if (!TAPE_ONLY && !pool.empty()) {
            const std::vector<int> pick = pool.sample(OpponentPool::N_SAMPLE, gen);
            for (size_t k = 0; k < pick.size(); ++k) {
                copy_model(pool.models[pick[k]], opp_slots[k]);
                opp_slots[k]->eval();
                opp.push_back(opp_slots[k]);
            }
        }

        // ---- the tapes: re-list the directory and draw this rollout's ----
        const std::vector<tape::Pick> picks =
            tape_lib.sample(TAPE_ONLY
                ? GAMES_AMOUNT - (int)(GAMES_AMOUNT * SELFPLAY_SHARE)
                : TAPES_PER_ITER, gen);
        if (TAPE_ONLY && picks.empty() && iteration % 50 == 0)
            std::cout << "  WARNING: no usable tapes in " << TAPE_DIR
                      << " -- this iteration is plain self-play" << std::endl;
        if (iteration % 50 == 0)
            std::cout << "  " << lfmt::label("tapes")
                      << picks.size() << " drawn from "
                      << tape_lib.files.size() << " files   ("
                      << tape_lib.cache.size() << " cached, "
                      << tape_lib.bad.size() << " unreadable)" << std::endl;

        player->train();
        std::vector<Experience> buffer;
        buffer.reserve((size_t)GAMES_AMOUNT * MACRO_DECISION_DAYS * 2);
        rollouts_macro(player, games, buffer, GAMES_AMOUNT, opp, picks);
        if (buffer.empty()) continue;

        // ---- normalise advantages over the whole rollout -----------------
        std::vector<float> all_advs;
        all_advs.reserve(buffer.size());
        for (const auto& e : buffer) all_advs.push_back(e.advantage);
        auto adv_tensor = torch::tensor(all_advs, torch::kFloat32);
        const float adv_mean = adv_tensor.mean().item<float>();
        const float adv_std  = adv_tensor.std().item<float>() + 1e-5f;
        for (auto& e : buffer) e.advantage = (e.advantage - adv_mean) / adv_std;

        int mini_batch_size = (int)buffer.size() / 10;
        if (mini_batch_size == 0) mini_batch_size = (int)buffer.size();

        optimizer.zero_grad();
        int accum_counter = 0, v = 0;

        // Pre-clip norms over this iteration's optimizer steps.
        double gn_sum[NUM_CG] = { 0, 0, 0 }, gn_max[NUM_CG] = { 0, 0, 0 };
        int gn_clipped[NUM_CG] = { 0, 0, 0 };
        int gn_steps = 0, gn_skipped = 0;

        std::shuffle(buffer.begin(), buffer.end(), gen);
        for (size_t start_idx = 0; start_idx < buffer.size(); start_idx += mini_batch_size)
        {
            ++v;
            size_t end_idx = std::min(buffer.size(), start_idx + mini_batch_size);
            if (end_idx - start_idx < (size_t)mini_batch_size && start_idx != 0) break;

            // ---- PASS A: sizes and per-state offsets ---------------------
            keep.clear(); off_set.clear(); off_cand.clear();
            int64_t tot_sets = 0, tot_cands = 0;
            for (size_t i = start_idx; i < end_idx; ++i) {
                const auto& e = buffer[i];
                if (e.joint_actions.empty()) continue;
                keep.push_back((int)i);
                off_set.push_back(tot_sets);
                off_cand.push_back(tot_cands);
                tot_sets += (int64_t)e.joint_actions.size();
                for (const auto& ja : e.joint_actions) tot_cands += (int64_t)ja.size();
            }
            if (keep.empty()) continue;

            const int64_t K = (int64_t)keep.size();
            const int64_t C = tot_cands, S = tot_sets;

            // Index block layout, all int64:
            //   [0, C)          actions_index
            //   [C, 2C)         output_move_indices
            //   [2C, 3C)        valid_action_indices
            //   [3C, 3C+S)      chosen_move_indices
            //   [3C+S, 3C+2S)   output_joint_move_indices
            //   [3C+2S, 3C+3S)  head id per choice set
            ensure_pin(pin_idx, 3 * C + 3 * S, torch::kLong);
            // Float block: [0,S) log_probs, [S,2S) each SET's advantage (the
            // state's one advantage, already normalised), then returns and
            // old_values, K each.
            ensure_pin(pin_flt, 2 * S + 2 * K, torch::kFloat32);
            ensure_pin(pin_obs, K * (int64_t)OBS_ELEMS, torch::kFloat32);

            int64_t* ix = pin_idx.data_ptr<int64_t>();
            float*   fl = pin_flt.data_ptr<float>();
            float*   ob = pin_obs.data_ptr<float>();

            // ---- PASS B: parallel fill ----------------------------------
            #pragma omp parallel for schedule(guided)
            for (int64_t k = 0; k < K; ++k) {
                const auto& e = buffer[keep[k]];
                std::memcpy(ob + k * (int64_t)OBS_ELEMS,
                            e.obs.data_ptr<float>(),
                            OBS_ELEMS * sizeof(float));
                fl[2 * S + k]     = e.return_val;
                fl[2 * S + K + k] = e.value;

                int64_t c = off_cand[k], g = off_set[k];
                for (size_t j = 0; j < e.joint_actions.size(); ++j, ++g) {
                    const auto& ja = e.joint_actions[j];
                    fl[g] = (float)e.log_probs[j];
                    fl[S + g] = e.advantage;
                    for (size_t q = 0; q < ja.size(); ++q, ++c) {
                        ix[c]         = ja[q].chosen_index;
                        ix[C + c]     = k;
                        ix[2 * C + c] = g;
                    }
                    // The filter swaps the chosen candidate to the back of the
                    // set, so it is the last entry written above.
                    ix[3 * C + g]         = c - 1;
                    ix[3 * C + S + g]     = k;
                    ix[3 * C + 2 * S + g] = (int64_t)e.head_ids[j];
                }
            }

            auto gpu_idx = pin_idx.narrow(0, 0, 3 * C + 3 * S).to(gpu_device);
            auto gpu_flt = pin_flt.narrow(0, 0, 2 * S + 2 * K).to(gpu_device);
            auto mini_observations =
                pin_obs.narrow(0, 0, K * (int64_t)OBS_ELEMS)
                       .view({K, NUM_INPUT_CH, BOARD_SIZE, BOARD_SIZE})
                       .to(gpu_device);

            auto mini_actions_index             = gpu_idx.narrow(0, 0, C);
            auto mini_output_move_indices       = gpu_idx.narrow(0, C, C);
            auto mini_valid_action_indices      = gpu_idx.narrow(0, 2 * C, C);
            auto mini_chosen_move_indices       = gpu_idx.narrow(0, 3 * C, S);
            auto mini_output_joint_move_indices = gpu_idx.narrow(0, 3 * C + S, S);
            auto mini_head_ids                  = gpu_idx.narrow(0, 3 * C + 2 * S, S);

            auto mini_log_prob   = gpu_flt.narrow(0, 0, S);
            auto adv_raw         = gpu_flt.narrow(0, S, S);
            auto adv_per_action  = torch::clamp(adv_raw, -ADV_CLIP, ADV_CLIP);
            auto mini_returns    = gpu_flt.narrow(0, 2 * S, K);
            auto old_values      = gpu_flt.narrow(0, 2 * S + K, K);
            (void)old_values;
            (void)mini_output_joint_move_indices;   // advantages are per set now

            value_policy out = player->forward(mini_observations);
            torch::Tensor entropy;
            torch::Tensor new_joint_log = policy_to_jointlog(
                out.policy, mini_actions_index, mini_output_move_indices,
                mini_valid_action_indices, mini_chosen_move_indices,
                (int)S, entropy, (int)K);
            auto log_ratio = new_joint_log - mini_log_prob;
            auto ratio = log_ratio.exp();
            auto clipped = torch::clamp(ratio, 1.0 - 0.16, 1.0 + 0.16);
            // PER-MODEL MEANS. head_is_global() heads read global_critic; the
            // rest read trunk_cell. Each model's policy loss is the mean over
            // ITS OWN sets, so the many per-cell draws never dilute the global
            // heads' signal against their entropy and value terms.
            auto per_set = torch::max(ratio * -adv_per_action,
                                      clipped * -adv_per_action);
            auto is_glob = head_glob_t.index_select(0, mini_head_ids).to(per_set.dtype());
            auto is_cell = 1.0f - is_glob;
            auto pl_cell = (per_set * is_cell).sum() / is_cell.sum().clamp_min(1.0f);
            auto pl_glob = (per_set * is_glob).sum() / is_glob.sum().clamp_min(1.0f);
            auto policy_loss = pl_cell + pl_glob;          // logged

            auto new_values = out.value.squeeze();
            // Standard MSE value loss (clipping removed to allow the critic to fix large errors)
            auto value_loss = (new_values - mini_returns).pow(2).mean();

            // Per-head entropy targeting.
            auto ones      = torch::ones_like(entropy);
            auto head_sum  = torch::zeros({NUM_HEADS}, entropy.options())
                                 .scatter_add(0, mini_head_ids, entropy);
            auto head_cnt  = torch::zeros({NUM_HEADS}, entropy.options())
                                 .scatter_add(0, mini_head_ids, ones);
            auto present   = (head_cnt > 0).to(entropy.dtype());
            auto head_mean = head_sum / (head_cnt + 1e-8f);
            auto head_scaled = head_mean * (coeff_t / (head_mean.detach() + 1e-8f));
            // Entropy, also per model: the mean over that model's heads.
            auto head_is_glob = head_glob_t.to(head_mean.dtype());
            auto pres_glob = present * head_is_glob;
            auto pres_cell = present - pres_glob;
            auto ent_cell = (head_scaled * pres_cell).sum() / pres_cell.sum().clamp_min(1.0f);
            auto ent_glob = (head_scaled * pres_glob).sum() / pres_glob.sum().clamp_min(1.0f);
            auto entropy_loss = ent_cell + ent_glob;       // logged
            auto entropy_mean = head_mean.sum() / present.sum().clamp_min(1.0f);

            if (iteration % 10 == 0 && v == 1) {
                auto row = torch::cat({policy_loss.detach().reshape({1}),
                                       value_loss.detach().reshape({1}),
                                       entropy_mean.detach().reshape({1}),
                                       entropy_loss.detach().reshape({1}),
                                       head_mean.detach(),
                                       head_cnt.detach(),
                                       adv_per_action.mean().reshape({1}),
                                       adv_per_action.std(/*unbiased=*/false).reshape({1}),
                                       adv_per_action.min().reshape({1}),
                                       adv_per_action.max().reshape({1}),
                                       (adv_raw.abs() > ADV_CLIP).to(torch::kFloat32)
                                           .mean().reshape({1})}, 0).to(torch::kCPU);
                const float* r  = row.data_ptr<float>();
                const float* hm = r + 4;
                const float* hc = r + 4 + NUM_HEADS;
                const float* av = r + 4 + 2 * NUM_HEADS;   // mean std min max clip_frac

                {
                    std::vector<std::string> cells = {
                        std::to_string(iteration), std::to_string(K),
                        csv_f((double)S / K),
                        csv_f(r[0]), csv_f(r[1]), csv_f(r[2]), csv_f(r[3]) };
                    // A head with no sets in this minibatch writes an empty
                    // cell, not 0: pandas reads it as NaN.
                    for (int h = 0; h < NUM_HEADS; ++h)
                        cells.push_back(hc[h] > 0.0f ? csv_f(hm[h]) : std::string(""));
                    for (int j = 0; j < 5; ++j) cells.push_back(csv_f(av[j]));
                    csv_row(cells);
                }

                // One labelled line per quantity, one row per LIVE head.
                // A head that drew nothing in this minibatch says so instead
                // of printing a number: untouched and collapsed must not look
                // the same.
                std::ostringstream o;
                o << "  " << lfmt::label("ppo")
                  << "batch " << K << " states, "
                  << std::fixed << std::setprecision(1) << ((double)S / K)
                  << std::defaultfloat << " choice sets per state\n"
                  << "  " << lfmt::label("")
                  << "policy loss" << lfmt::num(r[0], 10)
                  << "    value loss" << lfmt::num(r[1], 10)
                  << "    entropy bonus" << lfmt::num(r[3], 10)
                  << "    mean entropy" << lfmt::num(r[2], 8) << "\n"
                  << "  " << lfmt::label("")
                  << "advantage (clipped, per set)  mean" << lfmt::num(av[0], 9)
                  << "    std" << lfmt::num(av[1], 9)
                  << "    min" << lfmt::num(av[2], 9)
                  << "    max" << lfmt::num(av[3], 9)
                  << "    clipped" << lfmt::num(100.0f * av[4], 7) << "%\n"
                  << "  " << lfmt::label("")
                  << lfmt::label("head", 14) << "    entropy     coef      sets\n";
                for (int h = 0; h < NUM_HEADS; ++h) {
                    if (head_dead(h)) continue;
                    o << "  " << lfmt::label("") << lfmt::label(HEAD_NAMES[h], 14);
                    if (hc[h] > 0.0f)
                        o << lfmt::num(hm[h], 11) << lfmt::num(head_entropy[h], 9)
                          << std::setw(10) << (long)hc[h] << "\n";
                    else
                        o << "    (not drawn in this minibatch)\n";
                }
                std::cout << o.str();
            }

            // One backward per model, then the value. The two policy passes
            // share the logit gather, so the first keeps the graph. The value
            // has its own body and shares nothing with them.
            ((pl_cell - ent_cell) / GRAD_ACCUM_STEPS).backward({}, /*retain_graph=*/true);
            ((pl_glob - ent_glob) / GRAD_ACCUM_STEPS).backward();
            (VALUE_COEF * value_loss / GRAD_ACCUM_STEPS).backward();
            ++accum_counter;

            if (accum_counter % GRAD_ACCUM_STEPS == 0) {
                // Clip each model to its own norm. clip_grad_norm_ returns the
                // norm BEFORE clipping. A non-finite norm means a NaN or inf
                // got into that model's gradient: the step is skipped whole,
                // since clipping a NaN only spreads it into the weights.
                double gn[NUM_CG];
                bool finite = true;
                for (int g = 0; g < NUM_CG; ++g) {
                    gn[g] = torch::nn::utils::clip_grad_norm_(clip_params[g], CG_CLIP[g]);
                    if (!std::isfinite(gn[g])) finite = false;
                }
                if (finite) {
                    optimizer.step();
                    ++gn_steps;
                    for (int g = 0; g < NUM_CG; ++g) {
                        gn_sum[g] += gn[g];
                        gn_max[g] = std::max(gn_max[g], gn[g]);
                        if (gn[g] > CG_CLIP[g]) ++gn_clipped[g];
                    }
                } else {
                    ++gn_skipped;
                    std::cerr << "  grad: non-finite norm (cell " << gn[CG_CELL]
                              << ", global " << gn[CG_GLOB] << ", value " << gn[CG_VAL]
                              << "), step skipped" << std::endl;
                }
                optimizer.zero_grad();
            }
        }

        // ---- gradient norms: every iteration to the CSV, every 10th printed
        {
            const int n = std::max(1, gn_steps);
            grad_log << iteration << "," << gn_steps << "," << gn_skipped;
            for (int g = 0; g < NUM_CG; ++g)
                grad_log << "," << (gn_sum[g] / n) << "," << gn_max[g] << ","
                         << ((double)gn_clipped[g] / n);
            grad_log << std::endl;
            if (iteration % 10 == 0) {
                std::ostringstream o;
                o << "  " << lfmt::label("grad norm");
                for (int g = 0; g < NUM_CG; ++g)
                    o << CG_NAME[g] << lfmt::num(gn_sum[g] / n, 9)
                      << " (max" << lfmt::num(gn_max[g], 9) << ", clipped "
                      << gn_clipped[g] << "/" << gn_steps << ")   ";
                if (gn_skipped) o << "skipped " << gn_skipped;
                std::cout << o.str() << "\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  " << lfmt::label("time") << std::fixed << std::setprecision(1)
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - start_time).count() / 1000.0
                  << " s" << std::defaultfloat << "\n" << std::endl;

        if (iteration % 400 == 0)
            torch::save(player, "model_macro_" + std::to_string(iteration) + ".pt");

        if (iteration % EVAL_EVERY == 0) {
            // Is it still learning: vs an old checkpoint of itself.
            if (eval_vs(player, frozen, 200, iteration, winrate_log) >= 0.85f) {
                std::cout << "  promoting frozen opponent" << std::endl;
                copy_model(player, frozen);
            }
            // Is it getting better against real players: the fixed bench.
            eval_vs_tapes(player, bench, iteration, bench_log, winrate_log);
        }
    }
    return 0;
}