#ifndef GAZE_H
#define GAZE_H

/*
 * Gaze into the Pattern: Characterizing Spatial Patterns with Internal Temporal Correlations for Hardware Prefetching
 *
 * To appear in 31st IEEE International Symposium on High-Performance Computer Architecture (HPCA 2025),
 * 3/1/2025-3/5/2025, Las Vegas, NV, USA
 *
 * @Authors: Zixiao Chen, Chentao Wu, Yunfei Gu, Ranhao Jia, Jie Li, and Minyi Guo
 * @Manteiners: Zixiao Chen
 * @Email: chen_zx@sjtu.edu.cn
 * @Date: 12/02/2024
 *
 * Modified: Added Cross-Region Temporal Chaining (RTT - Region Transition Table)
 *           See sections marked with "===== RTT START =====" / "===== RTT END ====="
 */

#include "custom_util.h"
#include "cache.h"

#include <stdint.h>
#include <random>
#include <deque>

namespace gaze {

#define __region_offset(block_num) (block_num & REGION_OFFSET_MASK)

#define FT_TYPE custom_util::SRRIPSetAssociativeCache
#define AT_TYPE custom_util::LRUSetAssociativeCache
#define PT_TYPE custom_util::LRUSetAssociativeCache
#define PB_TYPE custom_util::LRUSetAssociativeCache

// ===== RTT START =====
#define RTT_TYPE custom_util::LRUSetAssociativeCache
// ===== RTT END =====

constexpr uint64_t REGION_SIZE = 4 * 1024; // '4KB', '8KB', '16KB, '32KB', '64KB, '128KB', '512KB', '1024KB', '2048KB'
constexpr uint64_t LOG2_REGION_SIZE = champsim::lg2(REGION_SIZE);
constexpr uint64_t REGION_OFFSET_MASK = (1ULL << (LOG2_REGION_SIZE - LOG2_BLOCK_SIZE)) - 1;

constexpr int NUM_BLOCKS = REGION_SIZE / BLOCK_SIZE;

constexpr int FT_SIZE = 64, FT_WAY = 8;
constexpr int AT_SIZE = 64, AT_WAY = 8;
constexpr int PT_WAY = 4;
constexpr int PT_SIZE = PT_WAY * NUM_BLOCKS;
constexpr int PB_SIZE = 32, PB_WAY = 8;

constexpr int STRIDE_PF_LOOKAHEAD = 2;
constexpr int PF_FILL_L1 = 1;
constexpr int PF_FILL_L2 = 2;

// ===== RTT START =====
// RTT (Region Transition Table) configuration.
// Captures cross-region temporal correlations: when region R_{n-1} ended with
// last_trigger T_prev and the new region R_n was activated with (T_curr, S_curr),
// we associate that 3-tuple with R_n's eventual learned pattern. This allows
// us to predict the pattern of R_n earlier and more accurately when the same
// transition recurs.
constexpr int RTT_SIZE = 64;
constexpr int RTT_WAY = 8;
constexpr int RTT_CONF_MAX = 3;        // 2-bit saturating counter (0..3)
constexpr int RTT_CONF_USE_THRESH = 2; // require confidence >= 2 to use RTT pattern
// ===== RTT END =====

// ------------------------- Util Functions ------------------------- //
std::pair<float, float> calculate_acc_and_cov(std::vector<int> pattern_1, std::vector<int> pattern_2);
bool different_patterns(std::vector<int> pattern_1, std::vector<int> pattern_2);
std::vector<int> pattern_bool2int(std::vector<bool> pattern);
bool pattern_all_set(std::vector<bool> pattern);
bool pattern_all_set(std::vector<int> pattern);

// ------------------------- Filter Table ------------------------- //
struct FilterTableData {
    uint64_t trigger_offset;
    uint64_t pc;
};

class FilterTable : public FT_TYPE<FilterTableData> {
    typedef FT_TYPE<FilterTableData> Super;

private:
    uint64_t build_key(uint64_t region_num);
    void write_data(Entry& entry, custom_util::Table& table, int row);

public:
    FilterTable(int size, int num_ways);

    Entry* find(uint64_t region_num);
    void insert(uint64_t region_num, uint64_t trigger_offset, uint64_t pc);
    Entry* erase(uint64_t region_num);

    std::string log();
};

// ------------------------- Accumulate Table ------------------------- //
struct AccumulateTableData {
    uint64_t trigger_offset;
    uint64_t second_offset;
    uint64_t pc;
    bool missed_in_pt;
    std::vector<bool> pattern;
    std::vector<int> order;

    int last_stride;
    uint64_t last_offset;

    bool con = false;

    int timestamp = 2;

    // ===== RTT START =====
    // Prev-region trigger captured at the moment this region was activated.
    // Used as the first component of the RTT key when this region is later
    // evicted and its (now-final) pattern is used to train the RTT.
    bool     rtt_prev_valid   = false;
    uint64_t rtt_prev_trigger = 0;
    // ===== RTT END =====
};

class AccumulateTable : public AT_TYPE<AccumulateTableData> {
    typedef AT_TYPE<AccumulateTableData> Super;

private:
    bool __stride_prefetch = false;

    void write_data(Entry& entry, custom_util::Table& table, int row);
    uint64_t build_key(uint64_t region_num);

public:
    bool get_stride_prefetch();
    void turn_off_stride_prefetch();

public:
    AccumulateTable(int size, int num_ways);

    Entry* set_pattern(uint64_t region_num, uint64_t offset);

    Entry insert(uint64_t region_num, uint64_t trigger_offset, uint64_t second_offset, uint64_t pc, bool missed_in_pt, bool con,
                 // ===== RTT START =====
                 bool rtt_prev_valid = false, uint64_t rtt_prev_trigger = 0
                 // ===== RTT END =====
    );
    Entry* erase(uint64_t region_num);

    std::string log();
};

// ------------------------- Pattern Table ------------------------- //
struct PatternTableData {
    std::vector<int> pattern;
    uint64_t pc;
    bool con = false;

    PatternTableData() {
        pattern = std::vector<int>(NUM_BLOCKS, 0);
        pc = 0;
    }
    PatternTableData(std::vector<int> pattern, uint64_t pc) :
        pattern(pattern), pc(pc) {}
    PatternTableData(std::vector<int> pattern, uint64_t pc, bool con) :
        pattern(pattern), pc(pc), con(con) {}
};

class PatternTable : public PT_TYPE<PatternTableData> {
    typedef PT_TYPE<PatternTableData> Super;

private:
    void write_data(Entry& entry, custom_util::Table& table, int row);
    uint64_t build_key(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num);

public:
    std::deque<uint64_t> con_pc;
    int con_counter = 0;

    PatternTable(int size, int num_ways);

    void insert(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num, std::vector<bool> pattern);
    Entry* find(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num);

    std::string log();
};

// ------------------------- Prefetch Buffer ------------------------- //
struct PrefetchBufferData {
public:
    std::vector<int> pattern;
    uint64_t trigger;
    uint64_t second;
    std::vector<int> pf_metadata;
};

class PrefetchBuffer : public PB_TYPE<PrefetchBufferData> {
    typedef PB_TYPE<PrefetchBufferData> Super;

private:
    int pattern_len;

    void write_data(Entry& entry, custom_util::Table& table, int row);
    uint64_t build_key(uint64_t region_num);

public:
    bool warmup;

    PrefetchBuffer(int size, int pattern_len, int debug_level, int num_ways);

    void insert(uint64_t region_num, std::vector<int> pattern, uint64_t trigger, uint64_t second, uint32_t pf_metadata);
    void prefetch(CACHE* cache, uint64_t block_num);

    std::string log();
};

// ===== RTT START =====
// ------------------------- Region Transition Table ------------------------- //
//
// RTT learns cross-region temporal chains. Each entry maps a transition
// signature (prev_trigger, curr_trigger, curr_second) to a learned bit-vector
// pattern for the *current* region. A small saturating confidence counter
// gates whether the stored pattern is trusted enough to be used for
// prefetching (preventing pollution from one-shot transitions).
//
// Key layout (18 bits packed):
//   bits [ 5: 0] = prev_trigger  (offset within previous region)
//   bits [11: 6] = curr_trigger  (offset within current region)
//   bits [17:12] = curr_second   (second offset within current region)
struct RegionTransitionTableData {
    std::vector<int> pattern;   // PF_FILL_L1 / PF_FILL_L2 / 0 - same encoding as PHT pattern
    int  confidence;            // saturating counter [0, RTT_CONF_MAX]

    RegionTransitionTableData() :
        pattern(NUM_BLOCKS, 0), confidence(0) {}

    RegionTransitionTableData(std::vector<int> p, int c) :
        pattern(std::move(p)), confidence(c) {}
};

class RegionTransitionTable : public RTT_TYPE<RegionTransitionTableData> {
    typedef RTT_TYPE<RegionTransitionTableData> Super;

private:
    void write_data(Entry& entry, custom_util::Table& table, int row);
    uint64_t build_key(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second);

public:
    // Statistics (also exposed via getters for the prefetcher's final stats dump)
    uint64_t stat_lookups          = 0;
    uint64_t stat_hits             = 0;
    uint64_t stat_inserts          = 0;
    uint64_t stat_updates          = 0;
    uint64_t stat_used             = 0; // times an RTT pattern was actually used for prefetching
    uint64_t stat_skipped_low_conf = 0;

    RegionTransitionTable(int size, int num_ways);

    // Lookup current entry (does not modify confidence). Returns nullptr on miss.
    Entry* find(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second);

    // Insert or update. If the entry exists and its stored pattern overlaps
    // with the new pattern significantly, confidence is bumped; otherwise the
    // pattern is overwritten and confidence is reset. New entries start at 1.
    void update(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second,
                const std::vector<int>& new_pattern);

    // Decrement the confidence of an entry on a misprediction signal (optional).
    void penalize(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second);

    std::string log();
};
// ===== RTT END =====

// ------------------------- Gaze Prefetcher ------------------------- //
class Gaze {
private:
    int cpu;
    int stride_pf_degree = 4;

    FilterTable ft;
    AccumulateTable at;
    PatternTable pt;
    PrefetchBuffer pb;

    // ===== RTT START =====
    RegionTransitionTable rtt;

    // Per-prefetcher tracking of the most recently completed region.
    // We learn (prev_trigger, curr_trigger, curr_second) -> pattern, so we
    // need to remember the prev region's trigger when its tracking ends.
    bool     rtt_have_prev    = false;
    uint64_t rtt_prev_region  = 0;
    uint64_t rtt_prev_trigger = 0;

    // Top-level (Gaze-side) counters for visibility in final stats.
    uint64_t rtt_used_for_pf = 0;   // RTT pattern was inserted into PB
    uint64_t rtt_fallback    = 0;   // RTT lookup happened but PHT path was used
    uint64_t rtt_no_prev     = 0;   // could not consult RTT - no prior region

    // Helper - update prev trigger bookkeeping when a region's tracking ends.
    void rtt_remember_prev(uint64_t region_num, uint64_t trigger);
    // ===== RTT END =====

    PatternTable::Entry* find_in_pt(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num);
    void insert_in_pt(const AccumulateTable::Entry& entry, uint64_t region_num);

public:
    int global_level = 0;
    bool warmup;

    Gaze(int ft_size, int ft_ways, int at_size, int at_ways, int pt_size, int pt_ways, int pb_size, int pb_ways, int cpu);
    Gaze();
    void set_warmup(bool warmup);

    void access(uint64_t block_num, uint64_t ip, CACHE* cache);
    void eviction(uint64_t block_num);
    void prefetch(CACHE* cache, uint64_t block_num);

    void stride_prefetch(uint64_t block_num);
    void pattern_prefetch(uint64_t trigger, uint64_t second);

    void tune_stride_degree(CACHE* cache);

    void log();
};

} // namespace gaze

#endif