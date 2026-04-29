#include "gaze_rtt.h"
#include "cache.h"

#include <assert.h>
#include <iostream>

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

static std::vector<gaze_rtt::Gaze> prefetchers;

namespace gaze_rtt {

// ------------------------- FT functions ------------------------- //

FilterTable::FilterTable(int size, int num_ways) :
    Super(size, num_ways) {}

FilterTable::Entry* FilterTable::find(uint64_t region_num) {
    uint64_t key = build_key(region_num);
    Entry* entry = Super::find(key);
    if (!entry) {
        return nullptr;
    } else {
        Super::rp_promote(key);
        return entry;
    }
}

void FilterTable::insert(uint64_t region_num, uint64_t trigger_offset, uint64_t pc) {
    uint64_t key = build_key(region_num);
    auto entry = Super::insert(key, {trigger_offset, pc});
    Super::rp_insert(key);
}

FilterTable::Entry* FilterTable::erase(uint64_t region_num) {
    uint64_t key = build_key(region_num);
    return Super::erase(key);
}

std::string FilterTable::log() {
    std::vector<std::string> headers({"RegionNum", "Trigger", "PC"});
    return Super::log(headers);
}

uint64_t FilterTable::build_key(uint64_t region_num) {
    uint64_t key = region_num & ((1ULL << 37) - 1);
    return custom_util::hash_index(key, this->index_len);
}

void FilterTable::write_data(Entry& entry, custom_util::Table& table, int row) {
    uint64_t key = custom_util::hash_index(entry.key, this->index_len);
    table.set_cell(row, 0, key);
    table.set_cell(row, 1, entry.data.trigger_offset);
    table.set_cell(row, 2, entry.data.pc);
}

// ------------------------- AT functions ------------------------- //
AccumulateTable::AccumulateTable(int size, int num_ways) :
    Super(size, num_ways) {}

AccumulateTable::Entry* AccumulateTable::set_pattern(uint64_t region_num, uint64_t offset) {
    uint64_t key = build_key(region_num);
    Entry* entry = Super::find(key);
    if (!entry)
        return nullptr;
    else {
        if (!entry->data.pattern[offset]) {
            entry->data.timestamp++;
            int stride = int(offset) - int(entry->data.last_offset);
            if (entry->data.missed_in_pt || entry->data.con)
                this->__stride_prefetch = (stride == entry->data.last_stride);
            entry->data.order[offset] = entry->data.timestamp;
            entry->data.pattern[offset] = true;
            entry->data.last_offset = offset;
            entry->data.last_stride = stride;
        }
        Super::rp_promote(key);
        return entry;
    }
}

AccumulateTable::Entry AccumulateTable::insert(uint64_t region_num, uint64_t trigger_offset, uint64_t second_offset, uint64_t pc, bool missed_in_pt, bool con,
                                               // ===== RTT START =====
                                               bool rtt_prev_valid, uint64_t rtt_prev_trigger
                                               // ===== RTT END =====
) {
    uint64_t key = build_key(region_num);
    std::vector<bool> pattern(NUM_BLOCKS, false);
    std::vector<int> order(NUM_BLOCKS, 0);
    pattern[trigger_offset] = pattern[second_offset] = true;
    order[trigger_offset] = 1;
    order[second_offset] = 2;
    int last_stride = int(second_offset) - int(trigger_offset);
    AccumulateTableData d{trigger_offset, second_offset, pc, missed_in_pt, pattern, order, last_stride, second_offset, con};
    // ===== RTT START =====
    d.rtt_prev_valid   = rtt_prev_valid;
    d.rtt_prev_trigger = rtt_prev_trigger;
    // ===== RTT END =====
    Entry old_entry = Super::insert(key, d);
    Super::rp_insert(key);
    return old_entry;
}

AccumulateTable::Entry* AccumulateTable::erase(uint64_t region_num) {
    uint64_t key = build_key(region_num);
    return Super::erase(key);
}

std::string AccumulateTable::log() {
    std::vector<std::string> headers({"RegionNum", "Trigger", "Second", "PC", "Pattern", "Order"});
    return Super::log(headers);
}

void AccumulateTable::write_data(Entry& entry, custom_util::Table& table, int row) {
    uint64_t key = custom_util::hash_index(entry.key, this->index_len);
    table.set_cell(row, 0, key);
    table.set_cell(row, 1, entry.data.trigger_offset);
    table.set_cell(row, 2, entry.data.second_offset);
    table.set_cell(row, 3, entry.data.pc);
    table.set_cell(row, 4, custom_util::pattern_to_string(entry.data.pattern));
    table.set_cell(row, 5, custom_util::pattern_to_string(entry.data.order));
}

uint64_t AccumulateTable::build_key(uint64_t region_num) {
    uint64_t key = region_num & ((1ULL << 37) - 1);
    return custom_util::hash_index(key, this->index_len);
}

bool AccumulateTable::get_stride_prefetch() {
    return __stride_prefetch;
}

void AccumulateTable::turn_off_stride_prefetch() {
    __stride_prefetch = false;
}

// ------------------------- PHT functions ------------------------- //
PatternTable::PatternTable(int size, int num_ways) :
    Super(size, num_ways) {
    std::cout << "Pattern Table index_len: " << Super::index_len << std::endl;
}

void PatternTable::insert(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num, std::vector<bool> pattern) {
    assert(pattern[trigger] && pattern[second]);
    bool all_set = pattern_all_set(pattern);

    if (trigger != 0 || second != 1) { // not spatial streaming
        uint64_t key = build_key(trigger, second, pc, region_num);
        Super::insert(key, {pattern_bool2int(pattern), pc});
        Super::rp_insert(key);
    } else { // spatial streaming
        if (all_set) {
            if (con_counter < 8)
                con_counter++;
            uint64_t hashed_pc = custom_util::my_hash_index(pc, LOG2_BLOCK_SIZE, 8);
            if (con_pc.end() == std::find_if(con_pc.begin(), con_pc.end(), [hashed_pc](auto& x) { return x == hashed_pc; })) {
                if (con_pc.size() == 8)
                    con_pc.pop_back();
                con_pc.push_front(hashed_pc);
            }
        } else {
            // ===== RTT FIX START =====
            // Bug fix: the original `con_counter >> 1;` is a no-op (computes
            // and discards). The intent is to halve the counter on a non-dense
            // region, so use the compound assignment `>>=`.
            if (con_counter > 2)
                con_counter >>= 1;
            else if (con_counter > 0)
                con_counter--;
            // ===== RTT FIX END =====
        }
    }
}

PatternTable::Entry* PatternTable::find(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num) {
    if (trigger != 0 || second != 1) { // not ss
        uint64_t key = build_key(trigger, second, pc, region_num);
        return Super::find(key);
    } else {
        uint64_t hashed_pc = custom_util::my_hash_index(pc, LOG2_BLOCK_SIZE, 8);

        if (con_counter == 8 || con_pc.end() != std::find_if(con_pc.begin(), con_pc.end(), [hashed_pc](auto& x) { return x == hashed_pc; })) {
            Entry* ret = new Entry();
            ret->data.con = true;
            for (int i = 0; i < NUM_BLOCKS / 4; i++) {
                ret->data.pattern[i] = PF_FILL_L1;
            }
            for (int i = NUM_BLOCKS / 4; i < NUM_BLOCKS; i++) {
                ret->data.pattern[i] = PF_FILL_L2;
            }
            return ret;
        } else if (con_counter > 2) {
            Entry* ret = new Entry();
            ret->data.con = true;
            for (int i = 0; i < NUM_BLOCKS / 4; i++) {
                ret->data.pattern[i] = PF_FILL_L2;
            }
            return ret;
        }
        return nullptr;
    }
    assert(0);
}

std::string PatternTable::log() {
    std::vector<std::string> headers({"Trigger", "Second", "Pattern"});
    return Super::log(headers);
}

void PatternTable::write_data(Entry& entry, custom_util::Table& table, int row) {
    table.set_cell(row, 0, int(entry.key & uint64_t((1ULL << this->index_len) - 1)));
    table.set_cell(row, 1, int((entry.key >> this->index_len) & ((1ULL << this->index_len) - 1)));
    table.set_cell(row, 2, custom_util::pattern_to_string(entry.data.pattern));
}

uint64_t PatternTable::build_key(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num) {
    assert(trigger >= 0 && trigger < NUM_BLOCKS && second >= 0 && second < NUM_BLOCKS);
    uint64_t key = (second << this->index_len) | trigger;
    return key;
}

// ------------------------- PB functions ------------------------- //
PrefetchBuffer::PrefetchBuffer(int size, int pattern_len, int debug_level = 0, int num_ways = 16) :
    Super(size, num_ways), pattern_len(pattern_len) {
}

void PrefetchBuffer::insert(uint64_t region_num, std::vector<int> pattern, uint64_t trigger, uint64_t second, uint32_t pf_metadata) {
    uint64_t key = this->build_key(region_num);
    if ((pf_metadata & 3) == 0 || (pf_metadata & 3) == 3) { // stride & promote
        auto entry = find(key);
        if (!entry) {
            Super::insert(key, {pattern, trigger, trigger, std::vector<int>(gaze_rtt::NUM_BLOCKS, pf_metadata)});
            Super::rp_insert(key);
        } else {
            for (int i = 0; i < NUM_BLOCKS; i++) {
                if (pattern[i] == PF_FILL_L1) {
                    if (entry->data.pattern[i] != PF_FILL_L1) {
                        if (entry->data.pf_metadata[i] == 2) {
                            entry->data.pf_metadata[i] = 3;
                        }
                    }
                    entry->data.pattern[i] = PF_FILL_L1;
                }
            }
            Super::rp_promote(key);
        }
    } else { // con and dis
        Super::insert(key, {pattern, trigger, second, std::vector<int>(gaze_rtt::NUM_BLOCKS, pf_metadata)});
        Super::rp_insert(key);
    }
}

void PrefetchBuffer::prefetch(CACHE* cache, uint64_t block_num) {
    uint64_t region_offset = __region_offset(block_num);
    uint64_t region_num = block_num >> (LOG2_REGION_SIZE - LOG2_BLOCK_SIZE);
    uint64_t key = this->build_key(region_num);
    auto entry = Super::find(key);
    if (!entry) {
        return;
    }
    Super::rp_promote(key);
    std::vector<int>& pattern = entry->data.pattern;
    auto pf_metadatas = entry->data.pf_metadata;
    uint32_t pf_metadata = 0;
    uint64_t trigger = entry->data.trigger;
    uint64_t second = entry->data.second;

    pattern[region_offset] = 0;
    for (uint64_t i = 1; i < (REGION_SIZE / BLOCK_SIZE); i++) {
        uint64_t pf_offset = ((uint64_t)region_offset + i) % (REGION_SIZE / BLOCK_SIZE);
        if (pf_offset != trigger && pf_offset != second && pattern[pf_offset] != 0) {
            uint64_t pf_addr = (region_num << LOG2_REGION_SIZE) + (pf_offset << LOG2_BLOCK_SIZE);
            if (cache->get_occupancy(3, 0) + cache->get_occupancy(0, 0) < cache->get_size(0, 0) - 1 && cache->get_occupancy(3, 0) < cache->get_size(3, 0)) {
                pf_metadata = pf_metadatas[pf_offset];
                pf_metadata = __add_pf_sour_level(pf_metadata, 1);
                if (pattern[pf_offset] == PF_FILL_L1) {
                    pf_metadata = __add_pf_dest_level(pf_metadata, 1);
                } else {
                    pf_metadata = __add_pf_dest_level(pf_metadata, 2);
                }
                int ok = cache->prefetch_line(pf_addr, pattern[pf_offset] == PF_FILL_L1 ? true : false, pf_metadata);

                if (ok) {
                    pattern[pf_offset] = 0;
                }
            } else {
                return;
            }
        }
    }
    Super::erase(key);
    return;
}

std::string PrefetchBuffer::log() {
    std::vector<std::string> headers({"RegionNum", "Trigger", "Second", "Meta", "Pattern"});
    return Super::log(headers);
}

void PrefetchBuffer::write_data(Entry& entry, custom_util::Table& table, int row) {
    uint64_t key = custom_util::hash_index(entry.key, this->index_len);
    table.set_cell(row, 0, key);
    table.set_cell(row, 1, entry.data.trigger);
    table.set_cell(row, 2, entry.data.second);
    table.set_cell(row, 3, (uint64_t)entry.data.pf_metadata[0]);
    table.set_cell(row, 4, custom_util::pattern_to_string(entry.data.pattern));
}

uint64_t PrefetchBuffer::build_key(uint64_t region_num) {
    uint64_t key = region_num;
    return key;
}

// ===== RTT START =====
// ------------------------- RTT functions ------------------------- //
RegionTransitionTable::RegionTransitionTable(int size, int num_ways) :
    Super(size, num_ways) {}

uint64_t RegionTransitionTable::build_key(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second) {
    // ===== RTT FIX START =====
    // Per spec: key = (prev_trigger << 12) | (curr_trigger << 6) | curr_second
    // Each offset is 6 bits for a 4KB / 64B-line region.
    constexpr int OFF_BITS = LOG2_REGION_SIZE - LOG2_BLOCK_SIZE; // 6 for 4KB/64B
    constexpr uint64_t OFF_MASK = (1ULL << OFF_BITS) - 1;

    uint64_t key = ((prev_trigger & OFF_MASK) << (2 * OFF_BITS))
                 | ((curr_trigger & OFF_MASK) << OFF_BITS)
                 |  (curr_second  & OFF_MASK);
    return key;
    // ===== RTT FIX END =====
}

RegionTransitionTable::Entry* RegionTransitionTable::find(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second) {
    stat_lookups++;
    uint64_t key = build_key(prev_trigger, curr_trigger, curr_second);
    Entry* entry = Super::find(key);
    if (entry) {
        stat_hits++;
        Super::rp_promote(key);
    }
    return entry;
}

void RegionTransitionTable::update(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second,
                                   const std::vector<int>& new_pattern) {
    uint64_t key = build_key(prev_trigger, curr_trigger, curr_second);
    Entry* entry = Super::find(key);

    if (entry) {
        // Existing entry - measure overlap to decide whether to bump or reset confidence.
        int overlap = 0, set_old = 0, set_new = 0;
        for (int i = 0; i < NUM_BLOCKS; i++) {
            if (entry->data.pattern[i] != 0) set_old++;
            if (new_pattern[i]        != 0) set_new++;
            if (entry->data.pattern[i] != 0 && new_pattern[i] != 0) overlap++;
        }
        // Use coverage of the union as a similarity metric.
        int union_set = set_old + set_new - overlap;
        bool similar  = (union_set == 0) ||
                        (2 * overlap >= union_set); // overlap >= 50% of union

        if (similar) {
            // Reinforce: bump confidence (saturate at RTT_CONF_MAX) and OR-merge
            // L1 hints from the new pattern (taking the stronger signal).
            if (entry->data.confidence < RTT_CONF_MAX)
                entry->data.confidence++;
            for (int i = 0; i < NUM_BLOCKS; i++) {
                if (new_pattern[i] == PF_FILL_L1)
                    entry->data.pattern[i] = PF_FILL_L1;
                else if (new_pattern[i] == PF_FILL_L2 && entry->data.pattern[i] == 0)
                    entry->data.pattern[i] = PF_FILL_L2;
            }
        } else {
            // Different pattern under same key - be conservative: drop confidence,
            // and only overwrite the stored pattern when confidence has bottomed
            // out (this avoids thrashing under noisy transitions).
            if (entry->data.confidence > 0)
                entry->data.confidence--;
            if (entry->data.confidence == 0)
                entry->data.pattern = new_pattern;
        }
        stat_updates++;
        Super::rp_promote(key);
    } else {
        // New entry - start with confidence 1 (must be observed at least one
        // more time before being trusted enough to drive prefetches).
        Super::insert(key, {new_pattern, 1});
        Super::rp_insert(key);
        stat_inserts++;
    }
}

void RegionTransitionTable::penalize(uint64_t prev_trigger, uint64_t curr_trigger, uint64_t curr_second) {
    uint64_t key = build_key(prev_trigger, curr_trigger, curr_second);
    Entry* entry = Super::find(key);
    if (entry && entry->data.confidence > 0) {
        entry->data.confidence--;
    }
}

std::string RegionTransitionTable::log() {
    std::vector<std::string> headers({"PrevTrig", "CurTrig", "CurSecond", "Conf", "Pattern"});
    return Super::log(headers);
}

void RegionTransitionTable::write_data(Entry& entry, custom_util::Table& table, int row) {
    // ===== RTT FIX START =====
    // Decode key per the spec packing: prev<<12 | curr<<6 | second
    constexpr int OFF_BITS = LOG2_REGION_SIZE - LOG2_BLOCK_SIZE;
    constexpr uint64_t OFF_MASK = (1ULL << OFF_BITS) - 1;
    uint64_t cur_s  =  entry.key                          & OFF_MASK;
    uint64_t cur_t  = (entry.key >> OFF_BITS)             & OFF_MASK;
    uint64_t prev_t = (entry.key >> (2 * OFF_BITS))       & OFF_MASK;
    table.set_cell(row, 0, prev_t);
    table.set_cell(row, 1, cur_t);
    table.set_cell(row, 2, cur_s);
    table.set_cell(row, 3, entry.data.confidence);
    table.set_cell(row, 4, custom_util::pattern_to_string(entry.data.pattern));
    // ===== RTT FIX END =====
}
// ===== RTT END =====

// ------------------------- Gaze functions ------------------------- //

Gaze::Gaze(int ft_size, int ft_ways, int at_size, int at_ways, int pt_size, int pt_ways, int pb_size, int pb_ways, int cpu = 0) :
    cpu(cpu),
    ft(ft_size, ft_ways), at(at_size, at_ways), pt(pt_size, pt_ways), pb(pb_size, pb_ways)
    // ===== RTT START =====
    , rtt(RTT_SIZE, RTT_WAY)
    // ===== RTT END =====
{
}

// ===== RTT START =====
void Gaze::rtt_remember_prev(uint64_t region_num, uint64_t trigger) {
    rtt_have_prev    = true;
    rtt_prev_region  = region_num;
    rtt_prev_trigger = trigger;
}
// ===== RTT END =====

void Gaze::access(uint64_t block_num, uint64_t pc, CACHE* cache) {
    uint64_t region_num = block_num >> (LOG2_REGION_SIZE - LOG2_BLOCK_SIZE);
    uint64_t region_offset = __region_offset(block_num);

    // ===== RTT FIX START =====
    // Detect a cross-region transition at the very top of access(). We do not
    // overwrite `rtt_prev_trigger` here - that field is the *trigger of the
    // previous region*, and is meaningful only when set by rtt_remember_prev()
    // at second-offset arrival of the previous region. Here we just note that
    // we've crossed into a new region for bookkeeping.
    if (rtt_have_prev && region_num != rtt_prev_region) {
        // Genuine cross-region transition - the prev_trigger captured by
        // rtt_remember_prev() for `rtt_prev_region` is now eligible to drive
        // the RTT key for the *current* region.
    } else if (!rtt_have_prev) {
        // First time we ever see anything - we don't have a prev yet. The
        // first call to rtt_remember_prev() will set this to true.
    }
    // ===== RTT FIX END =====

    auto at_entry = this->at.set_pattern(region_num, region_offset);
    if (at_entry) {
        if (at.get_stride_prefetch()) {
            int stride = at_entry->data.last_stride;
            int begin_offset = at_entry->data.last_offset;
            at_entry->data.last_offset = at_entry->data.last_stride = 0;
            std::vector<int> pattern(NUM_BLOCKS, 0);
            for (int i = 1; i <= stride_pf_degree; i++) {
                if (begin_offset + (i + STRIDE_PF_LOOKAHEAD) * stride < NUM_BLOCKS && begin_offset + (i + STRIDE_PF_LOOKAHEAD) * stride >= 0) {
                    if (!(at_entry->data.pattern[begin_offset + (i + STRIDE_PF_LOOKAHEAD) * stride]))
                        pattern[begin_offset + (i + STRIDE_PF_LOOKAHEAD) * stride] = PF_FILL_L1;
                }
            }
            if (at_entry->data.missed_in_pt)
                pb.insert(region_num, pattern, begin_offset, begin_offset, 0);
            else if (at_entry->data.con)
                pb.insert(region_num, pattern, begin_offset, begin_offset, 3);
            at.turn_off_stride_prefetch();
        }
        return;
    } else {
        auto entry = ft.find(region_num);
        if (!entry) {
            ft.insert(region_num, region_offset, pc);
            return;
        } else if (entry->data.trigger_offset != region_offset) { // SECOND OFFSET
            uint64_t trigger_offset = entry->data.trigger_offset;
            uint64_t second_offset  = region_offset;

            // 1. find pattern via the standard PHT path
            auto pt_entry = find_in_pt(trigger_offset, second_offset, pc, region_num);
            // pattern empty?
            bool pattern_empty = (!pt_entry) || (2 == std::count_if(pt_entry->data.pattern.begin(), pt_entry->data.pattern.end(), [](auto& x) { return x != 0; }));
            bool all_set = pattern_empty ? false : pattern_all_set(pt_entry->data.pattern);

            // ===== RTT START =====
            // Try the RTT path. We only consult the RTT when we know the
            // *previous* region's last trigger and we're crossing into a
            // genuinely different region (rtt_prev_region != region_num).
            // The RTT pattern is preferred when:
            //   (a) the PHT had no useful pattern (pattern_empty), OR
            //   (b) the RTT entry is highly confident (== RTT_CONF_MAX).
            // Otherwise we fall back to the original Gaze behavior.
            //
            // Important: we deliberately do NOT modify `pattern_empty` based
            // on whether RTT supplied a pattern. `pattern_empty` is later fed
            // to AT as `missed_in_pt`, which gates the AT-side stride backup
            // prefetcher. Keeping that backup active when RTT is the only
            // source of the pattern is correct - the RTT prediction may be
            // wrong, and the stride backup is our recovery path.
            bool used_rtt = false;
            if (rtt_have_prev && rtt_prev_region != region_num) {
                auto rtt_entry = rtt.find(rtt_prev_trigger, trigger_offset, second_offset);
                if (rtt_entry) {
                    if (rtt_entry->data.confidence >= RTT_CONF_USE_THRESH) {
                        bool rtt_better = pattern_empty ||
                                          (rtt_entry->data.confidence == RTT_CONF_MAX && !all_set);
                        if (rtt_better) {
                            // Make sure the trigger and second offsets are
                            // marked (the PB will skip them on issue, but the
                            // pattern still needs to be self-consistent).
                            std::vector<int> rtt_pattern = rtt_entry->data.pattern;
                            if (rtt_pattern[trigger_offset] == 0) rtt_pattern[trigger_offset] = PF_FILL_L1;
                            if (rtt_pattern[second_offset]  == 0) rtt_pattern[second_offset]  = PF_FILL_L1;

                            // pf_metadata = 1 -> regular (non-streaming, non-stride) insert
                            pb.insert(region_num, rtt_pattern, trigger_offset, second_offset, 1);
                            rtt.stat_used++;
                            rtt_used_for_pf++;
                            used_rtt = true;
                        }
                    } else {
                        rtt.stat_skipped_low_conf++;
                    }
                }
            } else {
                rtt_no_prev++;
            }

            // Only fall back to the PHT path if RTT did NOT already supply a
            // prefetch pattern. Otherwise we'd risk double-inserting into PB.
            if (!used_rtt) {
                if (!pattern_empty) {
                    uint32_t pf_metadata = pt_entry->data.con ? 2 : 1;
                    pb.insert(region_num, pt_entry->data.pattern, trigger_offset, second_offset, pf_metadata);
                }
                if (rtt_have_prev && rtt_prev_region != region_num) {
                    rtt_fallback++;
                }
            }
            // ===== RTT END =====

            // 2. insert into at
            // ===== RTT START =====
            // Snapshot the current rtt_prev_* into the AT entry so that when
            // this region is later evicted from AT, we can train the RTT with
            // the correct (prev_trigger -> trigger -> second) -> pattern
            // tuple. A region transition is meaningful only when the prev
            // region differs from the current one.
            bool snapshot_valid = rtt_have_prev && (rtt_prev_region != region_num);
            uint64_t snapshot_prev_trigger = rtt_prev_trigger;
            // ===== RTT END =====
            auto at_victim = at.insert(region_num, trigger_offset, second_offset, entry->data.pc,
                                       pattern_empty, (!pattern_empty) && pt_entry && pt_entry->data.con,
                                       // ===== RTT START =====
                                       snapshot_valid, snapshot_prev_trigger
                                       // ===== RTT END =====
            );
            ft.erase(region_num);
            if (at_victim.valid) {
                insert_in_pt(at_victim, region_num);
                // ===== RTT START =====
                // Train RTT using the victim's *own* stored prev trigger -
                // not the live rtt_prev_*, which now points to the region
                // that just displaced the victim.
                if (at_victim.data.rtt_prev_valid) {
                    rtt.update(at_victim.data.rtt_prev_trigger,
                               at_victim.data.trigger_offset,
                               at_victim.data.second_offset,
                               pattern_bool2int(at_victim.data.pattern));
                }
                // ===== RTT END =====
            }

            // ===== RTT START =====
            // Now that this region is being tracked, remember its trigger so
            // the *next* region transition can look up the chain.
            rtt_remember_prev(region_num, trigger_offset);
            // ===== RTT END =====
        }
    }
}

void Gaze::eviction(uint64_t block_num) {
    uint64_t region_num = block_num >> (LOG2_REGION_SIZE - LOG2_BLOCK_SIZE);
    ft.erase(region_num);
    auto entry = at.erase(region_num);
    if (entry) {
        insert_in_pt(*entry, region_num);
        // ===== RTT FIX START =====
        // Train RTT with the now-final pattern of this region. The prev
        // trigger we use is the snapshot captured when this AT entry was
        // first inserted (in access()'s second-offset path), NOT the current
        // live rtt_prev_trigger - that field has likely advanced past this
        // region by now. The pattern is stored as vector<bool> in the AT but
        // RTT expects vector<int>, so we convert via pattern_bool2int().
        if (entry->data.rtt_prev_valid) {
            std::vector<int> pattern_int = pattern_bool2int(entry->data.pattern);
            rtt.update(entry->data.rtt_prev_trigger,
                       entry->data.trigger_offset,
                       entry->data.second_offset,
                       pattern_int);
        }
        // ===== RTT FIX END =====
    }
}

void Gaze::prefetch(CACHE* cache, uint64_t block_num) {
    // ===== RTT FIX START =====
    // RTT-first prefetch path. Per the spec, RTT is consulted BEFORE the
    // pattern-buffer-driven (PB) prefetch issues anything. If we find a
    // confident RTT entry whose key matches the (prev_trigger, trigger,
    // second) triple of the region currently under tracking, we install its
    // pattern into the PB and let the PB issue the actual prefetches in this
    // very call.
    //
    // Note: the spec asks us to call `at.find(region)`. The public AT API
    // exposes `set_pattern(region, offset)` instead - it returns the entry
    // pointer on a tracked region and only mutates the bit at `offset`,
    // which is the block we are currently demanding (so it's the same
    // mutation `access()` would do on the next observation). We use that as
    // the lookup primitive without modifying the header.
    uint64_t region_num    = block_num >> (LOG2_REGION_SIZE - LOG2_BLOCK_SIZE);
    uint64_t region_offset = __region_offset(block_num);

    auto at_entry = at.set_pattern(region_num, region_offset);
    if (at_entry && at_entry->data.rtt_prev_valid) {
        uint64_t prev_trigger = at_entry->data.rtt_prev_trigger;
        uint64_t curr_trigger = at_entry->data.trigger_offset;
        uint64_t curr_second  = at_entry->data.second_offset;

        auto rtt_entry = rtt.find(prev_trigger, curr_trigger, curr_second);
        if (rtt_entry && rtt_entry->data.confidence >= RTT_CONF_USE_THRESH) {
            // Install RTT pattern into the PB and run the PB issue path so
            // prefetches actually go out this cycle. pf_metadata = 0 matches
            // the spec ("pb.insert(region, pattern, curr_trigger,
            // curr_second, 0)").
            //
            // We make a local copy of the pattern so PB-side mutations
            // (clearing issued bits) don't affect the persistent RTT entry.
            std::vector<int> pattern_copy = rtt_entry->data.pattern;

            // Make sure the trigger and second offsets are non-zero so the
            // PB's self-skip logic remains consistent.
            if (pattern_copy[curr_trigger] == 0) pattern_copy[curr_trigger] = PF_FILL_L1;
            if (pattern_copy[curr_second]  == 0) pattern_copy[curr_second]  = PF_FILL_L1;

            pb.insert(region_num, pattern_copy, curr_trigger, curr_second, 0);

            rtt.stat_used++;
            rtt_used_for_pf++;

            // Issue the prefetches for the freshly-installed pattern, then
            // return. Per spec: "RETURN immediately (this is CRITICAL)".
            pb.prefetch(cache, block_num);
            return;
        } else {
            // RTT either missed or confidence was below threshold - fall
            // through to the normal PB path.
            rtt_fallback++;
        }
    }
    // ===== RTT FIX END =====

    pb.prefetch(cache, block_num);
}

void Gaze::log() {
    std::cout << "Filter table begin" << std::dec << std::endl;
    std::cout << this->ft.log();
    std::cout << "Filter table end" << std::endl;

    std::cout << "Accumulation table begin" << std::dec << std::endl;
    std::cout << this->at.log();
    std::cout << "Accumulation table end" << std::endl;

    std::cout << "Pattern table begin" << std::dec << std::endl;
    std::cout << this->pt.log();
    std::cout << "Pattern table end" << std::endl;

    std::cout << "Prefetch buffer begin" << std::dec << std::endl;
    std::cout << this->pb.log();
    std::cout << "Prefetch buffer end" << std::endl;

    // ===== RTT START =====
    std::cout << "Region transition table begin" << std::dec << std::endl;
    std::cout << this->rtt.log();
    std::cout << "Region transition table end" << std::endl;

    std::cout << "RTT stats (cpu " << cpu << "):" << std::dec << std::endl;
    std::cout << "  RTT lookups          : " << rtt.stat_lookups          << std::endl;
    std::cout << "  RTT hits             : " << rtt.stat_hits             << std::endl;
    std::cout << "  RTT inserts          : " << rtt.stat_inserts          << std::endl;
    std::cout << "  RTT updates          : " << rtt.stat_updates          << std::endl;
    std::cout << "  RTT used for prefetch: " << rtt.stat_used             << std::endl;
    std::cout << "  RTT skipped (low conf): " << rtt.stat_skipped_low_conf << std::endl;
    std::cout << "  Gaze used RTT path   : " << rtt_used_for_pf           << std::endl;
    std::cout << "  Gaze fallback to PHT : " << rtt_fallback              << std::endl;
    std::cout << "  Gaze no-prev events  : " << rtt_no_prev               << std::endl;
    if (rtt.stat_lookups > 0) {
        double hit_rate = (double)rtt.stat_hits / (double)rtt.stat_lookups;
        std::cout << "  RTT hit rate         : " << hit_rate << std::endl;
    }
    // ===== RTT END =====
}

void Gaze::tune_stride_degree(CACHE* cache) {}

PatternTable::Entry* Gaze::find_in_pt(uint64_t trigger, uint64_t second, uint64_t pc, uint64_t region_num) {
    auto entry = pt.find(trigger, second, pc, region_num);
    if (!entry) {
        return entry;
    } else {
        return entry;
    }
}

void Gaze::insert_in_pt(const AccumulateTable::Entry& entry, uint64_t region_num) {
    uint64_t trigger = entry.data.trigger_offset, second = entry.data.second_offset, pc = entry.data.pc;
    pt.insert(trigger, second, pc, region_num, entry.data.pattern);
}

void Gaze::set_warmup(bool warmup) {
    this->warmup = warmup;
    this->pb.warmup = warmup;
}

// ------------------------- util functions ------------------------- //
std::pair<float, float> calculate_acc_and_cov(std::vector<int> pattern_1, std::vector<int> pattern_2) {
    uint64_t num_bits_1 = 0, num_bits_2 = 0, num_same = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        if (pattern_1[i] != 0)
            num_bits_1++;
        if (pattern_2[i] != 0)
            num_bits_2++;
        if (pattern_1[i] != 0 || pattern_2[i] != 0) {
            if (pattern_1[i] != 0 && pattern_2[i] != 0)
                num_same++;
        }
    }

    return {(float)num_same / num_bits_1, (float)num_same / num_bits_2};
}

bool different_patterns(std::vector<int> pattern_1, std::vector<int> pattern_2) {
    auto [acc, cov] = calculate_acc_and_cov(pattern_1, pattern_2);
    if (cov < 0.25)
        return true;
    return false;
}

std::vector<int> pattern_bool2int(std::vector<bool> pattern) {
    std::vector<int> pattern_int(NUM_BLOCKS, 0);
    for (int i = 0; i < NUM_BLOCKS; i++)
        pattern_int[i] = (pattern[i] ? PF_FILL_L1 : 0);
    return pattern_int;
}

bool pattern_all_set(std::vector<bool> pattern) {
    for (int i = 0; i < NUM_BLOCKS; i++)
        if (!pattern[i])
            return false;
    return true;
}

bool pattern_all_set(std::vector<int> pattern) {
    for (int i = 0; i < NUM_BLOCKS; i++)
        if (pattern[i] == 0)
            return false;
    return true;
}
} // namespace gaze_rtt

void CACHE::prefetcher_initialize() {
    std::cout << NAME << " Gaze + RTT (Region Transition Table) prefetcher" << std::endl;

    prefetchers = std::vector<gaze_rtt::Gaze>(NUM_CPUS, gaze_rtt::Gaze(gaze_rtt::FT_SIZE, gaze_rtt::FT_WAY, gaze_rtt::AT_SIZE, gaze_rtt::AT_WAY, gaze_rtt::PT_SIZE, gaze_rtt::PT_WAY, gaze_rtt::PB_SIZE, gaze_rtt::PB_WAY, cpu));
}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in) {
    uint64_t line_addr = (addr >> LOG2_BLOCK_SIZE);
    uint64_t region_num = (addr >> LOG2_PAGE_SIZE);
    int offset = line_addr % gaze_rtt::NUM_BLOCKS;

    prefetchers[cpu].set_warmup(warmup);

    if (type != LOAD)
        return metadata_in;
    uint64_t block_num = addr >> LOG2_BLOCK_SIZE;

    prefetchers[cpu].access(block_num, ip, this);
    prefetchers[cpu].prefetch(this, block_num);

    return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in) {
    uint64_t evicted_block_num = evicted_addr >> LOG2_BLOCK_SIZE;

    return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {
    prefetchers[cpu].log();
}