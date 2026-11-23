// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// PRT-STALLCAUSE (pr-tests, RYGT_STALL_CAUSE, GT7, on; 0 = off): what the threads waited on while
// the game showed no new frame. Totals only grow (relaxed atomics); the PRT-STALLLOG reader keeps
// a copy at each new game frame and logs the difference when a stall ends. The longest fence wait
// and its site are reset by the reader at each new frame. Waits of several threads add up, so a
// total can be longer than the stall.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/game_title.h"
#include "common/types.h"

namespace Common::StallCause {

enum class Wait : u32 { ClockHold, Fence, Present, Acquire, Count };

/// The DIAG-GTS-518 hold sites (Common::GuestClock::HoldSite), timed on the GPU command thread.
inline constexpr size_t SiteCount = 8;

inline bool On() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_STALL_CAUSE");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

inline std::array<std::atomic<u64>, size_t(Wait::Count)> g_wait_us{};
inline std::array<std::atomic<u64>, size_t(Wait::Count)> g_wait_count{};
inline std::array<std::atomic<u64>, SiteCount> g_site_us{};
inline std::atomic<s64> g_hold_start_ns{0};

inline std::mutex g_fence_mutex;
inline u64 g_fence_longest_us = 0;  ///< guarded by g_fence_mutex
inline std::string g_fence_longest; ///< guarded by g_fence_mutex

/// The caller of a fence wait (Scheduler::Wait's call site, or a named wait), read by
/// Semaphore::Wait on the same thread.
inline thread_local const char* t_fence_site = nullptr;
inline thread_local u32 t_fence_line = 0;
/// PRT-FENCESITE: what the wait is for, when the site sets it (a priority op's caller).
inline thread_local const char* t_fence_kind = nullptr;
inline thread_local const char* t_fence_func = nullptr; ///< PRT-FENCESITE: the caller's function

/// PRT-FENCESITE: "Class::Function" out of a compiler's full function signature.
inline std::string ShortFunction(const char* signature) {
    if (signature == nullptr || *signature == 0) {
        return {};
    }
    std::string text{signature};
    if (const auto paren = text.find('('); paren != std::string::npos) {
        text.resize(paren);
    }
    if (const auto space = text.find_last_of(' '); space != std::string::npos) {
        text.erase(0, space + 1);
    }
    // Keep the last two names (Class::Function).
    size_t cut = std::string::npos;
    if (const auto last = text.rfind("::"); last != std::string::npos && last > 0) {
        cut = text.rfind("::", last - 1);
    }
    if (cut != std::string::npos) {
        text.erase(0, cut + 2);
    }
    if (text.size() > 48) {
        text.resize(48);
    }
    return text;
}

// PRT-FENCESITE (pr-tests, Round 11, RYGT_FENCE_SITE_LOG, GT7 on; 0 = off): every CPU wait on the
// GPU timeline that blocked is counted by its call site (file:line of the caller, with a name
// for what it waits for: readback, guest unmap, region flush, ...). The PRT-STALLLOG line, the
// PRT-HITCHLOG line and a line every 10 s (with PRT-PERF) give the top sites by count and ms.
// Round 10: stalls of 600 to 650 ms were 110 to 120 small fence waits at "direct", with nothing
// compiling: the site was not known. Log only; the table is touched on blocking waits only.
inline bool FenceSiteOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_FENCE_SITE_LOG");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

/// PRT-FENCESITE: the counting slots, each reset by its own reader.
enum class SiteSlot : u32 { Stall, Window, Hitch, Count };

struct SiteStat {
    const char* file = nullptr;
    u32 line = 0;
    const char* kind = nullptr;
    const char* func = nullptr;
    u64 total_count = 0;
    u64 total_us = 0;
    std::array<u64, size_t(SiteSlot::Count)> count{};
    std::array<u64, size_t(SiteSlot::Count)> us{};
    std::array<u64, size_t(SiteSlot::Count)> max_us{};
};
inline constexpr size_t MaxFenceSites = 64;
inline std::array<SiteStat, MaxFenceSites> g_fence_sites{}; ///< guarded by g_fence_mutex
inline size_t g_fence_sites_used = 0;                       ///< guarded by g_fence_mutex

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Note(Wait wait, u64 us) {
    g_wait_us[size_t(wait)].fetch_add(us, std::memory_order_relaxed);
    g_wait_count[size_t(wait)].fetch_add(1, std::memory_order_relaxed);
}

inline void NoteSite(size_t site, u64 us) {
    if (site < SiteCount) {
        g_site_us[site].fetch_add(us, std::memory_order_relaxed);
    }
}

inline std::string FileName(const char* path) {
    if (path == nullptr) {
        return "direct";
    }
    const char* name = path;
    for (const char* c = path; *c; ++c) {
        if (*c == '/' || *c == '\\') {
            name = c + 1;
        }
    }
    return name;
}

/// A fence wait that blocked (the slow path only), its semaphore and how far behind the GPU was.
/// PRT-FENCESITE: a short name for what a wait from this file is for; `after_gpu` for the
/// priority op runner, which waits for the GPU before running an op queued from that file.
inline const char* FenceKindOf(const char* path, bool after_gpu = false) {
    struct Kind {
        const char* file;
        const char* kind;
        const char* after;
    };
    static constexpr Kind kinds[] = {
        {"buffer.cpp", "buffer readback watch", "after GPU: buffer readback"},
        {"fault_manager.cpp", "fault / region flush", "after GPU: fault / region flush"},
        {"vk_staging_buffer_pool.cpp", "staging buffer reuse", "after GPU: staging buffer"},
        {"vk_presenter.cpp", "present / flip", "after GPU: present / screenshot"},
        {"vk_rasterizer.cpp", "readback", "after GPU: readback to guest"},
        {"vk_rasterizer_native.cpp", "native renderer", "after GPU: native fill receipts"},
        {"buffer_cache.cpp", "buffer cache (fault arena, download)",
         "after GPU: buffer cache fault / download"},
        {"direct_memory.cpp", "guest unmap / invalidate (imported views)",
         "after GPU: guest unmap (imported view release)"},
        {"texture_cache.cpp", "texture download", "after GPU: texture download"},
        {"vk_resource_pool.cpp", "descriptor pool", "after GPU: descriptor pool"},
        {"liverpool.cpp", "command processor", "after GPU: command processor"},
        {"vk_pipeline_cache.cpp", "pipeline cache", "after GPU: pipeline cache"},
        {"proto_import.cpp", "imported memory", "after GPU: imported memory"},
        {"vk_scheduler.cpp", "scheduler", "after GPU: scheduler"},
    };
    if (path == nullptr) {
        return after_gpu ? "after GPU: unnamed" : "unnamed";
    }
    const std::string name = FileName(path);
    for (const auto& k : kinds) {
        if (name == k.file) {
            return after_gpu ? k.after : k.kind;
        }
    }
    return after_gpu ? "after GPU: other" : "other";
}

/// PRT-FENCESITE: with g_fence_mutex held.
inline void NoteFenceSiteLocked(u64 us) {
    const char* file = t_fence_site;
    const u32 line = t_fence_line;
    const char* kind = t_fence_kind ? t_fence_kind : FenceKindOf(file);
    const char* func = t_fence_func;
    SiteStat* stat = nullptr;
    for (size_t i = 0; i < g_fence_sites_used; ++i) {
        SiteStat& s = g_fence_sites[i];
        if (s.file == file && s.line == line && s.kind == kind) {
            stat = &s;
            break;
        }
    }
    if (stat == nullptr) {
        // Full: the last slot takes the rest.
        const size_t at = std::min(g_fence_sites_used, MaxFenceSites - 1);
        if (g_fence_sites_used < MaxFenceSites) {
            ++g_fence_sites_used;
            g_fence_sites[at].file = file;
            g_fence_sites[at].line = line;
            g_fence_sites[at].kind = kind;
            g_fence_sites[at].func = func;
        }
        stat = &g_fence_sites[at];
    }
    ++stat->total_count;
    stat->total_us += us;
    for (size_t slot = 0; slot < size_t(SiteSlot::Count); ++slot) {
        ++stat->count[slot];
        stat->us[slot] += us;
        stat->max_us[slot] = std::max(stat->max_us[slot], us);
    }
}

/// PRT-FENCESITE: the top sites of a slot by ms ("kind file:line Nx a ms (max b)"), then the
/// slot is reset when `reset`.
inline std::string FenceSiteText(SiteSlot slot, size_t top, bool reset) {
    std::vector<std::pair<u64, size_t>> order;
    std::string text;
    std::scoped_lock lock{g_fence_mutex};
    const size_t s = size_t(slot);
    for (size_t i = 0; i < g_fence_sites_used; ++i) {
        if (g_fence_sites[i].count[s] != 0) {
            order.emplace_back(g_fence_sites[i].us[s], i);
        }
    }
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    for (size_t k = 0; k < order.size() && k < top; ++k) {
        const SiteStat& stat = g_fence_sites[order[k].second];
        char item[320];
        const std::string func = ShortFunction(stat.func);
        std::snprintf(item, sizeof(item), "%s%s %s:%u%s%s %llux %.1f ms (max %.1f)",
                      text.empty() ? "" : ", ", stat.kind ? stat.kind : "-",
                      FileName(stat.file).c_str(), stat.line, func.empty() ? "" : " ",
                      func.c_str(),
                      static_cast<unsigned long long>(stat.count[s]), double(stat.us[s]) / 1000.0,
                      double(stat.max_us[s]) / 1000.0);
        text += item;
    }
    if (order.size() > top) {
        text += ", +" + std::to_string(order.size() - top) + " more sites";
    }
    if (reset) {
        for (size_t i = 0; i < g_fence_sites_used; ++i) {
            g_fence_sites[i].count[s] = 0;
            g_fence_sites[i].us[s] = 0;
            g_fence_sites[i].max_us[s] = 0;
        }
    }
    return text;
}

/// PRT-FENCESITE: the slot's totals (count, us).
inline std::pair<u64, u64> FenceSiteTotals(SiteSlot slot) {
    std::scoped_lock lock{g_fence_mutex};
    u64 count = 0, us = 0;
    for (size_t i = 0; i < g_fence_sites_used; ++i) {
        count += g_fence_sites[i].count[size_t(slot)];
        us += g_fence_sites[i].us[size_t(slot)];
    }
    return {count, us};
}

inline void NoteFence(u64 us, u64 semaphore, u64 ticks_behind) {
    Note(Wait::Fence, us);
    std::scoped_lock lock{g_fence_mutex};
    if (FenceSiteOn()) {
        NoteFenceSiteLocked(us);
    }
    if (us <= g_fence_longest_us) {
        return;
    }
    g_fence_longest_us = us;
    g_fence_longest = FileName(t_fence_site);
    if (FenceSiteOn()) { // PRT-FENCESITE: what it waited for
        g_fence_longest = std::string{t_fence_kind ? t_fence_kind : FenceKindOf(t_fence_site)} +
                          " " + g_fence_longest;
    }
    if (t_fence_line != 0) {
        g_fence_longest += ':' + std::to_string(t_fence_line);
    }
    char text[64];
    std::snprintf(text, sizeof(text), " timeline 0x%llx, %llu ticks behind",
                  static_cast<unsigned long long>(semaphore),
                  static_cast<unsigned long long>(ticks_behind));
    g_fence_longest += text;
}

/// Times a wait of one kind for its lifetime.
class Timer {
public:
    explicit Timer(Wait wait) : wait{wait}, from{On() ? NowNs() : 0} {}
    ~Timer() {
        if (from != 0) {
            Note(wait, u64(std::max<s64>(NowNs() - from, 0) / 1000));
        }
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    Wait wait;
    s64 from;
};

/// Sets the fence wait site for this thread while it lives.
class FenceSite {
public:
    FenceSite(const char* site, u32 line, const char* kind = nullptr, const char* func = nullptr)
        : site{t_fence_site}, line{t_fence_line}, kind{t_fence_kind}, func{t_fence_func} {
        t_fence_site = site;
        t_fence_line = line;
        t_fence_func = func;
        if (kind != nullptr) {
            t_fence_kind = kind; // PRT-FENCESITE (an outer site's name stays otherwise)
        }
    }
    ~FenceSite() {
        t_fence_site = site;
        t_fence_line = line;
        t_fence_kind = kind;
        t_fence_func = func;
    }
    FenceSite(const FenceSite&) = delete;
    FenceSite& operator=(const FenceSite&) = delete;

private:
    const char* site;
    u32 line;
    const char* kind;
    const char* func;
};

} // namespace Common::StallCause
