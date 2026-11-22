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
inline void NoteFence(u64 us, u64 semaphore, u64 ticks_behind) {
    Note(Wait::Fence, us);
    std::scoped_lock lock{g_fence_mutex};
    if (us <= g_fence_longest_us) {
        return;
    }
    g_fence_longest_us = us;
    g_fence_longest = FileName(t_fence_site);
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
    FenceSite(const char* site, u32 line) : site{t_fence_site}, line{t_fence_line} {
        t_fence_site = site;
        t_fence_line = line;
    }
    ~FenceSite() {
        t_fence_site = site;
        t_fence_line = line;
    }
    FenceSite(const FenceSite&) = delete;
    FenceSite& operator=(const FenceSite&) = delete;

private:
    const char* site;
    u32 line;
};

} // namespace Common::StallCause
