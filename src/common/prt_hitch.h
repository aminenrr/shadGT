// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// PRT-HITCHLOG (pr-tests, Round 11, RYGT_HITCH_LOG, GT7 50; 0 = off, n = ms): a game frame of n ms
// or more logs the local clock (HH:MM:SS, to match the owner's notes) and what happened during
// it: fence waits (and their top site, PRT-FENCESITE), pipeline compile waits, tessellated
// pipeline waits and builds (PRT-TESSWAIT), imported view imports (PRT-VIEWCHURN), guest unmaps,
// region creates, uploads and VRAM. Totals only grow (relaxed atomics); the frame's line takes the
// difference from the totals at the previous game frame. Round 10: hitches under the 500 ms stall
// log at the same track spots every lap were not caught.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "common/game_title.h"
#include "common/types.h"

namespace Common::PrtHitch {

/// The frame time in ms from which a frame is logged; 0 = off.
inline u32 ThresholdMs() {
    static const u32 ms = [] {
        const char* env = std::getenv("RYGT_HITCH_LOG");
        if (env && *env) {
            const unsigned long value = std::strtoul(env, nullptr, 10);
            return Common::IsGt7() ? u32(value == 1 ? 50 : value) : 0u;
        }
        return Common::IsGt7() ? 50u : 0u;
    }();
    return ms;
}

inline bool On() {
    return ThresholdMs() != 0;
}

/// The wall clock, local time, "HH:MM:SS".
inline std::string LocalClock() {
    const std::time_t seconds = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char text[16];
    std::snprintf(text, sizeof(text), "%02d:%02d:%02d", local.tm_hour, local.tm_min,
                  local.tm_sec);
    return text;
}

/// What the frame line counts (see Counters below).
enum class Count : u32 {
    UnmapUs,        ///< time in the native renderer's OnUnmap
    Unmaps,         ///< guest unmaps seen by the native renderer
    RegionCreates,  ///< regions made (MakeRegion)
    ViewImports,    ///< imported views made (PRT-VIEWCHURN)
    ViewImportUs,   ///< time importing them
    ViewDrops,      ///< imported views dropped on unmap
    TessWaits,      ///< tessellated draws that waited for their pipeline (PRT-TESSWAIT)
    TessWaitUs,     ///< time they waited
    TessSkips,      ///< tessellated draws skipped, pipeline not ready
    TessBuilds,     ///< tessellated pipelines the driver compiled (whole create)
    TessBuildUs,    ///< driver time on them (worker threads)
    UploadBytes,    ///< native renderer uploads (registry upload bytes, published each frame)
    GpuUs,          ///< HITCHLOG v2: GPU time of main lists resolved (PRT-PASSTIME timestamps)
    GuestSubmits,   ///< HITCHLOG v2: guest graphics submits (Liverpool::SubmitGfx)
    Count,
};

inline std::array<std::atomic<u64>, size_t(Count::Count)> g_counts{};

inline void Add(Count count, u64 value) {
    g_counts[size_t(count)].fetch_add(value, std::memory_order_relaxed);
}

inline void Set(Count count, u64 value) {
    g_counts[size_t(count)].store(value, std::memory_order_relaxed);
}

inline u64 Get(Count count) {
    return g_counts[size_t(count)].load(std::memory_order_relaxed);
}

inline s64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// HITCHLOG v2 (pr-tests, Round 12): the largest gap between two guest graphics submits since the
// last game frame (the presenter takes it), and the time of the last one. A long gap with little
// GPU time means the game's CPU (or a guest wait) held the frame.
inline std::atomic<s64> g_last_submit_us{};
inline std::atomic<s64> g_max_submit_gap_us{};

inline void NoteGuestSubmit() {
    if (!On()) {
        return;
    }
    const s64 now = NowUs();
    const s64 last = g_last_submit_us.exchange(now, std::memory_order_relaxed);
    Add(Count::GuestSubmits, 1);
    if (last != 0 && now > last) {
        const s64 gap = now - last;
        s64 max = g_max_submit_gap_us.load(std::memory_order_relaxed);
        while (gap > max &&
               !g_max_submit_gap_us.compare_exchange_weak(max, gap, std::memory_order_relaxed)) {
        }
    }
}

/// Adds its lifetime in microseconds to a count, when `on`.
class ScopeUs {
public:
    ScopeUs(Count count, bool on) : count{count}, from{on ? NowUs() : 0} {}
    ~ScopeUs() {
        if (from != 0) {
            const s64 us = NowUs() - from;
            Add(count, u64(us > 0 ? us : 0));
        }
    }
    ScopeUs(const ScopeUs&) = delete;
    ScopeUs& operator=(const ScopeUs&) = delete;

private:
    Count count;
    s64 from;
};

} // namespace Common::PrtHitch
