// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

namespace Common {

// PRT-QUIETLOG (pr-tests, RYGT_QUIET_LOGS, on; 0 = the old per-event lines come back): lines
// written for every pipeline, sampler, fault or view are left out or written once. In the
// Round 6 GT7 run they were most of the log (11,719 pipeline lines, 2,915 clamp mode lines on
// the recorder thread). Summaries, stalls and VRAM lines stay.
inline bool PrtQuietLogs() {
    static const bool quiet = [] {
        const char* env = std::getenv("RYGT_QUIET_LOGS");
        return !(env && env[0] == '0');
    }();
    return quiet;
}

// True the first time for this flag, or always when the old lines are on.
inline bool PrtLogOnce(std::atomic<bool>& logged) {
    return !PrtQuietLogs() || !logged.exchange(true, std::memory_order_relaxed);
}

} // namespace Common
