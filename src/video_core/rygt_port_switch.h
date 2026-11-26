// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// pr-tests Round 22: the switch of an upstream port. The fork's RYGT_ name decides first, then the
// upstream SHADGT_ name; with neither set the port is on in GT7 only (GT Sport as before).
// Read once per call site (callers keep the answer in a function-local static).

#include <cstdlib>

#include "common/game_title.h"

namespace RyGT {

inline bool PortSwitchOn(const char* rygt_name, const char* upstream_name) {
    for (const char* name : {rygt_name, upstream_name}) {
        if (!name) {
            continue;
        }
        if (const char* env = std::getenv(name); env && *env) {
            return env[0] != '0';
        }
    }
    return Common::IsGt7();
}

} // namespace RyGT
