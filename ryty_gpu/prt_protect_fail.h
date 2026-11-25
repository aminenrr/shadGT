// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// PRT-PROTECTFAIL (pr-tests, Round 21, RYGT_PROTECT_FAIL_UNTRACK, GT7 on; 0 = off): page tracking
// (AddressSpace::Protect, ryty_core_shims.cpp) records the ranges where VirtualProtect failed, so
// the native renderer can treat them as not write tracked (their textures and regions are checked
// against guest memory every frame instead of waiting for a write fault that never comes).

#pragma once

#include <vector>
#include "common/types.h"

namespace Core::PrtProtectFail {

struct Record {
    VAddr address = 0;
    u64 size = 0;
    u32 protect = 0;    ///< the protection asked for (PAGE_*)
    u32 error = 0;      ///< GetLastError
    bool watch = false; ///< a write watch (no write access asked): the pages stay writable
    VAddr region_base = 0;
    VAddr allocation_base = 0;
    u64 region_size = 0;
    u32 state = 0, type = 0, current_protect = 0; ///< VirtualQuery at the failure
    u32 thread_id = 0;
};

/// Whether the failures are recorded (RYGT_PROTECT_FAIL_UNTRACK; GT7 on).
bool On();
/// Records one failure (cap 256 waiting).
void Note(const Record& record);
/// The failures recorded since the last call.
std::vector<Record> Take();

} // namespace Core::PrtProtectFail
