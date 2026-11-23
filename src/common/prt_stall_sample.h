// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// PRT-STALLSAMPLE (pr-tests, Round 8, RYGT_STALL_SAMPLE, on; 0 = off; RYGT_STALL_SAMPLE_MS, 400):
// once a stall (no new game frame) has lasted RYGT_STALL_SAMPLE_MS, the game threads that last
// submitted GPU work and last flipped are suspended for a moment, their instruction pointer and
// the code addresses found in the top 2 KB of their stack read, then resumed. The addresses are
// named (module+offset) only after the resume, so a thread suspended inside the loader cannot
// deadlock it. Round 7 GT7 had 0.6 to 0.8 s pauses with nothing compiling, short fence waits
// and no mutex wait: this shows where the game's own threads are.

namespace Common::PrtStallSample {

/// The game thread that last submitted GPU work (sceGnmSubmit*) / last flipped.
void NoteSubmitThread();
void NoteFlipThread();

/// Called by the stall logger while a stall goes on; samples once per stall.
void MaybeSample(double stalled_ms, bool& sampled_this_stall);

} // namespace Common::PrtStallSample
