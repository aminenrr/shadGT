// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <fmt/format.h>
#include <imgui.h>

#include "common/perf_monitor.h"
#include "common/prt_stall_cause.h"
#include "common/types.h"
#include "imgui/compile_notice_layer.h"
#include "imgui/imgui_layer.h"
#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/compile_pack.h"

namespace ImGui::CompileNotice {

namespace {

using Clock = std::chrono::steady_clock;

// FIX-GT7-100: a first GT7 race load (every shader new) showed one frame for 185 s while the
// draws into one-off targets waited for their pipelines: players took it for a crash. The present
// thread shows the last frame again while a layer asks for it (ShouldKeepDrawing), so the notice
// moves even when the game's own loading screen can't.
constexpr auto ShowAfter = std::chrono::milliseconds{500};

bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("SHADGT_COMPILE_NOTICE");
        return !(env && env[0] == '0');
    }();
    return on;
}

class CompileNoticeUI final : public ImGui::Layer {
public:
    void Draw() override;
    bool ShouldKeepDrawing() override {
        return Stalled(Clock::now());
    }

private:
    /// True while the game has shown no new frame for ShowAfter (after its first one).
    bool Stalled(Clock::time_point now) {
        if (!Enabled()) {
            return false;
        }
        const u64 frames = Common::GetGameFrameTotals().frames;
        const u64 seen = last_frames.load(std::memory_order_relaxed);
        if (frames != seen) {
            last_frames.store(frames, std::memory_order_relaxed);
            changed_ns.store(now.time_since_epoch().count(), std::memory_order_relaxed);
            return false;
        }
        const s64 since = changed_ns.load(std::memory_order_relaxed);
        return frames != 0 && since != 0 &&
               now.time_since_epoch().count() - since >
                   std::chrono::nanoseconds{ShowAfter}.count();
    }

    std::atomic<u64> last_frames{~u64{0}};
    std::atomic<s64> changed_ns{0};

public:
    bool stall_logging = false;
    u32 stall_jobs_max = 0;
    s64 stall_from = 0;
    // PRT-STALLCAUSE: the wait totals at the last new game frame.
    u64 cause_frames = ~u64{0};
    std::array<u64, size_t(Common::StallCause::Wait::Count)> cause_us{};
    std::array<u64, size_t(Common::StallCause::Wait::Count)> cause_count{};
    std::array<u64, Common::StallCause::SiteCount> cause_site_us{};
    void TakeCauseBaseline();
    std::string CauseText();
};

CompileNoticeUI g_layer;

void CompileNoticeUI::TakeCauseBaseline() {
    namespace SC = Common::StallCause;
    for (size_t i = 0; i < cause_us.size(); ++i) {
        cause_us[i] = SC::g_wait_us[i].load(std::memory_order_relaxed);
        cause_count[i] = SC::g_wait_count[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < cause_site_us.size(); ++i) {
        cause_site_us[i] = SC::g_site_us[i].load(std::memory_order_relaxed);
    }
    std::scoped_lock lock{SC::g_fence_mutex};
    SC::g_fence_longest_us = 0;
    SC::g_fence_longest.clear();
}

std::string CompileNoticeUI::CauseText() {
    namespace SC = Common::StallCause;
    // Common::GuestClock::HoldSite order (DIAG-GTS-518 names).
    static constexpr std::array<const char*, SC::SiteCount> site_names{
        "other",   "lookup translations",  "required build",  "build wait",
        "compute", "shader on GPU thread", "translation job", "retranslate"};
    const auto delta = [&](SC::Wait wait) {
        const size_t i = size_t(wait);
        return std::pair<double, u64>{
            double(SC::g_wait_us[i].load(std::memory_order_relaxed) - cause_us[i]) / 1000.0,
            SC::g_wait_count[i].load(std::memory_order_relaxed) - cause_count[i]};
    };
    const auto [hold_ms, holds] = delta(SC::Wait::ClockHold);
    const auto [fence_ms, fences] = delta(SC::Wait::Fence);
    const auto [present_ms, presents] = delta(SC::Wait::Present);
    const auto [acquire_ms, acquires] = delta(SC::Wait::Acquire);
    double compile_ms = 0.0;
    std::string sites;
    for (size_t i = 0; i < SC::SiteCount; ++i) {
        const double ms =
            double(SC::g_site_us[i].load(std::memory_order_relaxed) - cause_site_us[i]) / 1000.0;
        if (ms < 1.0) {
            continue;
        }
        if (i != 0) {
            compile_ms += ms;
        }
        sites += fmt::format("{}{} {:.0f}", sites.empty() ? "" : ", ", site_names[i], ms);
    }
    double longest_ms = 0.0;
    std::string longest;
    {
        std::scoped_lock lock{SC::g_fence_mutex};
        longest_ms = double(SC::g_fence_longest_us) / 1000.0;
        longest = SC::g_fence_longest;
    }
    return fmt::format(
        "PRT-STALLCAUSE: guest clock held {:.0f} ms ({}), fence waits {:.0f} ms ({}, longest "
        "{:.0f} ms at {}), DIAG-GTS-518 sites ms [{}], present wait {:.0f} ms ({}), acquire wait "
        "{:.0f} ms ({}), pipeline compile waits {:.0f} ms",
        hold_ms, holds, fence_ms, fences, longest_ms, longest.empty() ? "-" : longest,
        sites.empty() ? "none" : sites, present_ms, presents, acquire_ms, acquires, compile_ms);
}

void CompileNoticeUI::Draw() {
    const auto now = Clock::now();
    // PERF-GT7-108: the compile pack's start-up compile shows its progress while it runs.
    const bool pack = Enabled() && Vulkan::CompilePack::g_running.load(std::memory_order_acquire);
    const bool stalled = Stalled(now);
    {
        // PRT-STALLLOG (pr-tests, RYGT_STALL_LOG, on; 0 = off): one line per stall when it ends,
        // its length and whether pipelines were building, to find what the menu waits on.
        static const bool log_on = [] {
            const char* env = std::getenv("RYGT_STALL_LOG");
            return !(env && env[0] == '0');
        }();
        if (log_on) {
            const bool cause_on = Common::StallCause::On();
            if (cause_on && !stalled && !stall_logging) {
                const u64 frames = Common::GetGameFrameTotals().frames;
                if (frames != cause_frames) {
                    cause_frames = frames;
                    TakeCauseBaseline();
                }
            }
            if (stalled) {
                if (!stall_logging) {
                    stall_logging = true;
                    stall_jobs_max = 0;
                    stall_from = changed_ns.load(std::memory_order_relaxed);
                }
                stall_jobs_max = std::max(stall_jobs_max,
                                          Vulkan::g_build_jobs.load(std::memory_order_relaxed));
            } else if (stall_logging) {
                stall_logging = false;
                const double ms = double(now.time_since_epoch().count() - stall_from) / 1e6;
                LOG_WARNING(Render_Vulkan,
                            "PRT-STALLLOG: no new game frame for {:.0f} ms, {} (most build jobs "
                            "{}), frame {}{}{}",
                            ms, stall_jobs_max != 0 ? "pipelines building" : "nothing compiling",
                            stall_jobs_max, Common::GetGameFrameTotals().frames,
                            cause_on ? "; " : "", cause_on ? CauseText() : std::string{});
                if (cause_on) {
                    cause_frames = Common::GetGameFrameTotals().frames;
                    TakeCauseBaseline();
                }
            }
        }
    }
    if (!stalled && !pack) {
        return;
    }
    const double stalled_s =
        std::chrono::duration<double>(now.time_since_epoch() -
                                      Clock::duration{changed_ns.load(std::memory_order_relaxed)})
            .count();
    const u32 jobs = Vulkan::g_build_jobs.load(std::memory_order_relaxed);
    char text[96];
    if (!stalled) {
        std::snprintf(text, sizeof(text), "Compiling shader pack  %u / %u",
                      Vulkan::CompilePack::g_done.load(std::memory_order_relaxed),
                      Vulkan::CompilePack::g_total.load(std::memory_order_relaxed));
    } else if (jobs != 0) {
        std::snprintf(text, sizeof(text), "Compiling shaders (%u)  %.0f s", jobs, stalled_s);
    } else {
        std::snprintf(text, sizeof(text), "Loading  %.0f s", stalled_s);
    }

    const auto& io = ImGui::GetIO();
    const float scale = std::max(1.0f, io.DisplaySize.y / 1080.0f);
    const float margin = 24.0f * scale;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - margin, io.DisplaySize.y - margin),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.6f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize;
    if (ImGui::Begin("##compile_notice", nullptr, flags)) {
        ImGui::SetWindowFontScale(scale);
        // A spinner: a quarter circle turning once a second.
        const float radius = ImGui::GetTextLineHeight() * 0.45f;
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const ImVec2 center(pos.x + radius, pos.y + ImGui::GetTextLineHeight() * 0.5f);
        const float turn = static_cast<float>(std::fmod(ImGui::GetTime(), 1.0)) * 6.2831853f;
        auto* draw_list = ImGui::GetWindowDrawList();
        draw_list->PathArcTo(center, radius, turn, turn + 4.7f, 16);
        draw_list->PathStroke(IM_COL32(255, 255, 255, 230), 0, radius * 0.35f);
        ImGui::Dummy(ImVec2(radius * 2.0f, ImGui::GetTextLineHeight()));
        ImGui::SameLine(0.0f, radius);
        ImGui::TextUnformatted(text);
    }
    ImGui::End();
}

} // namespace

void Register() {
    ImGui::Layer::AddLayer(&g_layer);
}

void Unregister() {
    ImGui::Layer::RemoveLayer(&g_layer);
}

} // namespace ImGui::CompileNotice
