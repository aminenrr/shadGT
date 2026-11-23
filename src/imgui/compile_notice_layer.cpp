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
#include "common/prt_hitch.h"
#include "common/prt_stall_cause.h"
#include "common/prt_stall_sample.h"
#include "common/sampling_profiler.h"
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
    bool stall_sampled = false; // PRT-STALLSAMPLE (Round 8)
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
    {
        std::scoped_lock lock{SC::g_fence_mutex};
        SC::g_fence_longest_us = 0;
        SC::g_fence_longest.clear();
    }
    if (SC::FenceSiteOn()) {
        SC::FenceSiteText(SC::SiteSlot::Stall, 0, true); // PRT-FENCESITE: from this frame on
    }
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
    // PRT-FENCESITE (Round 11): the fence waits of the stall by call site.
    std::string fence_sites;
    if (SC::FenceSiteOn()) {
        const std::string top = SC::FenceSiteText(SC::SiteSlot::Stall, 6, false);
        fence_sites = fmt::format("; PRT-FENCESITE: {}", top.empty() ? "no fence waits" : top);
    }
    return fmt::format(
        "PRT-STALLCAUSE: guest clock held {:.0f} ms ({}), fence waits {:.0f} ms ({}, longest "
        "{:.0f} ms at {}), DIAG-GTS-518 sites ms [{}], present wait {:.0f} ms ({}), acquire wait "
        "{:.0f} ms ({}), pipeline compile waits {:.0f} ms{}",
        hold_ms, holds, fence_ms, fences, longest_ms, longest.empty() ? "-" : longest,
        sites.empty() ? "none" : sites, present_ms, presents, acquire_ms, acquires, compile_ms,
        fence_sites);
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
                // PRT-STALLSAMPLE (Round 8): where the game's threads are, once per stall.
                Common::PrtStallSample::MaybeSample(
                    double(now.time_since_epoch().count() - stall_from) / 1e6, stall_sampled);
            } else if (stall_logging) {
                stall_logging = false;
                stall_sampled = false;
                const double ms = double(now.time_since_epoch().count() - stall_from) / 1e6;
                LOG_WARNING(Render_Vulkan,
                            "PRT-STALLLOG [{}]: no new game frame for {:.0f} ms, {} (most build "
                            "jobs {}), frame {}{}{}",
                            Common::PrtHitch::LocalClock(), // PRT-HITCHLOG (Round 11)
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

// PRT-STALLSAMPLE (Round 8): see common/prt_stall_sample.h.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dbghelp.h>
#include <mutex>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace Common::PrtStallSample {

static std::atomic<std::uint32_t> g_submit_tid{0};
static std::atomic<std::uint32_t> g_flip_tid{0};

static bool On() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_STALL_SAMPLE");
        return !(env && env[0] == '0');
    }();
    return on;
}

static std::uint32_t AfterMs() {
    static const std::uint32_t ms = [] {
        const char* env = std::getenv("RYGT_STALL_SAMPLE_MS");
        const long value = env && *env ? std::strtol(env, nullptr, 10) : 400;
        return value < 50 ? std::uint32_t{50} : static_cast<std::uint32_t>(value);
    }();
    return ms;
}

void NoteSubmitThread() {
#ifdef _WIN32
    g_submit_tid.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
}

void NoteFlipThread() {
#ifdef _WIN32
    g_flip_tid.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
}

#ifdef _WIN32
struct Sample {
    std::uint32_t tid = 0;
    bool ok = false;
    std::uint64_t rip = 0;
    std::uint32_t count = 0;
    std::uint64_t code[16]{};
};

// Is the page executable? VirtualQuery takes no loader lock.
static bool IsCode(std::uint64_t value) {
    if (value < 0x10000) {
        return false;
    }
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(value), &info, sizeof(info)) != sizeof(info)) {
        return false;
    }
    if (info.State != MEM_COMMIT) {
        return false;
    }
    const DWORD protect = info.Protect & 0xff;
    return protect == PAGE_EXECUTE || protect == PAGE_EXECUTE_READ ||
           protect == PAGE_EXECUTE_READWRITE || protect == PAGE_EXECUTE_WRITECOPY;
}

static Sample TakeSample(std::uint32_t tid) {
    Sample sample;
    sample.tid = tid;
    if (tid == 0 || tid == GetCurrentThreadId()) {
        return sample;
    }
    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, tid);
    if (thread == nullptr) {
        return sample;
    }
    if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        if (GetThreadContext(thread, &context)) {
            sample.ok = true;
            sample.rip = context.Rip;
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &info, sizeof(info)) ==
                    sizeof(info) &&
                info.State == MEM_COMMIT && (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
                const std::uint64_t region_end =
                    reinterpret_cast<std::uint64_t>(info.BaseAddress) + info.RegionSize;
                std::uint64_t end = context.Rsp + 2048;
                if (end > region_end) {
                    end = region_end;
                }
                for (std::uint64_t at = context.Rsp & ~7ull; at + 8 <= end && sample.count < 16;
                     at += 8) {
                    const std::uint64_t value = *reinterpret_cast<const std::uint64_t*>(at);
                    if (IsCode(value)) {
                        sample.code[sample.count++] = value;
                    }
                }
            }
        }
        ResumeThread(thread);
    }
    CloseHandle(thread);
    return sample;
}

// PRT-STALLSYM (pr-tests, Round 9, RYGT_STALL_SAMPLE_SYMBOLS, on; 0 = off): the sampled IP and
// the first stack addresses also get a function name from libSceVideoOut.pdb, which the pr-tests
// package now ships next to libSceVideoOut.prx. R8: 9 of 13 stalls sat in libSceVideoOut.prx+0xd8e6xx.
static bool StallSymbolsOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_STALL_SAMPLE_SYMBOLS");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return on;
}

static std::string SymbolName(std::uint64_t address) {
    static std::once_flag once;
    static bool ready = false;
    static std::mutex lock;
    std::call_once(once, [] {
        wchar_t path[MAX_PATH] = {};
        HMODULE self = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&StallSymbolsOn), &self)) {
            const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
            for (DWORD i = n; i > 0; --i) {
                if (path[i - 1] == L'\\' || path[i - 1] == L'/') {
                    path[i - 1] = L'\0';
                    break;
                }
            }
        }
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
        ready = SymInitializeW(GetCurrentProcess(), path[0] ? path : nullptr, TRUE) != FALSE;
        LOG_WARNING(Render_Vulkan, "PRT-STALLSYM: symbol lookup {}", ready ? "ready" : "unavailable");
    });
    if (!ready) {
        return {};
    }
    std::lock_guard guard(lock);
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256] = {};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 255;
    DWORD64 displacement = 0;
    if (!SymFromAddr(GetCurrentProcess(), address, &displacement, symbol)) {
        return {};
    }
    return fmt::format(" [{}+{:#x}]", symbol->Name, displacement);
}

static std::string Describe(const Sample& sample, const char* role) {
    if (!sample.ok) {
        return fmt::format("{} tid {}: not sampled", role, sample.tid);
    }
    const bool symbols = StallSymbolsOn();
    std::string out = fmt::format("{} tid {}: at {}{}; stack", role, sample.tid,
                                  Common::SamplingProfiler::DescribeCode(sample.rip),
                                  symbols ? SymbolName(sample.rip) : std::string{});
    for (std::uint32_t i = 0; i < sample.count; ++i) {
        out += ' ';
        out += Common::SamplingProfiler::DescribeCode(sample.code[i]);
        if (symbols && i < 6) {
            out += SymbolName(sample.code[i]);
        }
    }
    return out;
}
#endif

// Called by the stall logger while a stall goes on; samples once per stall.
void MaybeSample(double stalled_ms, bool& sampled_this_stall) {
#ifdef _WIN32
    if (sampled_this_stall || !On() || stalled_ms < static_cast<double>(AfterMs())) {
        return;
    }
    sampled_this_stall = true;
    static std::atomic<std::uint32_t> logged{0};
    const std::uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 60) {
        return;
    }
    const std::uint32_t submit_tid = g_submit_tid.load(std::memory_order_relaxed);
    const std::uint32_t flip_tid = g_flip_tid.load(std::memory_order_relaxed);
    const Sample submit = TakeSample(submit_tid);
    const Sample flip = flip_tid != submit_tid ? TakeSample(flip_tid) : Sample{};
    LOG_WARNING(Render_Vulkan, "PRT-STALLSAMPLE: stalled {:.0f} ms ({} so far): {}{}{}",
                stalled_ms, n + 1, Describe(submit, "GPU submit thread"),
                flip_tid != submit_tid ? "; " : "",
                flip_tid != submit_tid ? Describe(flip, "flip thread") : std::string{});
#else
    (void)stalled_ms;
    (void)sampled_this_stall;
#endif
}

} // namespace Common::PrtStallSample
