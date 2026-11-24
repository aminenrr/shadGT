// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>

#include "common/game_title.h"
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
#include <psapi.h>
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
    std::uint64_t regs[16]{}; // PRT-AUDIOLOOP (Round 14): rax..r15 in CONTEXT order
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
            const std::uint64_t regs[16] = {context.Rax, context.Rcx, context.Rdx, context.Rbx,
                                            context.Rsp, context.Rbp, context.Rsi, context.Rdi,
                                            context.R8,  context.R9,  context.R10, context.R11,
                                            context.R12, context.R13, context.R14, context.R15};
            std::copy(std::begin(regs), std::end(regs), std::begin(sample.regs));
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

// PRT-AUDIOLOAD (pr-tests, Round 13): see common/prt_stall_sample.h.
bool AudioLoadOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_AUDIO_LOAD_LOG");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

static std::atomic<std::uint32_t> g_audio_tid{0};

void AudioLoadArm(std::uint32_t tid) {
    g_audio_tid.store(tid, std::memory_order_relaxed);
}

static std::atomic<std::uint32_t> g_audio_mxcsr_tid{0};

#ifdef _WIN32
static void AudioFaultArm(std::uint32_t tid);
#endif

void AudioMxcsrArm(std::uint32_t tid) {
    g_audio_mxcsr_tid.store(tid, std::memory_order_relaxed);
#ifdef _WIN32
    AudioFaultArm(tid); // PRT-AUDIOFAULT (Round 16b)
#endif
}

#ifdef _WIN32
struct AudioSample {
    std::uint64_t rip;
    std::uint64_t caller; // the first code address on its stack (0 = none)
    std::uint64_t regs[16];
};

// PRT-AUDIOLOOP (pr-tests, Round 14, RYGT_AUDIO_LOOP_LOG, GT7 on with PRT-AUDIOLOAD; 0 = off; log
// only): R13 put 46-61% of the hot audio thread's samples in one 16-byte loop (GT7.exe+0x2b68360,
// stack +0x2b5a2a5) while its load climbed from ~20% to 95% of a core over a race. For the
// window's hottest spot this gives each register's low, middle and high value over the samples
// in it (a loop count or list length that grows shows up as a rising middle from one line to the
// next), and once per run the 128 code bytes around it, to read the loop offline.
static bool AudioLoopOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_AUDIO_LOOP_LOG");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

static std::string AudioLoopText(const std::vector<AudioSample>& samples, std::uint64_t bucket) {
    static constexpr const char* names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
                                              "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                                              "r12", "r13", "r14", "r15"};
    std::vector<std::uint64_t> values;
    std::string text;
    for (int reg = 0; reg < 16; ++reg) {
        if (reg == 4) {
            continue; // rsp
        }
        values.clear();
        for (const AudioSample& sample : samples) {
            if ((sample.rip & ~std::uint64_t{63}) == bucket) {
                values.push_back(sample.regs[reg]);
            }
        }
        if (values.empty()) {
            return {};
        }
        std::sort(values.begin(), values.end());
        const std::uint64_t low = values.front();
        const std::uint64_t mid = values[values.size() / 2];
        const std::uint64_t high = values.back();
        if (high <= 0xffffffffull) {
            text += fmt::format("{}{} {}/{}/{}", text.empty() ? "" : ", ", names[reg], low, mid,
                                high);
        } else {
            text += fmt::format("{}{} {:#x}/+{:#x}/+{:#x}", text.empty() ? "" : ", ", names[reg],
                                low, mid - low, high - low);
        }
    }
    static bool dumped = false;
    if (!dumped) {
        dumped = true;
        const std::uint64_t start = bucket >= 64 ? bucket - 64 : bucket;
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(start), &info, sizeof(info)) ==
                sizeof(info) &&
            info.State == MEM_COMMIT && (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0 &&
            reinterpret_cast<std::uint64_t>(info.BaseAddress) + info.RegionSize >= start + 192) {
            std::string bytes;
            const auto* code = reinterpret_cast<const std::uint8_t*>(start);
            for (int i = 0; i < 192; ++i) {
                bytes += fmt::format("{:02x}", code[i]);
            }
            LOG_WARNING(Render_Vulkan, "PRT-AUDIOLOOP: code bytes from {} (192): {}",
                        Common::SamplingProfiler::DescribeCode(start), bytes);
        }
    }
    return text;
}
static constexpr std::size_t AudioSampleMax = 1200; // 10 s at one per 10 ms, and some slack
static std::vector<AudioSample> g_audio_samples;     // only the PRT-PERF thread touches these
static std::uint32_t g_audio_failed = 0;

// PRT-AUDIOMXCSR (pr-tests, Round 16, RYGT_AUDIO_MXCSR_LOG, GT7 on with PRT-AUDIOLOAD; 0 = off;
// log only) and PRT-AUDIOFTZ (RYGT_AUDIO_FTZ, GT7 on since Round 17; 0 = off): R15 took the red-zone jumps out of
// the hot 128-sample mix loop (GT7.exe+0x2b68340) and the audio thread still ran 80-98% of a
// core. Orbis threads run with MXCSR 0x9fc0 (flush-to-zero + denormals-are-zero); without them,
// fading voices go denormal and float math slows 10-100x. Once a second this reads the audio
// thread's MXCSR; the PRT-AUDIOLOAD line gives the values seen and, with RYGT_AUDIO_FTZ=1, how
// often FTZ+DAZ had to be set back on. R16 test: the audio thread starts with 0x1fa0 (FTZ and DAZ
// off), runs 80-92% of a core with 1429 underruns; with FTZ+DAZ set it runs 15-18% with 0.
static bool AudioMxcsrLogOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_AUDIO_MXCSR_LOG");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

static bool AudioFtzOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_AUDIO_FTZ");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

static constexpr std::uint32_t MxcsrFtzDaz = 0x8040;
static std::vector<std::pair<std::uint32_t, std::uint32_t>> g_audio_mxcsr; // value, times seen
static std::uint32_t g_audio_ftz_set = 0;
static std::uint32_t g_audio_ftz_failed = 0;

static void AudioMxcsrPoll(std::uint32_t tid) {
    static auto last = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds{1}) {
        return;
    }
    last = now;
    const bool force = AudioFtzOn();
    if (tid == 0 || tid == GetCurrentThreadId()) {
        return;
    }
    const DWORD access = THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | (force ? THREAD_SET_CONTEXT : 0);
    HANDLE thread = OpenThread(access, FALSE, tid);
    if (thread == nullptr) {
        return;
    }
    if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
        if (GetThreadContext(thread, &context)) {
            const std::uint32_t value = context.MxCsr;
            bool found = false;
            for (auto& [key, n] : g_audio_mxcsr) {
                if (key == value) {
                    ++n;
                    found = true;
                    break;
                }
            }
            if (!found && g_audio_mxcsr.size() < 8) {
                g_audio_mxcsr.emplace_back(value, 1);
            }
            if (force && (value & MxcsrFtzDaz) != MxcsrFtzDaz) {
                CONTEXT fixed{};
                fixed.ContextFlags = CONTEXT_FLOATING_POINT;
                if (GetThreadContext(thread, &fixed)) {
                    fixed.MxCsr |= MxcsrFtzDaz;
                    fixed.FltSave.MxCsr |= MxcsrFtzDaz;
                    if (SetThreadContext(thread, &fixed)) {
                        ++g_audio_ftz_set;
                    } else {
                        ++g_audio_ftz_failed;
                    }
                } else {
                    ++g_audio_ftz_failed;
                }
            }
        }
        ResumeThread(thread);
    }
    CloseHandle(thread);
}

// PRT-AUDIOFAULT (pr-tests, Round 16b, RYGT_AUDIO_FAULT_LOG, GT7 on with PRT-AUDIOLOAD; 0 = off;
// log only): R15 showed GT7's audio thread at 15-21% of a core in kernel time, also in windows
// nobody sampled it, while its hot spot is a plain float mix loop. A first-in-line vectored
// handler counts the exceptions raised on that thread (write-watch and guard faults, single
// steps, HLE traps) and keeps the last 256 (code, address, rip); it never handles one. The
// PRT-AUDIOLOAD line adds the count by code, the top 64 KB fault areas and code spots, and the
// process page faults (soft + hard) of the 10 s window.
static bool AudioFaultOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_AUDIO_FAULT_LOG");
        return !(env && env[0] == '0') && Common::IsGt7();
    }();
    return on;
}

struct AudioFault {
    std::atomic<std::uint32_t> code{0};
    std::atomic<std::uint64_t> address{0};
    std::atomic<std::uint64_t> rip{0};
};
static constexpr std::uint32_t AudioFaultRing = 256;
static AudioFault g_audio_faults[AudioFaultRing];
static std::atomic<std::uint32_t> g_audio_fault_tid{0};
static std::atomic<std::uint64_t> g_audio_fault_count{0};
static std::uint64_t g_audio_fault_seen = 0; // PRT-PERF thread only

static LONG CALLBACK AudioFaultCount(EXCEPTION_POINTERS* info) {
    const std::uint32_t tid = g_audio_fault_tid.load(std::memory_order_relaxed);
    if (tid == 0 || tid != GetCurrentThreadId() || info == nullptr ||
        info->ExceptionRecord == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const EXCEPTION_RECORD* record = info->ExceptionRecord;
    std::uint64_t address = reinterpret_cast<std::uint64_t>(record->ExceptionAddress);
    if ((record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
         record->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        record->NumberParameters >= 2) {
        address = static_cast<std::uint64_t>(record->ExceptionInformation[1]);
    }
    const std::uint64_t n = g_audio_fault_count.fetch_add(1, std::memory_order_relaxed);
    AudioFault& slot = g_audio_faults[n % AudioFaultRing];
    slot.code.store(static_cast<std::uint32_t>(record->ExceptionCode), std::memory_order_relaxed);
    slot.address.store(address, std::memory_order_relaxed);
    slot.rip.store(info->ContextRecord != nullptr ? info->ContextRecord->Rip : 0,
                   std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_SEARCH;
}

static void AudioFaultArm(std::uint32_t tid) {
    if (!AudioFaultOn()) {
        return;
    }
    static bool installed = false;
    if (tid != 0 && !installed) {
        installed = AddVectoredExceptionHandler(1, AudioFaultCount) != nullptr;
        LOG_WARNING(Render_Vulkan,
                    "PRT-AUDIOFAULT: {}, exceptions on the audio thread are counted "
                    "(RYGT_AUDIO_FAULT_LOG=0 turns it off)",
                    installed ? "on" : "handler not added");
    }
    g_audio_fault_tid.store(tid, std::memory_order_relaxed);
}

static std::string AudioFaultText() {
    if (!AudioFaultOn()) {
        return {};
    }
    std::string text;
    PROCESS_MEMORY_COUNTERS memory{};
    memory.cb = sizeof(memory);
    static DWORD last_page_faults = 0;
    static bool have_page_faults = false;
    using MemoryInfoFn = BOOL(WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    static const auto memory_info = reinterpret_cast<MemoryInfoFn>(
        GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo"));
    if (memory_info != nullptr && memory_info(GetCurrentProcess(), &memory, sizeof(memory))) {
        if (have_page_faults) {
            text += fmt::format("; process page faults {}", memory.PageFaultCount - last_page_faults);
        }
        last_page_faults = memory.PageFaultCount;
        have_page_faults = true;
    }
    const std::uint64_t total = g_audio_fault_count.load(std::memory_order_relaxed);
    const std::uint64_t fresh = total - g_audio_fault_seen;
    g_audio_fault_seen = total;
    if (g_audio_fault_tid.load(std::memory_order_relaxed) == 0) {
        return text;
    }
    text += fmt::format("; PRT-AUDIOFAULT {} exceptions on it", fresh);
    if (fresh == 0) {
        return text;
    }
    const std::uint64_t keep = std::min<std::uint64_t>(fresh, AudioFaultRing);
    std::vector<std::pair<std::uint64_t, std::uint32_t>> codes;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> areas;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> rips;
    const auto bump = [](std::vector<std::pair<std::uint64_t, std::uint32_t>>& list,
                         std::uint64_t value) {
        for (auto& [key, n] : list) {
            if (key == value) {
                ++n;
                return;
            }
        }
        list.emplace_back(value, 1);
    };
    for (std::uint64_t i = total - keep; i < total; ++i) {
        const AudioFault& slot = g_audio_faults[i % AudioFaultRing];
        bump(codes, slot.code.load(std::memory_order_relaxed));
        bump(areas, slot.address.load(std::memory_order_relaxed) & ~std::uint64_t{0xffff});
        bump(rips, slot.rip.load(std::memory_order_relaxed));
    }
    const auto by_count = [](const auto& a, const auto& b) { return a.second > b.second; };
    std::sort(codes.begin(), codes.end(), by_count);
    std::sort(areas.begin(), areas.end(), by_count);
    std::sort(rips.begin(), rips.end(), by_count);
    text += fmt::format(" (last {}: codes", keep);
    for (std::size_t i = 0; i < codes.size() && i < 4; ++i) {
        text += fmt::format(" {:#x} x{}", codes[i].first, codes[i].second);
    }
    text += "; areas";
    for (std::size_t i = 0; i < areas.size() && i < 3; ++i) {
        text += fmt::format(" {:#x} x{}", areas[i].first, areas[i].second);
    }
    text += "; at";
    for (std::size_t i = 0; i < rips.size() && i < 3; ++i) {
        text += fmt::format(" {} x{}", Common::SamplingProfiler::DescribeCode(rips[i].first),
                            rips[i].second);
    }
    text += ")";
    return text;
}

static std::string AudioMxcsrText() {
    const std::string faults = AudioFaultText(); // PRT-AUDIOFAULT (Round 16b)
    if (g_audio_mxcsr.empty() && g_audio_ftz_set == 0 && g_audio_ftz_failed == 0) {
        return faults;
    }
    std::string text = faults + "; MXCSR";
    for (const auto& [value, n] : g_audio_mxcsr) {
        text += fmt::format(" {:#x} x{} (FTZ {}, DAZ {})", value, n, (value & 0x8000) ? "on" : "off",
                            (value & 0x40) ? "on" : "off");
    }
    if (AudioFtzOn()) {
        text += fmt::format(", PRT-AUDIOFTZ set FTZ+DAZ {} times", g_audio_ftz_set);
        if (g_audio_ftz_failed != 0) {
            text += fmt::format(" ({} failed)", g_audio_ftz_failed);
        }
    }
    g_audio_mxcsr.clear();
    g_audio_ftz_set = 0;
    g_audio_ftz_failed = 0;
    return text;
}

void AudioLoadPoll() {
    const std::uint32_t tid = g_audio_tid.load(std::memory_order_relaxed);
    if (AudioMxcsrLogOn() || AudioFtzOn()) {
        AudioMxcsrPoll(g_audio_mxcsr_tid.load(std::memory_order_relaxed));
    }
    if (tid == 0 || g_audio_samples.size() >= AudioSampleMax) {
        return;
    }
    static auto last = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds{10}) {
        return;
    }
    last = now;
    const Sample sample = TakeSample(tid);
    if (!sample.ok) {
        ++g_audio_failed;
        return;
    }
    if (g_audio_samples.capacity() == 0) {
        g_audio_samples.reserve(AudioSampleMax);
    }
    AudioSample entry{sample.rip, sample.count != 0 ? sample.code[0] : 0, {}};
    std::copy(std::begin(sample.regs), std::end(sample.regs), std::begin(entry.regs));
    g_audio_samples.push_back(entry);
}

std::string AudioLoadReport(std::size_t top, std::uint32_t& samples) {
    samples = static_cast<std::uint32_t>(g_audio_samples.size());
    if (g_audio_samples.empty()) {
        g_audio_failed = 0;
        return AudioMxcsrText();
    }
    // Spots are 64-byte code ranges; each names its most seen address and most seen caller.
    struct Spot {
        std::uint64_t bucket = 0;
        std::uint32_t count = 0;
        std::vector<std::pair<std::uint64_t, std::uint32_t>> rips;
        std::vector<std::pair<std::uint64_t, std::uint32_t>> callers;
    };
    const auto bump = [](std::vector<std::pair<std::uint64_t, std::uint32_t>>& list,
                         std::uint64_t value) {
        for (auto& [key, n] : list) {
            if (key == value) {
                ++n;
                return;
            }
        }
        list.emplace_back(value, 1);
    };
    const auto most = [](const std::vector<std::pair<std::uint64_t, std::uint32_t>>& list) {
        std::pair<std::uint64_t, std::uint32_t> best{0, 0};
        for (const auto& item : list) {
            if (item.second > best.second) {
                best = item;
            }
        }
        return best;
    };
    std::vector<Spot> spots;
    for (const AudioSample& sample : g_audio_samples) {
        const std::uint64_t bucket = sample.rip & ~std::uint64_t{63};
        Spot* spot = nullptr;
        for (Spot& s : spots) {
            if (s.bucket == bucket) {
                spot = &s;
                break;
            }
        }
        if (spot == nullptr) {
            spots.push_back(Spot{bucket});
            spot = &spots.back();
        }
        ++spot->count;
        bump(spot->rips, sample.rip);
        bump(spot->callers, sample.caller);
    }
    std::sort(spots.begin(), spots.end(),
              [](const Spot& a, const Spot& b) { return a.count > b.count; });
    const bool symbols = StallSymbolsOn();
    std::string text;
    for (std::size_t i = 0; i < spots.size() && i < top; ++i) {
        const Spot& spot = spots[i];
        const std::uint64_t rip = most(spot.rips).first;
        const auto caller = most(spot.callers);
        text += fmt::format("{}{} {:.0f}%{}", text.empty() ? "" : ", ",
                            Common::SamplingProfiler::DescribeCode(rip),
                            spot.count * 100.0 / double(samples),
                            symbols ? SymbolName(rip) : std::string{});
        if (caller.first != 0) {
            text += fmt::format(" (stack {}{} in {:.0f}%)",
                                Common::SamplingProfiler::DescribeCode(caller.first),
                                symbols ? SymbolName(caller.first) : std::string{},
                                caller.second * 100.0 / double(spot.count));
        }
    }
    if (spots.size() > top) {
        text += fmt::format(", +{} more spots", spots.size() - top);
    }
    if (g_audio_failed != 0) {
        text += fmt::format("; {} samples failed", g_audio_failed);
    }
    if (AudioLoopOn() && !spots.empty()) {
        const std::string regs = AudioLoopText(g_audio_samples, spots.front().bucket);
        if (!regs.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "PRT-AUDIOLOOP [{}]: {} ({} samples), registers low/middle/high: {}",
                        Common::PrtHitch::LocalClock(),
                        Common::SamplingProfiler::DescribeCode(spots.front().bucket),
                        spots.front().count, regs);
        }
    }
    text += AudioMxcsrText();
    g_audio_samples.clear();
    g_audio_failed = 0;
    return text;
}
#else
void AudioLoadPoll() {}

std::string AudioLoadReport(std::size_t, std::uint32_t& samples) {
    samples = 0;
    return {};
}
#endif

} // namespace Common::PrtStallSample
