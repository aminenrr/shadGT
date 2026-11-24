// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "common/types.h"

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
// tlhelp32.h needs the Windows types declared first.
#include <tlhelp32.h>
#endif

namespace Common {
namespace {
std::atomic<u64> cadence_frames{};
std::atomic<u64> cadence_us{};
}
void NoteGameFrameCadence() {
    static thread_local auto last = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (last != std::chrono::steady_clock::time_point{}) {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
        cadence_us.fetch_add(u64(us), std::memory_order_relaxed);
        cadence_frames.fetch_add(1, std::memory_order_release);
    }
    last = now;
}
GameFrameTotals GetGameFrameTotals() {
    const auto frames = cadence_frames.load(std::memory_order_acquire);
    return {frames, cadence_us.load(std::memory_order_relaxed)};
}

// PRT-PERF: the thread sampling below is built without diagnostics too (the PRT-PERF line).
#ifdef _WIN32
namespace {

struct ThreadSample {
    HANDLE handle{};
    u64 cpu_100ns{};
    u64 user_100ns{}; ///< PRT-AUDIOLOAD
    bool seen{};
};

u64 FileTimeTo100ns(const FILETIME& time) {
    return (u64(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

bool PrtThreadModuleOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_PERF_THREAD_MODULE");
        return !(env && env[0] == '0');
    }();
    return on;
}

std::unordered_map<DWORD, std::string>& PrtThreadModules() {
    static std::unordered_map<DWORD, std::string> modules;
    return modules;
}

std::string PrtThreadModule(DWORD tid) {
    auto& modules = PrtThreadModules();
    if (const auto it = modules.find(tid); it != modules.end()) {
        return it->second;
    }
    using QueryThread = LONG(WINAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
    static const QueryThread query = [] {
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll ? reinterpret_cast<QueryThread>(
                           reinterpret_cast<void*>(GetProcAddress(ntdll, "NtQueryInformationThread")))
                     : nullptr;
    }();
    std::string module_name;
    const HANDLE thread = query ? OpenThread(THREAD_QUERY_INFORMATION, FALSE, tid) : nullptr;
    if (thread) {
        PVOID start = nullptr;
        constexpr int ThreadQuerySetWin32StartAddress = 9;
        if (query(thread, ThreadQuerySetWin32StartAddress, &start, sizeof(start), nullptr) >= 0 &&
            start) {
            HMODULE module = nullptr;
            wchar_t path[MAX_PATH] = {};
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCWSTR>(start), &module) &&
                GetModuleFileNameW(module, path, MAX_PATH) != 0) {
                const wchar_t* file = path;
                for (const wchar_t* c = path; *c; ++c) {
                    if (*c == L'\\' || *c == L'/') {
                        file = c + 1;
                    }
                }
                for (const wchar_t* c = file; *c; ++c) {
                    module_name.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
                }
            } else {
                module_name = "no module";
            }
        }
        CloseHandle(thread);
    }
    modules.emplace(tid, module_name);
    return module_name;
}

std::string ThreadName(HANDLE handle, DWORD tid) {
    PWSTR description = nullptr;
    std::string name;
    if (SUCCEEDED(GetThreadDescription(handle, &description)) && description) {
        for (const wchar_t* c = description; *c; ++c) {
            name.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
        }
        LocalFree(description);
    }
    if (name.empty()) {
        name = fmt::format("tid{}", tid);
        if (PrtThreadModuleOn()) {
            const std::string module_name = PrtThreadModule(tid);
            if (!module_name.empty()) {
                name += "@" + module_name;
            }
        }
    }
    return name;
}

std::mutex g_cpu_usage_mutex;
std::unordered_map<DWORD, ThreadSample> g_cpu_threads; ///< guarded by g_cpu_usage_mutex

// PRT-AUDIOLOAD (Round 13): the last window's usage of every thread, for LastNamedThreadLoad.
struct LastUsage {
    DWORD tid;
    double percent;
    double user_percent;
    double kernel_percent;
};
std::vector<LastUsage> g_last_usages;                   ///< guarded by g_cpu_usage_mutex
std::unordered_map<DWORD, std::string> g_usage_names; ///< guarded by g_cpu_usage_mutex

} // namespace

std::string SampleThreadCpuUsage(size_t max_threads) {
    auto& mutex = g_cpu_usage_mutex;
    std::scoped_lock lk{mutex};
    auto& threads = g_cpu_threads;
    static auto last_time = std::chrono::steady_clock::now();

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_100ns =
        std::chrono::duration<double>(now - last_time).count() * 10'000'000.0;
    last_time = now;

    for (auto& [tid, sample] : threads) {
        sample.seen = false;
    }
    struct Usage {
        double percent;
        DWORD tid;
        HANDLE handle;
    };
    std::vector<Usage> usages;
    double total_percent = 0.0;
    g_last_usages.clear(); // PRT-AUDIOLOAD

    const auto account = [&](DWORD tid, bool inserted, auto& sample) {
        FILETIME creation, exit, kernel, user;
        if (!sample.handle || !GetThreadTimes(sample.handle, &creation, &exit, &kernel, &user)) {
            return;
        }
        sample.seen = true;
        const u64 user_time = FileTimeTo100ns(user);
        const u64 cpu = FileTimeTo100ns(kernel) + user_time;
        if (!inserted && elapsed_100ns > 0.0 && cpu >= sample.cpu_100ns) {
            const double percent = double(cpu - sample.cpu_100ns) * 100.0 / elapsed_100ns;
            total_percent += percent;
            usages.push_back({percent, tid, sample.handle});
            const double user_percent =
                user_time >= sample.user_100ns
                    ? double(user_time - sample.user_100ns) * 100.0 / elapsed_100ns
                    : 0.0;
            g_last_usages.push_back(
                {tid, percent, user_percent, std::max(0.0, percent - user_percent)});
        }
        sample.cpu_100ns = cpu;
        sample.user_100ns = user_time;
    };
    // PERF-GT7-054: this process's threads only (NtGetNextThread); the Toolhelp snapshot walks
    // every thread on the machine, 60 ms here with browsers open, and ran in a frame.
    const bool walked = ForEachProcessThread(
        THREAD_QUERY_LIMITED_INFORMATION, [&](void* handle, unsigned long tid) {
            auto [it, inserted] = threads.try_emplace(tid);
            if (inserted) {
                it->second.handle = static_cast<HANDLE>(handle);
            }
            account(tid, inserted, it->second);
            return inserted; // a new thread's handle is kept; a known one's is closed
        });
    if (!walked) {
        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) {
            return {};
        }
        const DWORD pid = GetCurrentProcessId();
        THREADENTRY32 entry{.dwSize = sizeof(THREADENTRY32)};
        for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != pid) {
                continue;
            }
            auto [it, inserted] = threads.try_emplace(entry.th32ThreadID);
            if (inserted) {
                it->second.handle =
                    OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
            }
            account(entry.th32ThreadID, inserted, it->second);
        }
        CloseHandle(snapshot);
    }

    std::erase_if(threads, [](const auto& pair) {
        if (!pair.second.seen && pair.second.handle) {
            CloseHandle(pair.second.handle);
        }
        return !pair.second.seen;
    });
    std::erase_if(PrtThreadModules(),
                  [&](const auto& pair) { return !threads.contains(pair.first); });
    std::erase_if(g_usage_names, [&](const auto& pair) { return !threads.contains(pair.first); });

    std::ranges::sort(usages, std::greater{}, &Usage::percent);
    std::string summary = fmt::format("process {:.0f}% of {}%;", total_percent,
                                      std::max(1u, std::thread::hardware_concurrency()) * 100);
    for (size_t i = 0; i < std::min(max_threads, usages.size()); ++i) {
        summary += fmt::format(" {} {:.0f}%", ThreadName(usages[i].handle, usages[i].tid),
                               usages[i].percent);
    }
    return summary;
}

// PRT-AUDIOLOAD (pr-tests, Round 13): see perf_monitor.h.
NamedThreadLoad LastNamedThreadLoad(const char* needle) {
    std::scoped_lock lk{g_cpu_usage_mutex};
    NamedThreadLoad best;
    for (const LastUsage& usage : g_last_usages) {
        std::string name;
        if (const auto name_it = g_usage_names.find(usage.tid); name_it != g_usage_names.end()) {
            name = name_it->second;
        } else {
            const auto thread_it = g_cpu_threads.find(usage.tid);
            if (thread_it == g_cpu_threads.end() || !thread_it->second.handle) {
                continue;
            }
            name = ThreadName(thread_it->second.handle, usage.tid);
            if (!name.starts_with("tid")) { // a thread named later is looked up again
                g_usage_names.emplace(usage.tid, name);
            }
        }
        if (name.find(needle) == std::string::npos ||
            (best.tid != 0 && usage.percent <= best.percent)) {
            continue;
        }
        best.tid = usage.tid;
        best.percent = usage.percent;
        best.user_percent = usage.user_percent;
        best.kernel_percent = usage.kernel_percent;
        best.name = std::move(name);
    }
    return best;
}

u32 CurrentOsThreadId() {
    return static_cast<u32>(GetCurrentThreadId());
}
#else
std::string SampleThreadCpuUsage(size_t) {
    return {};
}

NamedThreadLoad LastNamedThreadLoad(const char*) {
    return {};
}

u32 CurrentOsThreadId() {
    return 0;
}
#endif

#ifdef SHADGT_DIAGNOSTICS

u64 PhaseTimer::ReadTscFallback() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

std::array<std::atomic<u64>, size_t(Phase::Count)>& GetPhaseTicks() {
    return g_phase_ticks;
}

// The counters only grow; each reader keeps its own snapshot of the values it last reported.
template <size_t N>
static std::array<u64, N> Snapshot(const std::array<std::atomic<u64>, N>& values) {
    std::array<u64, N> out{};
    for (size_t i = 0; i < N; ++i) {
        out[i] = values[i].load(std::memory_order_relaxed);
    }
    return out;
}

static std::string TakePhaseTimes() {
    static constexpr std::array<const char*, size_t(Phase::Count)> Names = {
        "setup",    "pipeline",   "targets",         "vtx/idx",     "buffers",
        "textures", "rebind",     "begin-rendering", "descriptors", "dynamic",
        "record",   "draw-total", "dispatch-total",  "submit",      "[tex-find]",
        "[tex-view]", "[tex-transit]", "[buf-obtain]", "[read-ahead-eval]",
        "[program-match]", "[select-ahead]", "[tex-pass1]", "[tex-samplers]",
        "[buf-direct]", "[vtx-bind]", "[idx-bind]", "[sel-flatten]", "[sel-spec]"};
    static u64 last_tsc = PhaseTimer::ReadTsc();
    static auto last_time = std::chrono::steady_clock::now();
    const u64 tsc = PhaseTimer::ReadTsc();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_time).count();
    const double ticks_per_ms = seconds > 0 ? double(tsc - last_tsc) / (seconds * 1000.0) : 1.0;
    last_tsc = tsc;
    last_time = now;
    static auto last_ticks = Snapshot(GetPhaseTicks());
    const auto ticks = Snapshot(GetPhaseTicks());
    std::string out;
    for (size_t i = 0; i < Names.size(); ++i) {
        out +=
            fmt::format(" {}={:.1f}ms", Names[i], double(ticks[i] - last_ticks[i]) / ticks_per_ms);
    }
    last_ticks = ticks;
    return out;
}

bool LeanRun() {
    static const bool lean = [] {
        const char* env = std::getenv("SHADGT_LEAN");
        return env && env[0] == '1';
    }();
    return lean;
}

#endif // SHADGT_DIAGNOSTICS
bool PerfFeatureEnabled(u32 id) {
    static const std::vector<u32> disabled = [] {
        std::vector<u32> ids;
        if (const char* env = std::getenv("SHADGT_DISABLE_PERF")) {
            std::string list{env};
            size_t pos = 0;
            while (pos < list.size()) {
                const size_t end = std::min(list.find(',', pos), list.size());
                try {
                    ids.push_back(static_cast<u32>(std::stoul(list.substr(pos, end - pos))));
                } catch (...) {
                }
                pos = end + 1;
            }
        }
        return ids;
    }();
    return std::ranges::find(disabled, id) == disabled.end();
}

#ifdef SHADGT_DIAGNOSTICS
WorkCounters& GetWorkCounters() {
    static WorkCounters counters;
    return counters;
}

// A copy of the counters at one moment, for differences between two moments.
struct WorkSnapshot {
    u64 pm4_packets, draws, dispatches, find_image, obtain_buffer, obtain_stream, uploads,
        upload_bytes, protects, protect_bytes, shaders_compiled, pipelines_compiled, upload_epochs,
        waits_skipped, gpu_waits, gpu_wait_us, frontend_waits, frontend_wait_us, pipeline_waits,
        pipeline_wait_us, file_reads, file_read_bytes, file_read_us;
};

static WorkSnapshot TakeSnapshot() {
    const auto& c = GetWorkCounters();
    const auto l = [](const auto& v) { return v.load(std::memory_order_relaxed); };
    return {l(c.pm4_packets),      l(c.draws),
            l(c.dispatches),       l(c.find_image),
            l(c.obtain_buffer),    l(c.obtain_stream),
            l(c.uploads),          l(c.upload_bytes),
            l(c.protects),         l(c.protect_bytes),
            l(c.shaders_compiled), l(c.pipelines_compiled),
            l(c.upload_epochs),    l(c.waits_skipped),
            l(c.gpu_waits),        l(c.gpu_wait_us),
            l(c.frontend_waits),   l(c.frontend_wait_us),
            l(c.pipeline_waits),   l(c.pipeline_wait_us),
            l(c.file_reads),       l(c.file_read_bytes),
            l(c.file_read_us)};
}

static WorkSnapshot Difference(const WorkSnapshot& now, const WorkSnapshot& then) {
    WorkSnapshot d;
    const auto* a = reinterpret_cast<const u64*>(&now);
    const auto* b = reinterpret_cast<const u64*>(&then);
    auto* out = reinterpret_cast<u64*>(&d);
    for (size_t i = 0; i < sizeof(WorkSnapshot) / sizeof(u64); ++i) {
        out[i] = a[i] - b[i];
    }
    return d;
}

void NoteGameFrame() {
    static const double threshold_ms = [] {
        const char* env = DiagEnv("SHADGT_STALL_MS");
        return env && *env ? std::stod(env) : 100.0;
    }();
    static auto last_time = std::chrono::steady_clock::now();
    static u64 last_tsc = PhaseTimer::ReadTsc();
    static auto last_work = TakeSnapshot();
    static auto last_ticks = Snapshot(GetPhaseTicks());
    static u64 frame = 0;
    static u64 stalls = 0;
    const auto now = std::chrono::steady_clock::now();
    const u64 tsc = PhaseTimer::ReadTsc();
    const auto work = TakeSnapshot();
    const auto ticks = Snapshot(GetPhaseTicks());
    const double ms = std::chrono::duration<double, std::milli>(now - last_time).count();
    ++frame;
    // DIAG-046: frame-time distribution every 10 s, and frames well above the recent median
    // (micro hitches) with where the GPU threads' time went.
    {
        static std::vector<double> window_ms;
        static std::array<double, 64> recent{};
        static u32 recent_count = 0;
        static auto window_start = now;
        static u32 hitch_lines = 0;
        window_ms.push_back(ms);
        std::array<double, 64> sorted_recent = recent;
        const u32 n_recent = std::min<u32>(recent_count, 64);
        std::sort(sorted_recent.begin(), sorted_recent.begin() + n_recent);
        const double median = n_recent ? sorted_recent[n_recent / 2] : ms;
        recent[recent_count++ % 64] = ms;
        if (DiagnosticsBuilt && n_recent >= 16 && ms >= 25.0 && ms > 2.0 * median &&
            ms < threshold_ms && hitch_lines < 40) {
            ++hitch_lines;
            const double ticks_per_ms = ms > 0 ? double(tsc - last_tsc) / ms : 1.0;
            const auto phase_ms = [&](Phase phase) {
                return double(ticks[size_t(phase)] - last_ticks[size_t(phase)]) / ticks_per_ms;
            };
            const auto d = Difference(work, last_work);
            LOG_WARNING(Render_Vulkan,
                        "DIAG-046 hitch: frame {} took {:.1f} ms (median {:.1f}). Draws {:.1f} ms "
                        "(pipeline {:.1f}, buffers {:.1f}, textures {:.1f}), dispatches {:.1f}, "
                        "submits {:.1f}; GPU waits {} {:.1f} ms, command-buffer waits {} {:.1f} "
                        "ms, pipeline waits {} {:.1f} ms; {} draws, {} pipelines, {} shaders, "
                        "{} uploads {} KB, {} protections, file reads {} {:.1f} ms",
                        frame, ms, median, phase_ms(Phase::DrawTotal), phase_ms(Phase::Pipeline),
                        phase_ms(Phase::Buffers), phase_ms(Phase::Textures),
                        phase_ms(Phase::DispatchTotal), phase_ms(Phase::Submit), d.gpu_waits,
                        d.gpu_wait_us / 1000.0, d.frontend_waits, d.frontend_wait_us / 1000.0,
                        d.pipeline_waits, d.pipeline_wait_us / 1000.0, d.draws,
                        d.pipelines_compiled, d.shaders_compiled, d.uploads,
                        d.upload_bytes / 1024, d.protects, d.file_reads, d.file_read_us / 1000.0);
        }
        if (now - window_start >= std::chrono::seconds{10} && !window_ms.empty()) {
            std::vector<double> sorted = window_ms;
            std::sort(sorted.begin(), sorted.end());
            double total = 0;
            u32 over25 = 0, over33 = 0, over50 = 0, over100 = 0;
            for (const double f : sorted) {
                total += f;
                over25 += f > 25.0;
                over33 += f > 34.0;
                over50 += f > 50.0;
                over100 += f > 100.0;
            }
            const double avg = total / sorted.size();
            // 1% low: the average frame rate of the slowest 1% of frames.
            const size_t slow = std::max<size_t>(1, sorted.size() / 100);
            double slow_total = 0;
            for (size_t i = sorted.size() - slow; i < sorted.size(); ++i) {
                slow_total += sorted[i];
            }
            LOG_WARNING(Render_Vulkan,
                        "DIAG-046 frames in {:.1f} s: {} frames, avg {:.1f} ms ({:.1f} fps), "
                        "median {:.1f} ms, 1% low {:.1f} fps, max {:.0f} ms; over 25 ms {}, over "
                        "34 ms {}, over 50 ms {}, over 100 ms {}; {} hitch lines",
                        std::chrono::duration<double>(now - window_start).count(), sorted.size(),
                        avg, 1000.0 / avg, sorted[sorted.size() / 2],
                        1000.0 / (slow_total / slow), sorted.back(), over25, over33, over50,
                        over100, hitch_lines);
            window_ms.clear();
            window_start = now;
            hitch_lines = 0;
        }
    }
    if (DiagnosticsBuilt && ms >= threshold_ms) {
        const double ticks_per_ms = ms > 0 ? double(tsc - last_tsc) / ms : 1.0;
        const auto phase_ms = [&](Phase phase) {
            return double(ticks[size_t(phase)] - last_ticks[size_t(phase)]) / ticks_per_ms;
        };
        const auto d = Difference(work, last_work);
        const double draw = phase_ms(Phase::DrawTotal);
        const double dispatch = phase_ms(Phase::DispatchTotal);
        const double submit = phase_ms(Phase::Submit);
        const double gpu_wait = d.gpu_wait_us / 1000.0;
        const double frontend_wait = d.frontend_wait_us / 1000.0;
        const double pipeline_wait = d.pipeline_wait_us / 1000.0;
        const double accounted = draw + dispatch + submit + gpu_wait + frontend_wait;
        if (++stalls <= 400 || stalls % 50 == 0) {
            LOG_WARNING(
                Render_Vulkan,
                "DIAG-033 stall {}: game frame {} took {:.0f} ms. GPU thread: draws {:.0f} ms "
                "(pipeline {:.0f}, buffers {:.0f}, textures {:.0f}, vtx/idx {:.0f}), dispatches "
                "{:.0f} ms, submits {:.0f} ms; waited for the GPU {} times {:.0f} ms, for "
                "command-buffer waits {} times {:.0f} ms, for pipelines {} times {:.0f} ms; "
                "unaccounted {:.0f} ms. Work: {} draws, {} dispatches, {} shaders and {} "
                "pipelines compiled, {} uploads {} KB, {} protections {} KB. Game file reads: {} "
                "({} KB, {:.0f} ms)",
                stalls, frame, ms, draw, phase_ms(Phase::Pipeline), phase_ms(Phase::Buffers),
                phase_ms(Phase::Textures), phase_ms(Phase::VertexIndex), dispatch, submit,
                d.gpu_waits, gpu_wait, d.frontend_waits, frontend_wait, d.pipeline_waits,
                pipeline_wait, std::max(0.0, ms - accounted), d.draws, d.dispatches,
                d.shaders_compiled, d.pipelines_compiled, d.uploads, d.upload_bytes / 1024,
                d.protects, d.protect_bytes / 1024, d.file_reads, d.file_read_bytes / 1024,
                d.file_read_us / 1000.0);
        }
    }
    last_time = now;
    last_tsc = tsc;
    last_work = work;
    last_ticks = ticks;
}

std::string TakeWorkCounters() {
    static auto last = TakeSnapshot();
    const auto now = TakeSnapshot();
    const auto c = Difference(now, last);
    last = now;
    return fmt::format("{} PM4 packets, {} draws, {} dispatches, {} image lookups, {} buffer "
                       "binds ({} streamed), {} uploads ({} KB), {} protection calls ({} KB), "
                       "{} shaders and {} pipelines compiled, {} upload epochs, {} GPU-side waits "
                       "ordered behind pending fences",
                       c.pm4_packets, c.draws, c.dispatches, c.find_image, c.obtain_buffer,
                       c.obtain_stream, c.uploads, c.upload_bytes / 1024, c.protects,
                       c.protect_bytes / 1024, c.shaders_compiled, c.pipelines_compiled,
                       c.upload_epochs, c.waits_skipped) +
           ";" + TakePhaseTimes();
}

#endif

} // namespace Common
