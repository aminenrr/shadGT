// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include "common/diagnostics.h"
#include "common/elf_info.h"
#include "common/game_title.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/scm_rev.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/lane_read_policy.h"

#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/cache_storage.h"
#include "video_core/shader_store_file.h"
#include "video_core/shader_store_policy.h"
#include "video_core/deferred_task_queue.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

#include <miniz.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <array>
#include <atomic>
#include <cstring>
#include <string_view>
#include <utility>

namespace {

Storage::DeferredTaskQueue<std::packaged_task<void()>> requests;
std::atomic_bool directory_writes_paused{};
bool reported_blob_error{}; // Only the I/O worker uses this after Open.

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

constexpr std::array<u8, 4> CompressedMagic{'R', 'Z', 'S', '1'};
constexpr std::size_t CompressedHeader = 8;
std::atomic<u64> compress_writes{};
std::atomic<u64> compress_plain_bytes{};
std::atomic<u64> compress_stored_bytes{};
std::atomic<u64> inflate_count{};
std::atomic<u64> inflate_us{};
std::atomic<u64> inflate_failed{};

bool CompressOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_SHADER_STORE_COMPRESS");
        if (env && *env) {
            return env[0] != '0';
        }
        return Common::IsGt7();
    }();
    return on;
}

void ReportCompress(bool final_line) {
    const u64 writes = compress_writes.load(std::memory_order_relaxed);
    const u64 plain = compress_plain_bytes.load(std::memory_order_relaxed);
    const u64 stored = compress_stored_bytes.load(std::memory_order_relaxed);
    if (writes == 0 && inflate_count.load(std::memory_order_relaxed) == 0) {
        return;
    }
    LOG_WARNING(Render, "PRT-STORECOMPRESS{}: {} SPIR-V written, {} KB -> {} KB (ratio {:.2f}); {} "
                        "compressed read, {} ms inflating, {} unreadable",
                final_line ? " (session)" : "", writes, plain >> 10, stored >> 10,
                stored ? double(plain) / double(stored) : 0.0,
                inflate_count.load(std::memory_order_relaxed),
                inflate_us.load(std::memory_order_relaxed) / 1000,
                inflate_failed.load(std::memory_order_relaxed));
}

std::vector<u8> DeflateShaderBlob(const u8* data, std::size_t size) {
    mz_ulong bound = mz_compressBound(static_cast<mz_ulong>(size));
    std::vector<u8> out(CompressedHeader + bound);
    if (mz_compress2(out.data() + CompressedHeader, &bound, data, static_cast<mz_ulong>(size),
                     MZ_DEFAULT_LEVEL) != MZ_OK ||
        CompressedHeader + bound >= size) {
        return {};
    }
    std::memcpy(out.data(), CompressedMagic.data(), CompressedMagic.size());
    const u32 plain = static_cast<u32>(size);
    std::memcpy(out.data() + 4, &plain, sizeof(plain));
    out.resize(CompressedHeader + bound);
    return out;
}

} // namespace

namespace Storage {

namespace {

// SHD-010 (shadgt7) cleared a store filled by another renderer build. FIX-271 keeps that intent
// without build or GPU identity purges: stores carry a translation version (FIX-213) and an
// incompatible store is preserved while a compatible numbered sibling is opened instead.

// Tests: a store of its own for this tag. A test keeps one store per renderer and switches.
// NEO-GT7-007: SHADGT_NEO_SDWA_CMP=0 translates SDWA compares the old way (to VCC), which breaks the
// PS4 Pro checkerboard resolve; those translations get a store of their own, so a session run once
// with the switch leaves nothing wrong in the usual store. Only with the Pro mode's switch
// (SHADGT_NEO=1; ryGT7's libkernel reads the same switch as RYGT_NEO, which SWITCH-001 hands
// on under this name): the base mode has no SDWA, so its store stays.
std::string StoreTag() {
    const char* env = std::getenv("SHADGT_SHADER_STORE_TAG");
    std::string tag = env && *env ? std::string{"-"} + env : std::string{};
    // NEO-GTS-505: whatever turned the Pro mode on (GT Sport's SHADGT_PS4_PRO or config too):
    // the base mode has no SDWA, so the tag costs it nothing.
    const char* sdwa = std::getenv("SHADGT_NEO_SDWA_CMP");
    if (sdwa && sdwa[0] == '0') {
        tag += "-sdwa-vcc";
    }
    return tag;
}


bool StoreCleanupOn() {
    const char* env = std::getenv("RYGT_SHADER_STORE_CLEANUP");
    if (env && *env) {
        return env[0] != '0';
    }
    return Common::IsGt7();
}

std::string StoreFamilyPrefix(std::string_view version) {
    return std::string{version.substr(0, version.find(" ("))};
}

std::pair<u64, u64> TreeSize(const std::filesystem::path& path) {
    u64 bytes = 0;
    u64 files = 0;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator{path, ec};
         !ec && it != std::filesystem::recursive_directory_iterator{}; it.increment(ec)) {
        std::error_code size_ec;
        if (it->is_regular_file(size_ec)) {
            const auto size = it->file_size(size_ec);
            if (!size_ec) {
                bytes += size;
                ++files;
            }
        }
    }
    return {bytes, files};
}

void CleanStaleStores(const std::stop_token& stop, std::filesystem::path original,
                      std::filesystem::path live, std::string expected) {
    Common::SetCurrentThreadName("shadGT:StoreClean");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    const auto family = StoreFamilyPrefix(expected);
    u64 freed = 0;
    u32 deleted = 0;
    for (unsigned suffix = 0; suffix != 1024 && !stop.stop_requested(); ++suffix) {
        auto path = original;
        if (suffix) path += ".v" + std::to_string(suffix);
        std::error_code ec;
        if (!std::filesystem::is_directory(path, ec)) {
            if (suffix) break;
            continue;
        }
        if (path == live) continue;
        std::string version;
        bool has_version = false;
        {
            std::ifstream file(path / "store-version.txt");
            has_version = static_cast<bool>(file);
            if (file) std::getline(file, version);
        }
        const bool stale = !has_version || (version != expected && StoreFamilyPrefix(version) == family);
        if (!stale) {
            LOG_INFO(Render, "PRT-STORECLEAN: kept {} (store of another GPU family: {})",
                     path.filename().string(), version);
            continue;
        }
        const auto [bytes, files] = TreeSize(path);
        std::filesystem::remove_all(path, ec);
        if (ec) {
            LOG_WARNING(Render, "PRT-STORECLEAN: could not delete {} ({})", path.filename().string(),
                        ec.message());
            continue;
        }
        freed += bytes;
        ++deleted;
        LOG_WARNING(Render, "PRT-STORECLEAN: deleted stale shader store {} ({} MB, {} files, {})",
                    path.filename().string(), bytes >> 20, files,
                    has_version ? "older translations" : "no store version");
    }
    const auto old_cache = original.parent_path() /
                           (std::string{Common::ElfInfo::Instance().GameSerial()} + ".vkpipelinecache");
    std::error_code ec;
    if (!stop.stop_requested() && std::filesystem::is_regular_file(old_cache, ec)) {
        const auto bytes = std::filesystem::file_size(old_cache, ec);
        if (!ec && std::filesystem::remove(old_cache, ec) && !ec) {
            freed += bytes;
            ++deleted;
            LOG_WARNING(Render, "PRT-STORECLEAN: deleted old driver cache {} ({} MB, no longer read)",
                        old_cache.filename().string(), bytes >> 20);
        } else {
            LOG_WARNING(Render, "PRT-STORECLEAN: could not delete {} ({})",
                        old_cache.filename().string(), ec.message());
        }
    }
    LOG_WARNING(Render, "PRT-STORECLEAN: {} deleted, {} MB freed (RYGT_SHADER_STORE_CLEANUP=0 keeps them)",
                deleted, freed >> 20);
}
} // namespace

bool IsCompressedShaderBlob(const std::vector<u8>& data) {
    return data.size() >= CompressedHeader &&
           std::memcmp(data.data(), CompressedMagic.data(), CompressedMagic.size()) == 0;
}

bool InflateShaderBlob(std::vector<u8>& data) {
    if (!IsCompressedShaderBlob(data)) {
        return true;
    }
    const auto start = std::chrono::steady_clock::now();
    u32 plain = 0;
    std::memcpy(&plain, data.data() + 4, sizeof(plain));
    std::vector<u8> out(plain);
    mz_ulong length = plain;
    const bool ok = mz_uncompress(out.data(), &length, data.data() + CompressedHeader,
                                  static_cast<mz_ulong>(data.size() - CompressedHeader)) == MZ_OK &&
                    length == plain;
    if (!ok) {
        inflate_failed.fetch_add(1, std::memory_order_relaxed);
        data.clear();
        return false;
    }
    data = std::move(out);
    inflate_count.fetch_add(1, std::memory_order_relaxed);
    inflate_us.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start).count()),
                         std::memory_order_relaxed);
    return true;
}

std::filesystem::path RetranslationMarksPath() {
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
           fmt::format("{}{}.else-scope-fix", Common::ElfInfo::Instance().GameSerial(), StoreTag());
}

bool StoreMayHoldOtherBuilds() {
    // FIX-271: stores are never cleared for another build, so they may always hold its
    // translations (FIX-GT7-055's retranslation checks stay on).
    return true;
}

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadGT:PipelineCacheIO");

    while (auto request = requests.Pop(stoken)) {
        if (request->valid()) (*request)();
    }
}

constexpr std::string GetBlobFileExtension(BlobType type) {
    switch (type) {
    case BlobType::ShaderMeta: {
        return "meta";
    }
    case BlobType::ShaderBinary: {
        return "spv";
    }
    case BlobType::PipelineKey: {
        return "key";
    }
    case BlobType::ShaderProfile: {
        return "bin";
    }
    default:
        UNREACHABLE();
    }
}

void DataBase::Open() {
    if (opened) {
        return;
    }
    compatible_sibling = false;

    const auto& game_info = Common::ElfInfo::Instance();

    using namespace Common::FS;
    archived = EmulatorSettings.IsPipelineCacheArchived();
    if (archived) {
        mz_zip_zero_struct(&zip_ar);

        cache_path = GetUserPath(PathType::CacheDir) /
                     std::filesystem::path{game_info.GameSerial()}.replace_extension(".zip");
        if (!mz_zip_reader_init_file(&zip_ar, cache_path.string().c_str(),
                                     MZ_ZIP_FLAG_READ_ALLOW_WRITING) ||
            !mz_zip_validate_archive(&zip_ar, 0)) {
            LOG_INFO(Render, "Cache archive {} is not found or archive is corrupted",
                     cache_path.string().c_str());
            mz_zip_reader_end(&zip_ar);
            // Never let a writer truncate the old archive. Use a directory store.
            LOG_WARNING(Render, "Preserving unreadable or missing cache archive {}; using a directory store",
                        cache_path.string());
            archived = false;
        }
    }
    if (!archived) {
        const char* writes = std::getenv("SHADGT_SHADER_STORE_WRITES");
        directory_writes_paused = writes && std::string_view{writes} == "0";
        reported_blob_error = false;
        cache_path = GetUserPath(PathType::CacheDir) / game_info.GameSerial();
        // D2: direct-memory shaders differ from the stored ones, which load unchecked.
        if (VideoCore::DirectMemory::Enabled()) {
            // SHD-002: stage descriptor sets: translations of their own.
            cache_path += "-direct-sets";
        }
        cache_path += StoreTag(); // SHADGT_SHADER_STORE_TAG (tests)
        // FIX-213: translations from incompatible shader policies must not be reused.
        // FIX-271 preserves existing stores and opens a compatible sibling instead.
        // Raise the version whenever a change makes stored translations wrong.
        // AMD-311: each vertex order of AMD's per-vertex reads has a store of its own version.
        // AMD-312: Radeon RX 9000 cards (RDNA4) use store 4: an RX 9060 XT drew the road at
        // Brands Hatch as streaks with a store made earlier and right once it was translated
        // again. Other GPUs (the RX 7000 series) keep store 3, so their store stays in use.
        const char* order_env = std::getenv("SHADGT_AMD_VERTEX_ORDER");
        // AMD-313: RX 7000 cards leave FIX-215/216 out (purple car); their store 5 sets aside
        // stores translated with them.
        const auto family = Vulkan::GetStoreGpuFamily();
        const std::string StoreVersion =
            std::string{family == Vulkan::StoreGpuFamily::AmdRdna4
                            ? "RyGT shader store 4 (FIX-215/216 NaN on AMD, AMD-311 vertex order)"
                        : family == Vulkan::StoreGpuFamily::AmdRdna3
                            ? "RyGT shader store 5 (RX 7000: FIX-215/216 off, AMD-313)"
                        : family == Vulkan::StoreGpuFamily::AmdLegacy
                            ? "RyGT shader store 8 (RX 6000 to RX 400: AMD-306/307/308/315, order "
                              "check gated)"
                            : "RyGT shader store 3 (FIX-215/216 MED3, MIN3, MAX3 NaN on AMD)"} +
            (order_env && *order_env ? fmt::format(", SHADGT_AMD_VERTEX_ORDER={}", order_env[0])
                                     : std::string{}) +
            [] {
                // AMD-313, AMD-306, AMD-307, AMD-308: the switches that change translations.
                std::string parts;
                for (const char* name :
                     {"SHADGT_AMD_NAN_RULES", "SHADGT_QUAD_CONTROL", "SHADGT_VERTEX_ORDER",
                      "SHADGT_FP32_IMAGE_ATOMICS", "SHADGT_MANUAL_INTERP"}) {
                    if (const char* env = std::getenv(name); env && *env) {
                        parts += fmt::format(", {}={}", name, env[0]);
                    }
                }
                // DIAG-230 (port): without SHADGT_MANUAL_INTERP, the diagnosis switch changes
                // NVIDIA's translations (vk_pipeline_cache.cpp); they are kept apart.
                if (const char* manual = std::getenv("SHADGT_MANUAL_INTERP");
                    !(manual && *manual) && Common::DiagEnv("SHADGT_DIAG_NO_MANUAL_INTERP")) {
                    parts += ", no manual interp";
                }
                // FIX-GTS-441: translations with the invariant position (the default) are kept
                // apart from those made without it.
                if (const char* env = std::getenv("SHADGT_INVARIANT_POSITION");
                    !(env && env[0] == '0')) {
                    parts += ", invariant position";
                }
                // DIAG-GTS-459: translations with this run's counter address are never reused.
                if (const char* env = std::getenv("SHADGT_DIAG_LDS_READS"); env && env[0] == '1') {
                    parts += fmt::format(", lds read diag {}",
                                         std::chrono::system_clock::now().time_since_epoch().count());
                }
                // PERF-GTS-458: hull LDS writer stores shared by the patch's invocations.
                if (const char* env = std::getenv("SHADGT_TESS_LDS_SPREAD");
                    !(env && env[0] == '0')) {
                    parts += ", lds spread";
                }
                // PERF-GTS-473: direct buffer reads that share one branch per run of reads.
                if (Shader::Backend::SPIRV::DirectLoadGroupsEnabled()) {
                    parts += ", direct groups";
                }
                // DIAG-GTS-447: constant-output diagnosis shaders get a store of their own.
                if (const char* env = std::getenv("SHADGT_DIAG_FS_CONST"); env && *env) {
                    parts += fmt::format(", fs const {}", env);
                }
                // FIX-GTS-443: -1..1 depth converted in the vertex shader (opt-in).
                if (const char* env = std::getenv("SHADGT_DEPTH_CLIP_CONTROL");
                    env && env[0] == '0') {
                    parts += ", depth in shader";
                }
                return parts;
            }() + std::string{Shader::LaneReadStoreSuffix(Shader::LaneReadUnbranchEnabled())};
        const auto original = cache_path;
        const auto selected = VideoCore::OpenCompatibleShaderStore(original, StoreVersion);
        if (!selected) {
            LOG_WARNING(Render, "No compatible shader-store directory available; preserving existing stores");
            return;
        }
        cache_path = *selected;
        compatible_sibling = cache_path != original;
        if (cache_path != original)
            LOG_INFO(Render, "Preserved incompatible shader store {}; using compatible sibling {}",
                     original.string(), cache_path.string());
        if (StoreCleanupOn()) {
            try {
                clean_worker = std::jthread{CleanStaleStores, original, cache_path, StoreVersion};
            } catch (const std::exception& e) {
                LOG_WARNING(Render, "PRT-STORECLEAN: unavailable: {}", e.what());
            }
        }
    }

    io_worker = std::jthread{ProcessIO};
    opened = true;
}

void DataBase::OpenLegacy() {
    if (opened || EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    const auto& game_info = Common::ElfInfo::Instance();
    cache_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / game_info.GameSerial();
    std::error_code ec;
    opened = std::filesystem::is_directory(cache_path, ec);
    LOG_INFO(Render, "D10: old renderer's shader store {} at {}", opened ? "found" : "not found",
             cache_path.string());
}

void DataBase::Close() {
    if (!IsOpened()) {
        return;
    }

    io_worker.request_stop();
    io_worker.join();
    ReportCompress(true);
    if (tidy_worker.joinable()) {
        tidy_worker.request_stop();
        tidy_worker.join();
    }
    if (clean_worker.joinable()) {
        clean_worker.request_stop();
        clean_worker.join();
    }

    if (archived) {
        mz_zip_writer_finalize_archive(&zip_ar);
        mz_zip_writer_end(&zip_ar);
    }

    opened = false;
    LOG_INFO(Render, "Cache dumped");
}

template <typename T>
bool WriteVector(const BlobType type, std::filesystem::path&& path_, std::vector<T>&& v, bool archived) {
    if (!archived && directory_writes_paused.load(std::memory_order_relaxed)) {
        return false;
    }
    {
        auto request = std::packaged_task<void()>{[type, path=std::move(path_), data=std::move(v), archived]() mutable {
            path.replace_extension(GetBlobFileExtension(type));
            if (archived) {
                ASSERT_MSG(!ar_is_read_only,
                           "The archive is read-only. Did you forget to call `FinishPreload`?");
                if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), data.data(),
                                           data.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                    LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
                }
            } else if (!directory_writes_paused.load(std::memory_order_relaxed)) {
                std::vector<u8> packed;
                if (type == BlobType::ShaderBinary && CompressOn() && !data.empty()) {
                    packed = DeflateShaderBlob(reinterpret_cast<const u8*>(data.data()),
                                               data.size() * sizeof(T));
                    const u64 writes = compress_writes.fetch_add(1, std::memory_order_relaxed) + 1;
                    compress_plain_bytes.fetch_add(data.size() * sizeof(T), std::memory_order_relaxed);
                    compress_stored_bytes.fetch_add(packed.empty() ? data.size() * sizeof(T)
                                                                   : packed.size(),
                                                    std::memory_order_relaxed);
                    if (writes % 1000 == 0) {
                        ReportCompress(false);
                    }
                }
                const auto error = PublishShaderBlob(path, [&](std::FILE* file) {
                    if (!packed.empty()) {
                        return std::fwrite(packed.data(), 1, packed.size(), file) == packed.size();
                    }
                    return data.empty() ||
                           std::fwrite(data.data(), sizeof(T), data.size(), file) == data.size();
                });
                if (error == std::errc::no_space_on_device) {
                    directory_writes_paused = true;
                    LOG_WARNING(Render, "Shader cache write failed for {} ({}); pausing cache "
                                        "writes for this session. Rendering and cache reads continue.",
                                path.string(), error.message());
                } else if (error && !reported_blob_error) {
                    // A reader or antivirus can briefly lock a single destination on Windows.
                    reported_blob_error = true;
                    LOG_WARNING(Render, "Skipping shader cache entry {} ({}); other entries will "
                                        "still be written.", path.string(), error.message());
                }
            }
        }};
        requests.Push(std::move(request));
    }

    return true;
}

template <typename T>
void LoadVector(BlobType type, std::filesystem::path& path, std::vector<T>& v, bool archived) {
    using namespace Common::FS;
    path.replace_extension(GetBlobFileExtension(type));
    if (archived) {
        int index{-1};
        index = mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0);
        if (index < 0) {
            LOG_WARNING(Render, "File {} is not found in the archive", path.string().c_str());
            return;
        }
        mz_zip_archive_file_stat stat{};
        mz_zip_reader_file_stat(&zip_ar, index, &stat);
        v.resize(stat.m_uncomp_size / sizeof(T));
        mz_zip_reader_extract_to_mem(&zip_ar, index, v.data(), stat.m_uncomp_size, 0);
    } else if (type == BlobType::ShaderBinary) {
        const auto file = IOFile{path, FileAccessMode::Read};
        std::vector<u8> bytes(file.GetSize());
        file.Read(bytes);
        InflateShaderBlob(bytes);
        v.resize(bytes.size() / sizeof(T));
        if (!v.empty()) {
            std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
        }
    } else {
        const auto file = IOFile{path, FileAccessMode::Read};
        v.resize(file.GetSize() / sizeof(T));
        file.Read(v);
    }
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u8>&& data) {
    if (!opened) {
        return false;
    }

    auto path = archived ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data), archived);
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u32>&& data) {
    if (!opened) {
        return false;
    }

    auto path = archived ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data), archived);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u8>& data) {
    if (!opened) {
        return;
    }

    auto path = archived ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data, archived);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    if (!opened) {
        return;
    }

    auto path = archived ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data, archived);
}

bool DataBase::Exists(BlobType type, const std::string& name) const {
    if (!opened) {
        return false;
    }
    if (archived) {
        std::filesystem::path path{name};
        path.replace_extension(GetBlobFileExtension(type));
        return mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0) >= 0;
    }
    auto path = cache_path / name;
    path.replace_extension(GetBlobFileExtension(type));
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

void DataBase::ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func) {
    const auto& ext = GetBlobFileExtension(type);
    if (archived) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            file_name.fill(0);
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            if (std::string{file_name.data()}.ends_with(ext)) {
                mz_zip_archive_file_stat stat{};
                mz_zip_reader_file_stat(&zip_ar, index, &stat);
                std::vector<u8> data(stat.m_uncomp_size);
                mz_zip_reader_extract_to_mem(&zip_ar, index, data.data(), data.size(), 0);
                func(std::move(data));
            }
        }
    } else {
        for (const auto& file_name : std::filesystem::directory_iterator{cache_path}) {
            if (file_name.path().extension().string().ends_with(ext)) {
                using namespace Common::FS;
                const auto& file = IOFile{file_name, FileAccessMode::Read};
                if (file.IsOpen()) {
                    std::vector<u8> data(file.GetSize());
                    file.Read(data);
                    if (type == BlobType::ShaderBinary) {
                        InflateShaderBlob(data);
                    }
                    func(std::move(data));
                }
            }
        }
    }
}

std::vector<std::pair<std::string, std::filesystem::file_time_type>> DataBase::ListBlobs(
    BlobType type) const {
    std::vector<std::pair<std::string, std::filesystem::file_time_type>> blobs;
    if (!opened || archived) {
        return blobs;
    }
    const auto ext = "." + GetBlobFileExtension(type);
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{cache_path, ec}) {
        const auto& path = entry.path();
        if (path.extension().string() != ext) {
            continue;
        }
        std::error_code time_ec;
        const auto time = entry.last_write_time(time_ec);
        blobs.emplace_back(path.stem().string(), time_ec ? std::filesystem::file_time_type{} : time);
    }
    return blobs;
}

void DataBase::FinishPreload() {
    if (archived) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

} // namespace Storage
