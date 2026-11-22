// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include "common/game_title.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/sampler.h"

namespace VideoCore {

Sampler::Sampler(const Vulkan::Instance& instance, const AmdGpu::Sampler& sampler,
                 const AmdGpu::BorderColorBuffer border_color_base, const bool is_depth) {
    using namespace Vulkan;
    const bool anisotropy_enable = instance.IsAnisotropicFilteringSupported() &&
                                   (AmdGpu::IsAnisoFilter(sampler.xy_mag_filter) ||
                                    AmdGpu::IsAnisoFilter(sampler.xy_min_filter));
    float max_anisotropy =
        anisotropy_enable ? std::clamp(sampler.MaxAniso(), 1.0f, instance.MaxSamplerAnisotropy())
                          : 1.0f;
    bool force_aniso = false;
    {
        // PRT-AF (pr-tests, RYGT_FORCE_ANISO, 16 on GT7; 0 = off): trilinear colour samplers the
        // game left without anisotropic filtering get it, so road and ground textures stay sharp at
        // a grazing angle. Point, bilinear-only, depth and already anisotropic samplers are left alone.
        static const float forced = [] {
            const char* env = std::getenv("RYGT_FORCE_ANISO");
            const float value = env && *env ? static_cast<float>(std::strtod(env, nullptr)) : 16.0f;
            const bool gt7 = Common::CurrentGameTitle() == Common::GameTitle::Gt7;
            LOG_WARNING(Render_Vulkan, "PRT-AF: forced anisotropic filtering {}x{}", value,
                        gt7 ? "" : " (not GT7: off)");
            return gt7 ? value : 0.0f;
        }();
        if (forced > 1.0f && !anisotropy_enable && !is_depth &&
            instance.IsAnisotropicFilteringSupported() &&
            sampler.xy_min_filter == AmdGpu::Filter::Bilinear &&
            sampler.xy_mag_filter == AmdGpu::Filter::Bilinear &&
            sampler.mip_filter == AmdGpu::MipFilter::Linear) {
            force_aniso = true;
            max_anisotropy = std::min(forced, instance.MaxSamplerAnisotropy());
        }
    }
    auto border_color = LiverpoolToVK::BorderColor(sampler.border_color_type);
    if (border_color == vk::BorderColor::eFloatCustomEXT &&
        !instance.IsCustomBorderColorSupported()) {
        LOG_WARNING(Render_Vulkan, "Custom border color is not supported, falling back to black");
        border_color = vk::BorderColor::eFloatOpaqueBlack;
    }

    const auto custom_color = [&]() -> std::optional<vk::SamplerCustomBorderColorCreateInfoEXT> {
        if (border_color == vk::BorderColor::eFloatCustomEXT) {
            const auto border_color_index = sampler.border_color_ptr.Value();
            const auto border_color_buffer = border_color_base.Address<std::array<float, 4>*>();
            const auto custom_border_color_array = border_color_buffer[border_color_index];

            const vk::SamplerCustomBorderColorCreateInfoEXT ret{
                .customBorderColor =
                    vk::ClearColorValue{
                        .float32 = custom_border_color_array,
                    },
                .format = vk::Format::eR32G32B32A32Sfloat,
            };
            return ret;
        } else {
            return std::nullopt;
        }
    }();

    const vk::SamplerCreateInfo sampler_ci = {
        .pNext = custom_color ? &*custom_color : nullptr,
        .magFilter = LiverpoolToVK::Filter(sampler.xy_mag_filter),
        .minFilter = LiverpoolToVK::Filter(sampler.xy_min_filter),
        .mipmapMode = LiverpoolToVK::MipFilter(sampler.mip_filter),
        .addressModeU = LiverpoolToVK::ClampMode(sampler.clamp_x),
        .addressModeV = LiverpoolToVK::ClampMode(sampler.clamp_y),
        .addressModeW = LiverpoolToVK::ClampMode(sampler.clamp_z),
        .mipLodBias = std::min(sampler.LodBias(), instance.MaxSamplerLodBias()),
        .anisotropyEnable = anisotropy_enable || force_aniso,
        .maxAnisotropy = max_anisotropy,
        // GCN compares per instruction; a plain read of a compare sampler is undefined in Vulkan.
        .compareEnable = is_depth,
        .compareOp = LiverpoolToVK::DepthCompare(sampler.depth_compare_func),
        .minLod = sampler.MinLod(),
        // Vulkan requires an ordered interval even when a guest sampler supplies reversed bounds.
        .maxLod = std::max(sampler.MinLod(), sampler.MaxLod()),
        .borderColor = border_color,
        .unnormalizedCoordinates = false, // Handled in shader due to Vulkan limitations.
    };
    auto [sampler_result, smplr] = instance.GetDevice().createSamplerUnique(sampler_ci);
    ASSERT_MSG(sampler_result == vk::Result::eSuccess, "Failed to create sampler: {}",
               vk::to_string(sampler_result));
    handle = std::move(smplr);
}

Sampler::~Sampler() = default;

} // namespace VideoCore
