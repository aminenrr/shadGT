// SPDX-FileCopyrightText: Copyright 2026 RyGT Team
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

// PRT-DRVHIT (pr-tests, Round 18, RYGT_DRIVER_CACHE_HIT_LOG, GT7 on; 0 = off): every graphics
// (library, link, whole) and compute pipeline create asks the driver, through
// VK_EXT_pipeline_creation_feedback (core in Vulkan 1.3), whether it came from the pipeline cache
// it was given. vk_pipeline_cache.cpp counts the answers per driver cache file (segment or shard)
// and logs them every 10 s and at exit.
bool PrtDrvHitOn();
void PrtDrvHitRecord(vk::PipelineCache cache, const vk::PipelineCreationFeedback& feedback);

// One create's feedback. Chain() returns a copy of the create info with the feedback struct in
// front of its pNext chain (unchanged when PRT-DRVHIT is off); Done() counts the answer. The probe
// must outlive the create call and must not be moved after Chain().
struct PrtDrvHitProbe {
    vk::PipelineCreationFeedback feedback{};
    std::array<vk::PipelineCreationFeedback, 8> stages{};
    vk::PipelineCreationFeedbackCreateInfo info{};
    vk::PipelineCache cache{};
    bool chained = false;

    vk::GraphicsPipelineCreateInfo Chain(const vk::GraphicsPipelineCreateInfo& create_info,
                                         vk::PipelineCache cache_) {
        vk::GraphicsPipelineCreateInfo out = create_info;
        if (!PrtDrvHitOn()) {
            return out;
        }
        const u32 count = create_info.stageCount <= stages.size() ? create_info.stageCount : 0u;
        info = vk::PipelineCreationFeedbackCreateInfo{
            .pNext = create_info.pNext,
            .pPipelineCreationFeedback = &feedback,
            .pipelineStageCreationFeedbackCount = count,
            .pPipelineStageCreationFeedbacks = count != 0 ? stages.data() : nullptr,
        };
        out.pNext = &info;
        cache = cache_;
        chained = true;
        return out;
    }

    vk::ComputePipelineCreateInfo Chain(const vk::ComputePipelineCreateInfo& create_info,
                                        vk::PipelineCache cache_) {
        vk::ComputePipelineCreateInfo out = create_info;
        if (!PrtDrvHitOn()) {
            return out;
        }
        info = vk::PipelineCreationFeedbackCreateInfo{
            .pNext = create_info.pNext,
            .pPipelineCreationFeedback = &feedback,
            .pipelineStageCreationFeedbackCount = 1,
            .pPipelineStageCreationFeedbacks = stages.data(),
        };
        out.pNext = &info;
        cache = cache_;
        chained = true;
        return out;
    }

    void Done(vk::Result result) const {
        if (chained && result == vk::Result::eSuccess) {
            PrtDrvHitRecord(cache, feedback);
        }
    }
};

} // namespace Vulkan
