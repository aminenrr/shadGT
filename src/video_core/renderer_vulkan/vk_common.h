// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Include vulkan-hpp header
#define VK_ENABLE_BETA_EXTENSIONS
#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_SETTERS
#define VULKAN_HPP_HAS_SPACESHIP_OPERATOR
#define VULKAN_HPP_NO_EXCEPTIONS
// Define assert-on-result to nothing to instead return the result for our handling.
#define VULKAN_HPP_ASSERT_ON_RESULT

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#include <vulkan/vulkan.hpp>
#pragma clang diagnostic pop

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

VK_DEFINE_HANDLE(VmaAllocator)
namespace Vulkan {
vk::DeviceSize TrimEmptyVmaBlocks(VmaAllocator allocator);
vk::DeviceSize EmptyVmaBytesFreed();
// PRT-VMAFRAG (pr-tests, Round 22): the small image pools' blocks (Tiny/Small/Medium images), and
// their empty blocks freed. Default-pool blocks: TrimEmptyVmaBlocks.
struct VmaPoolFrag {
    const char* name = "";
    vk::DeviceSize block_size = 0;
    vk::DeviceSize block_bytes = 0;
    vk::DeviceSize allocation_bytes = 0;
    uint32_t blocks = 0;
    uint32_t allocations = 0;
};
void SmallImagePoolFrag(VmaAllocator allocator, void (*each)(const VmaPoolFrag&, void*), void* user);
vk::DeviceSize TrimEmptySmallImagePoolBlocks(VmaAllocator allocator);
}
