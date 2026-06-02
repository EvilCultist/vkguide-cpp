#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

struct DescriptorLayputBuilder {
  std::vector<VkDescriptorSetLayoutBinding> Bindings;

  void add_binding(uint32_t binding, VkDescriptorType type);
  void clear();

  VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages,
                              void *pnext = nullptr,
                              VkDescriptorSetLayoutCreateFlags flags = 0);
};

struct DescriptorAllocator {

  struct PoolSizeRatio {
    VkDescriptorType type;
    float ratio;
  };

  VkDescriptorPool pool;

  void init_pool(VkDevice device, uint32_t maxSets,
                 std::span<PoolSizeRatio> poolRatios);
  void clear_descriptors(VkDevice device);
  void destroy_pool(VkDevice device);

  VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};
