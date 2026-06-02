#include "vk_types.h"
#include <cstddef>
#include <cstdint>
#include <vk_descriptors.h>
#include <vulkan/vulkan_core.h>

void DescriptorLayputBuilder::add_binding(uint32_t binding,
                                          VkDescriptorType type) {
  VkDescriptorSetLayoutBinding newbind{};

  newbind.binding = binding;
  newbind.descriptorCount = 1;
  newbind.descriptorType = type;

  Bindings.push_back(newbind);
}

void DescriptorLayputBuilder::clear() { Bindings.clear(); }

VkDescriptorSetLayout
DescriptorLayputBuilder::build(VkDevice device, VkShaderStageFlags shaderStages,
                               void *pNext,
                               VkDescriptorSetLayoutCreateFlags flags) {
  for (auto &b : Bindings) {
    b.stageFlags |= shaderStages;
  }

  VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
  };
  info.pNext = pNext;
  info.pBindings = Bindings.data();
  info.bindingCount = (uint32_t)Bindings.size();
  info.flags = flags;

  VkDescriptorSetLayout set;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

  return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets,
                                    std::span<PoolSizeRatio> poolRatios) {
  std::vector<VkDescriptorPoolSize> poolSizes;
  for (PoolSizeRatio ratio : poolRatios) {
    poolSizes.push_back(VkDescriptorPoolSize{
        .type = ratio.type,
        .descriptorCount = uint32_t(ratio.ratio * maxSets),
    });
  }

  VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
  };
  pool_info.flags = 0;
  pool_info.maxSets = maxSets;
  pool_info.poolSizeCount = (uint32_t)poolSizes.size();
  pool_info.pPoolSizes = poolSizes.data();

  vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device) {
  vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device) {
  vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device,
                                              VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
  };
  allocInfo.pNext = nullptr;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;

  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

  return ds;
}
