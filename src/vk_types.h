// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include "glm/detail/qualifier.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/gtx/quaternion.hpp"
#include "vk_descriptors.h"
#include <cstdint>
#include <deque>
#include <functional>

#include <vk_mem_alloc.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      fmt::println(stderr, "Detected Vulkan error: {}", string_VkResult(err)); \
      abort();                                                                 \
    }                                                                          \
  } while (0)

struct ComputePushConstants {
  glm::vec4 color1;
  glm::vec4 color2;
  glm::vec4 data3;
  glm::vec4 data4;
};

struct Pose {
  glm::vec3 location;
  glm::vec3 rotation;

  glm::mat4 createYawPitchRotation() {
    return glm::rotate(glm::rotate(glm::rotate(glm::mat4(1.f), rotation.z,
                                               glm::vec3(0.f, 0.f, 1.f)),
                                   rotation.y, glm::vec3(0.f, 1.f, 0.f)),
                       rotation.x, glm::vec3(1.f, 0.f, 0.f));
  }
};

struct ComputeEffect {
  const char *name;

  VkPipeline pipeline;
  VkPipelineLayout layout;

  ComputePushConstants data;
};

struct DeletionQueue {
  std::deque<std::function<void()>> deletors;

  void push_function(std::function<void()> &&function) {
    deletors.push_back(function);
  }

  void flush() {
    for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
      (*it)();
    }

    deletors.clear();
  }
};

struct FrameData {
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;

  VkSemaphore _swapchainSemaphore, _renderSemaphore;
  VkFence _renderFence;
  DeletionQueue _deletionQueue;

  DescriptorAllocatorGrowable _frameDescriptors;
};

struct AllocatedImage {
  VkImage image;
  VkImageView imageView;
  VmaAllocation allocation;
  VkExtent3D imageExtent;
  VkFormat imageFormat;
};

struct AllocatedBuffer {
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo info;
};

struct Vertex {

  glm::vec3 position;
  float uv_x;
  glm::vec3 normal;
  float uv_y;
  glm::vec4 color;
};

struct GPUMeshBuffers {

  AllocatedBuffer indexBuffer;
  AllocatedBuffer vertexBuffer;
  VkDeviceAddress vertexBufferAddress;
};

struct GPUSceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 projview;
  glm::vec3 ambientColor;
  glm::vec3 sunlightDirection;
  glm::vec3 sunLightColor;
};

struct GPUDrawPushConstants {
  glm::mat4 worldMatrix;
  VkDeviceAddress vertexBuffer;
};

enum class MaterialPass : uint8_t {
  MainColor,
  Transparent,
  Other,
};

struct MaterialPipeline {
  VkPipeline pipeline;
  VkPipelineLayout layout;
};

struct MaterialInstance {
  MaterialPipeline *pipeline;
  VkDescriptorSet materialSet;
  MaterialPass passType;
};

struct RenderObject {
  uint32_t indexCount;
  uint32_t firstIndex;

  VkBuffer indexBuffer;
  MaterialInstance *material;

  glm::mat4 transform;
  VkDeviceAddress vertexBufferAddress;
};
