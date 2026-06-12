// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

#include "VkBootstrap.h"
#include "glm/ext/vector_float4.hpp"
#include "vk_descriptors.h"

constexpr unsigned int FRAME_OVERLAP = 3;

class VulkanEngine {
public:
  bool _isInitialized{false};
  int _frameNumber{0};

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

  void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

private:
  bool stop_rendering{false};
  VkExtent2D _windowExtent{1700, 900};

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkPhysicalDevice _chosenGPU;
  VkDevice _device;
  VkSurfaceKHR _surface;

  VkSwapchainKHR _swapchain;
  VkFormat _swapchainImageFormat;

  std::vector<VkImage> _swapchainImages;
  std::vector<VkImageView> _swapchainImageViews;
  VkExtent2D _swapchainExtent;
  FrameData _frames[FRAME_OVERLAP];
  FrameData &get_current_frame() {
    return _frames[_frameNumber % (FRAME_OVERLAP)];
  };

  VkQueue _graphicsQueue;
  uint32_t _graphicsQueueFamily;

  struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  VmaAllocator _allocator;
  DeletionQueue _mainDeletionQueue;

  AllocatedImage _drawImage;
  VkExtent2D _drawExtent;

  VkDescriptorSet _drawImageDescriptors;
  VkDescriptorSetLayout _drawImageDescriptorLayout;

  VkPipeline _gradientPipeline;
  VkPipelineLayout _gradientPipelineLayout;

  VkPipeline _trianglePipeline;
  VkPipelineLayout _trianglePipelineLayout;

  VkFence _immFence;
  VkCommandBuffer _immCommandBuffer;
  VkCommandPool _immCommandPool;

  DescriptorAllocator globalDescriptorAllocator;

  std::vector<ComputeEffect> backgroundEffects;
  int currentBackgroundEffect{0};

  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();

  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();
  void init_triangle_pipeline(); // graphics pipeline
  void init_imgui();

  void create_swapchain(uint32_t width, uint32_t height);
  void destroy_swapchain();

  AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                VmaMemoryUsage memoryUsage);
  void destroy_buffer(const AllocatedBuffer &buff);

  GPUMeshBuffers uploadMesh(std::span<uint32_t> indices,
                            std::span<Vertex> vertices);

  void draw_background(VkCommandBuffer cmd);
  void draw_geometry(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
