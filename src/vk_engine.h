// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

// #include "VkBootstrap.h"
#include "camera.h"
#include "fastgltf/types.hpp"
#include "glm/detail/qualifier.hpp"
#include "vk_descriptors.h"
#include "vk_loader.h"

constexpr unsigned int FRAME_OVERLAP = 3;
constexpr bool RESIZABLE = false;

struct DrawContext {
  std::vector<RenderObject> OpaqueSurfaces;
};

struct MeshNode : public Node {
  std::shared_ptr<MeshAsset> mesh;
  virtual void Draw(const glm::mat4 &topMat, DrawContext &ctx) override;
};

struct GLTFMetallic_Roughness {
  MaterialPipeline opaquePipeline;
  MaterialPipeline transparentPipeline;

  VkDescriptorSetLayout materialLayout;

  struct MaterialConstants {
    glm::vec4 colorFactors;
    glm::vec4 metal_rough_factor;
    // padding
    glm::vec4 extra[14];
  };

  struct MaterialResources {
    AllocatedImage colorImage;
    VkSampler colorSampler;
    AllocatedImage metalRoughImage;
    VkSampler metalRoughSampler;
    VkBuffer dataBuffer;
    uint32_t dataBufferOffset;
  };

  DescriptorWriter writer;

  void build_pipelines(VulkanEngine *engine);
  void clear_resources(VkDevice device);

  MaterialInstance
  write_material(VkDevice device, MaterialPass pass,
                 const MaterialResources &resources,
                 DescriptorAllocatorGrowable &descriptorAllocator);
};

class VulkanEngine {
public:
  bool _isInitialized{false};
  int _frameNumber{0};
  VkDevice _device;

  GPUSceneData sceneData;
  VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

  AllocatedImage _drawImage;
  AllocatedImage _depthImage;
  VkExtent2D _drawExtent;
  float renderScale = 1.f;

  VmaAllocator _allocator;
  DeletionQueue _mainDeletionQueue;

  MaterialInstance defaultData;
  GLTFMetallic_Roughness metalRoughMaterial;

  DrawContext mainDrawContext;
  std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

  void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

  GPUMeshBuffers uploadMesh(std::span<uint32_t> indices,
                            std::span<Vertex> vertices);

private:
  std::mt19937 rnd;

  bool show_hud = true;

  bool stop_rendering{false};
  VkExtent2D _windowExtent{1700, 900};

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkPhysicalDevice _chosenGPU;
  VkSurfaceKHR _surface;

  VkSwapchainKHR _swapchain;
  VkFormat _swapchainImageFormat;

  std::vector<VkImage> _swapchainImages;
  std::vector<VkImageView> _swapchainImageViews;
  VkExtent2D _swapchainExtent;
  bool resize_requested = false;

  FrameData _frames[FRAME_OVERLAP];
  FrameData &get_current_frame() {
    return _frames[_frameNumber % (FRAME_OVERLAP)];
  };

  VkQueue _graphicsQueue;
  uint32_t _graphicsQueueFamily;

  struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  VkDescriptorSet _drawImageDescriptors;
  VkDescriptorSetLayout _drawImageDescriptorLayout;

  VkPipeline _gradientPipeline;
  VkPipelineLayout _gradientPipelineLayout;

  VkPipeline _meshPipeline;
  VkPipelineLayout _meshPipelineLayout;

  Pose monkey_model{};
  std::unique_ptr<Camera> mainCamera;
  float fov_user = 0;

  VkFence _immFence;
  VkCommandBuffer _immCommandBuffer;
  VkCommandPool _immCommandPool;

  DescriptorAllocatorGrowable globalDescriptorAllocator;

  // temp for monkey with texture
  VkDescriptorSetLayout _singleImageDescriptorLayout;

  AllocatedImage _whiteImage;
  AllocatedImage _blackImage;
  AllocatedImage _greyImage;
  AllocatedImage _errorCheckerboardImage;

  VkSampler _defaultSamplerLinear;
  VkSampler _defaultSamplerNearest;

  std::vector<ComputeEffect> backgroundEffects;
  int currentBackgroundEffect{0};

  std::vector<std::shared_ptr<MeshAsset>> testMesh;

  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();
  void init_default_data();

  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();
  void init_mesh_pipeline();
  void init_imgui();

  void create_swapchain(uint32_t width, uint32_t height);
  void resize_swapchain();
  void destroy_swapchain();

  AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                VmaMemoryUsage memoryUsage);
  void destroy_buffer(const AllocatedBuffer &buff);

  AllocatedImage create_image(VkExtent3D size, VkFormat format,
                              VkImageUsageFlags usage, bool mipmapped = false);
  AllocatedImage create_image(void *data, VkExtent3D size, VkFormat format,
                              VkImageUsageFlags usage, bool mipmapped = false);
  void destroy_image(const AllocatedImage &img);

  void make_draw_image(bool isResize);

  void update_scene();

  void draw_background(VkCommandBuffer cmd);
  void draw_geometry(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
