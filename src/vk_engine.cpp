//> includes
#include "vk_engine.h"
#include "VkBootstrap.h"
#include "fastgltf/types.hpp"
#include "fmt/core.h"

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/packing.hpp"
#include "glm/trigonometric.hpp"
#include "glm/vector_relational.hpp"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include "vk_descriptors.h"
#include "vk_loader.h"

#include <SDL.h>
#include <SDL_video.h>
#include <SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <emmintrin.h>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vk_images.h>
#include <vk_initializers.h>
#include <vk_pipelines.h>
#include <vk_types.h>

#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

void MeshNode::Draw(const glm::mat4 &topMat, DrawContext &ctx) {
  glm::mat4 nodeMatrix = topMat * worldTransform;
  for (auto &s : mesh->surfaces) {
    RenderObject def;
    def.indexCount = s.count;
    def.firstIndex = s.startIndex;
    def.indexBuffer = mesh->buffers.indexBuffer.buffer;
    def.material = &s.material->data;

    def.transform = nodeMatrix;
    def.vertexBufferAddress = mesh->buffers.vertexBufferAddress;

    ctx.OpaqueSurfaces.push_back(def);
  }
  Node::Draw(topMat, ctx);
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine *engine) {
  VkShaderModule meshFragShader;
  if (!vkutil::load_shader_module("./shaders/mesh.frag.spv", engine->_device,
                                  &meshFragShader)) {
    fmt::println(stderr, "Error when building the mesh fragment shader module");
  } else {
    fmt::println("Mesh fragment shader succesfully loaded");
  }
  VkShaderModule meshVertShader;
  if (!vkutil::load_shader_module("./shaders/mesh.vert.spv", engine->_device,
                                  &meshVertShader)) {
    fmt::println(stderr, "Error when building the mesh fragment shader module");
  } else {
    fmt::println("Mesh fragment shader succesfully loaded");
  }

  VkPushConstantRange matrixRange{};
  matrixRange.offset = 0;
  matrixRange.size = sizeof(GPUDrawPushConstants);
  matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  DescriptorLayoutBuilder layoutBuilder;
  layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

  materialLayout =
      layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT |
                                               VK_SHADER_STAGE_FRAGMENT_BIT);

  VkDescriptorSetLayout layouts[] = {engine->_gpuSceneDataDescriptorLayout,
                                     materialLayout};

  VkPipelineLayoutCreateInfo mesh_layout_info =
      vkinit::pipeline_layout_create_info();
  mesh_layout_info.setLayoutCount = 2;
  mesh_layout_info.pSetLayouts = layouts;
  mesh_layout_info.pPushConstantRanges = &matrixRange;
  mesh_layout_info.pushConstantRangeCount = 1;

  VkPipelineLayout newLayout;
  VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr,
                                  &newLayout));

  opaquePipeline.layout = newLayout;
  transparentPipeline.layout = newLayout;

  PipelineBuilder pipelineBuilder;
  pipelineBuilder.set_shaders(meshVertShader, meshFragShader);
  pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  pipelineBuilder.set_multisampling_none();
  pipelineBuilder.disable_blending();
  pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

  pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
  pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

  pipelineBuilder._pipelineLayout = newLayout;
  opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

  pipelineBuilder.enable_blending_additive();
  pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

  transparentPipeline.pipeline =
      pipelineBuilder.build_pipeline(engine->_device);

  vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
  vkDestroyShaderModule(engine->_device, meshVertShader, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(
    VkDevice device, MaterialPass pass, const MaterialResources &resources,
    DescriptorAllocatorGrowable &descriptorAllocator) {
  MaterialInstance matData;
  matData.passType = pass;
  if (pass == MaterialPass::Transparent) {
    matData.pipeline = &transparentPipeline;
  } else {
    matData.pipeline = &opaquePipeline;
  }

  matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

  writer.clear();
  writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants),
                      resources.dataBufferOffset,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  writer.write_image(1, resources.colorImage.imageView, resources.colorSampler,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  writer.write_image(2, resources.metalRoughImage.imageView,
                     resources.metalRoughSampler,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

  writer.update_set(device, matData.materialSet);

  return matData;
}

void GLTFMetallic_Roughness::clear_resources(VkDevice device) {
  vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);

  // uncomment this once you add an actual transparentPipeline
  vkDestroyPipelineLayout(device, this->opaquePipeline.layout, nullptr);
  // vkDestroyPipelineLayout(device, this->transparentPipeline.layout, nullptr);

  vkDestroyPipeline(device, this->opaquePipeline.pipeline, nullptr);
  vkDestroyPipeline(device, this->transparentPipeline.pipeline, nullptr);
}

VulkanEngine *loadedEngine = nullptr;
constexpr bool bUseValidationLayers = true;

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init() {
  // only one engine initialization is allowed with the application.
  assert(loadedEngine == nullptr);
  loadedEngine = this;

  rnd.seed(std::mt19937{}());

  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  if (RESIZABLE) {
    window_flags = (SDL_WindowFlags)(window_flags | SDL_WINDOW_RESIZABLE);
  }

  _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                             _windowExtent.height, window_flags);

  this->mainCamera = std::make_unique<Camera>(glm::vec3(0.f), 0.f, 0.f, 0.01f);

  init_vulkan();

  init_swapchain();

  init_commands();

  init_sync_structures();

  init_descriptors();

  init_pipelines();

  init_imgui();

  init_default_data();

  // everything went fine
  _isInitialized = true;
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {

    vkDeviceWaitIdle(_device);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

      _frames[i]._deletionQueue.flush();
    }

    _mainDeletionQueue.flush();

    destroy_swapchain();

    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);

    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);

    SDL_DestroyWindow(_window);
  }

  // clear engine pointer
  loadedEngine = nullptr;
}

void VulkanEngine::init_vulkan() {
  vkb::InstanceBuilder builder;

  auto inst_ret = builder.set_app_name("sample vulkan app")
                      .request_validation_layers(bUseValidationLayers)
                      .use_default_debug_messenger()
                      .require_api_version(1, 3, 0)
                      .build();

  vkb::Instance vkb_inst = inst_ret.value();

  _instance = vkb_inst.instance;
  _debug_messenger = vkb_inst.debug_messenger;

  SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

  VkPhysicalDeviceVulkan13Features features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  features.dynamicRendering = true;
  features.synchronization2 = true;

  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
  };
  features12.bufferDeviceAddress = true;
  features12.descriptorIndexing = true;

  vkb::PhysicalDeviceSelector selector{vkb_inst};
  vkb::PhysicalDevice physicalDevice =
      selector.set_minimum_version(1, 3)
          .set_required_features_13(features)
          .set_required_features_12(features12)
          .set_surface(_surface)
          // .add_required_extension(
          //     VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME)
          // .add_required_extension_features(
          //     VK_EXT_present_mode_fifo_latest_ready)
          .select()
          .value();

  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  vkb::Device vkbDevice = deviceBuilder.build().value();

  _device = vkbDevice.device;
  _chosenGPU = physicalDevice.physical_device;

  _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  _graphicsQueueFamily =
      vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = _chosenGPU;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocatorInfo, &_allocator);

  _mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_swapchain() {
  create_swapchain(_windowExtent.width, _windowExtent.height);
  make_draw_image(false);
  // VkExtent3D drawImageExtent = {
  //     _windowExtent.width,
  //     _windowExtent.height,
  //     1,
  // };
  // _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  // _drawImage.imageExtent = drawImageExtent;
  //
  // VkImageUsageFlags drawImageUsages{};
  // drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  // drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  //
  // VkImageCreateInfo rimg_info = vkinit::image_create_info(
  //     _drawImage.imageFormat, drawImageUsages, _drawImage.imageExtent);
  //
  // VmaAllocationCreateInfo rimg_allocinfo{};
  // rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  // rimg_allocinfo.requiredFlags =
  //     VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  //
  // vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image,
  //                &_drawImage.allocation, NULL);
  //
  // VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(
  //     _drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
  //
  // VK_CHECK(
  //     vkCreateImageView(_device, &rview_info, nullptr,
  //     &_drawImage.imageView));
  //
  // _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
  // _depthImage.imageExtent = drawImageExtent;
  //
  // VkImageUsageFlags depthImageUsages{};
  // depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  //
  // VkImageCreateInfo dimg_info = vkinit::image_create_info(
  //     _depthImage.imageFormat, depthImageUsages, _depthImage.imageExtent);
  //
  // vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image,
  //                &_depthImage.allocation, NULL);
  //
  // VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
  //     _depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
  //
  // VK_CHECK(
  //     vkCreateImageView(_device, &dview_info, nullptr,
  //     &_depthImage.imageView));
  //
  // _mainDeletionQueue.push_function([&, this]() {
  //   vkDestroyImageView(_device, _depthImage.imageView, nullptr);
  //   vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
  //   vkDestroyImageView(_device, _drawImage.imageView, nullptr);
  //   vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
  // });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
  vkb::SwapchainBuilder swapchainBuilder{
      _chosenGPU,
      _device,
      _surface,
  };

  swapchainBuilder.set_desired_min_image_count(FRAME_OVERLAP);
  _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

  vkb::Swapchain vkbSwapchain =
      swapchainBuilder
          .set_desired_format(VkSurfaceFormatKHR{
              .format = _swapchainImageFormat,
              .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
          })
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(width, height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build()
          .value();

  _swapchainExtent = vkbSwapchain.extent;
  _swapchain = vkbSwapchain.swapchain;
  _swapchainImages = vkbSwapchain.get_images().value();
  _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::make_draw_image(bool isResize) {

  AllocatedImage newDrawImage;
  AllocatedImage newDepthImage;

  _drawExtent.width = _windowExtent.width * renderScale;
  _drawExtent.height = _windowExtent.height * renderScale;

  VkExtent3D drawImageExtent = {
      _drawExtent.width,
      _drawExtent.height,
      1,
  };

  newDrawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  newDrawImage.imageExtent = drawImageExtent;

  VkImageUsageFlags drawImageUsages{};
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VkImageCreateInfo rimg_info = vkinit::image_create_info(
      newDrawImage.imageFormat, drawImageUsages, newDrawImage.imageExtent);

  VmaAllocationCreateInfo rimg_allocinfo{};
  rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  rimg_allocinfo.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &newDrawImage.image,
                 &newDrawImage.allocation, NULL);

  VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(
      newDrawImage.imageFormat, newDrawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

  VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr,
                             &newDrawImage.imageView));

  newDepthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
  newDepthImage.imageExtent = drawImageExtent;

  VkImageUsageFlags depthImageUsages{};
  depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  VkImageCreateInfo dimg_info = vkinit::image_create_info(
      newDepthImage.imageFormat, depthImageUsages, newDepthImage.imageExtent);

  vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &newDepthImage.image,
                 &newDepthImage.allocation, NULL);

  VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
      newDepthImage.imageFormat, newDepthImage.image,
      VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr,
                             &newDepthImage.imageView));

  std::swap(_drawImage, newDrawImage);
  std::swap(_depthImage, newDepthImage);

  if (isResize) {
    vkDeviceWaitIdle(_device);
    vkDestroyImageView(_device, newDepthImage.imageView, nullptr);
    vmaDestroyImage(_allocator, newDepthImage.image, newDepthImage.allocation);
    vkDestroyImageView(_device, newDrawImage.imageView, nullptr);
    vmaDestroyImage(_allocator, newDrawImage.image, newDrawImage.allocation);

    VkDescriptorImageInfo imgInfo{};

    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgInfo.imageView = _drawImage.imageView;

    VkWriteDescriptorSet drawImageWrite = {};
    drawImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    drawImageWrite.pNext = nullptr;

    drawImageWrite.dstBinding = 0;
    drawImageWrite.dstSet = _drawImageDescriptors;
    drawImageWrite.descriptorCount = 1;
    drawImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    drawImageWrite.pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);
  } else {
    _mainDeletionQueue.push_function([&, this]() {
      vkDestroyImageView(_device, _depthImage.imageView, nullptr);
      vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
      vkDestroyImageView(_device, _drawImage.imageView, nullptr);
      vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
    });
  }
}

void VulkanEngine::resize_swapchain() {
  vkDeviceWaitIdle(_device);

  destroy_swapchain();

  int w, h;
  SDL_GetWindowSize(_window, &w, &h);
  _windowExtent.height = h;
  _windowExtent.width = w;

  create_swapchain(_windowExtent.width, _windowExtent.height);

  resize_requested = false;
}

void VulkanEngine::destroy_swapchain() {
  vkDestroySwapchainKHR(_device, _swapchain, nullptr);

  for (int i = 0; i < _swapchainImageViews.size(); i++) {
    vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
  }
}

void VulkanEngine::init_commands() {
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &_frames[i]._commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
                                      &_frames[i]._mainCommandBuffer));
  }

  VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                               &_immCommandPool));
  VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_immCommandPool, 1);
  VK_CHECK(
      vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

  _mainDeletionQueue.push_function(
      [=, this]() { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures() {
  VkFenceCreateInfo fenceCreateInfo =
      vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
  VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr,
                           &_frames[i]._renderFence));

    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &_frames[i]._swapchainSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &_frames[i]._renderSemaphore));
  }

  VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
  _mainDeletionQueue.push_function(
      [=, this]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::init_descriptors() {
  std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
  };

  globalDescriptorAllocator.init(_device, 10, sizes);

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _drawImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }

  _drawImageDescriptors =
      globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

  DescriptorWriter writer;
  writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE,
                     VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
  writer.update_set(_device, _drawImageDescriptors);

  for (int i = 0; i < FRAME_OVERLAP; i++) {
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
    _frames[i]._frameDescriptors.init(_device, 200, frame_sizes);

    _mainDeletionQueue.push_function(
        [&, i]() { _frames[i]._frameDescriptors.destroy_pools(_device); });
  }

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    _gpuSceneDataDescriptorLayout = builder.build(
        _device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  }

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _singleImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
  }

  _mainDeletionQueue.push_function([&]() {
    globalDescriptorAllocator.destroy_pools(_device);
    vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
  });
}

void VulkanEngine::init_pipelines() {
  init_background_pipelines();
  init_mesh_pipeline();
}

void VulkanEngine::init_mesh_pipeline() {
  VkShaderModule triangleFragShader{};

  if (!vkutil::load_shader_module("./shaders/tex_image.frag.spv", _device,
                                  &triangleFragShader)) {
    fmt::println(stderr, "Error when building the mesh fragment shader module");
  } else {
    fmt::println("Mesh fragment shader succesfully loaded");
  }

  VkShaderModule triangleVertShader;
  if (!vkutil::load_shader_module("./shaders/colored_triangle_mesh.vert.spv",
                                  _device, &triangleVertShader)) {
    fmt::println(stderr, "Error when building the mesh vertex shader module");
  } else {
    fmt::println("Mesh vertex shader succesfully loaded");
  }

  VkPushConstantRange bufferRange{};
  bufferRange.offset = 0;
  bufferRange.size = sizeof(GPUDrawPushConstants);
  bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo pipeline_layout_info =
      vkinit::pipeline_layout_create_info();
  pipeline_layout_info.pPushConstantRanges = &bufferRange;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
  pipeline_layout_info.setLayoutCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
                                  &_meshPipelineLayout));

  PipelineBuilder pb;
  pb._pipelineLayout = _meshPipelineLayout;
  pb.set_shaders(triangleVertShader, triangleFragShader);
  pb.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pb.set_polygon_mode(VK_POLYGON_MODE_FILL);
  pb.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  pb.set_multisampling_none();
  pb.disable_blending();
  // pb.enable_blending_additive();
  // pb.disable_depthtest();
  pb.enable_depthtest(VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL);

  pb.set_color_attachment_format(_drawImage.imageFormat);
  pb.set_depth_format(_depthImage.imageFormat);

  _meshPipeline = pb.build_pipeline(_device);
  vkDestroyShaderModule(_device, triangleFragShader, nullptr);
  vkDestroyShaderModule(_device, triangleVertShader, nullptr);

  metalRoughMaterial.build_pipelines(this);

  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    vkDestroyPipeline(_device, _meshPipeline, nullptr);
  });
}

void VulkanEngine::init_background_pipelines() {
  VkPipelineLayoutCreateInfo computeLayout{};
  computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  computeLayout.pNext = nullptr;
  computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
  computeLayout.setLayoutCount = 1;

  VkPushConstantRange pushConstant{};
  pushConstant.offset = 0;
  pushConstant.size = sizeof(ComputePushConstants);
  pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  computeLayout.pPushConstantRanges = &pushConstant;
  computeLayout.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr,
                                  &_gradientPipelineLayout));

  VkShaderModule gradientShader;
  if (!vkutil::load_shader_module("./shaders/gradient_color.comp.spv", _device,
                                  &gradientShader)) {
    fmt::print("Error when compiling compute shader");
  };

  VkShaderModule skyShader;
  if (!vkutil::load_shader_module("./shaders/sky.comp.spv", _device,
                                  &skyShader)) {
    fmt::print("error when building skyShader \n");
  }

  VkPipelineShaderStageCreateInfo stageInfo{};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.pNext = nullptr;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = gradientShader;
  stageInfo.pName = "main";

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.layout = _gradientPipelineLayout;
  computePipelineCreateInfo.stage = stageInfo;

  ComputeEffect gradient;
  gradient.layout = _gradientPipelineLayout;
  gradient.name = "gradient";
  gradient.data = {};
  gradient.pipeline = VK_NULL_HANDLE;

  gradient.data.color1 = glm::vec4(1, 0, 0, 1);
  gradient.data.color2 = glm::vec4(0, 0, 1, 1);

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &computePipelineCreateInfo, nullptr,
                                    &gradient.pipeline));

  computePipelineCreateInfo.stage.module = skyShader;

  ComputeEffect sky;
  sky.layout = _gradientPipelineLayout;
  sky.name = "sky";
  sky.data = {};
  sky.pipeline = VK_NULL_HANDLE;

  sky.data.color1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &computePipelineCreateInfo, nullptr,
                                    &sky.pipeline));

  backgroundEffects.push_back(sky);
  backgroundEffects.push_back(gradient);

  vkDestroyShaderModule(_device, gradientShader, nullptr);
  vkDestroyShaderModule(_device, skyShader, nullptr);

  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyPipeline(_device, sky.pipeline, nullptr);
    vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
  });
}

void VulkanEngine::init_default_data() {

  testMesh = std::move(loadGltfMeshes(this, "./assets/basicmesh.glb")).value();

  monkey_model.location.z = -5.f;
  fov_user = 70;

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  _whiteImage =
      create_image((void *)&white, VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.f));
  _greyImage =
      create_image((void *)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
  _blackImage =
      create_image((void *)&black, VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
  std::array<uint32_t, 16 * 16> pixels;
  for (int x = 0; x < 16; x++) {
    for (int y = 0; y < 16; y++) {
      pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
    }
  }
  _errorCheckerboardImage =
      create_image((void *)pixels.data(), VkExtent3D{16, 16, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  sampl.magFilter = VK_FILTER_NEAREST;
  sampl.minFilter = VK_FILTER_NEAREST;

  vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

  sampl.magFilter = VK_FILTER_LINEAR;
  sampl.magFilter = VK_FILTER_LINEAR;

  vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

  _mainDeletionQueue.push_function([&] {
    vkDestroySampler(_device, _defaultSamplerLinear, nullptr);
    vkDestroySampler(_device, _defaultSamplerNearest, nullptr);

    destroy_image(_errorCheckerboardImage);
    destroy_image(_blackImage);
    destroy_image(_greyImage);
    destroy_image(_whiteImage);

    for (auto &i : testMesh) {
      destroy_buffer(i->buffers.vertexBuffer);
      destroy_buffer(i->buffers.indexBuffer);
    }
  });

  GLTFMetallic_Roughness::MaterialResources materialResources;
  materialResources.colorImage = _whiteImage;
  materialResources.colorSampler = _defaultSamplerLinear;
  materialResources.metalRoughImage = _whiteImage;
  materialResources.metalRoughSampler = _defaultSamplerLinear;

  AllocatedBuffer materialConstants = create_buffer(
      sizeof(GLTFMetallic_Roughness::MaterialConstants),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  GLTFMetallic_Roughness::MaterialConstants *sceneUniformData =
      (GLTFMetallic_Roughness::MaterialConstants *)
          materialConstants.allocation->GetMappedData();
  sceneUniformData->colorFactors = glm::vec4{1, 1, 1, 1};
  sceneUniformData->metal_rough_factor = glm::vec4{1, 0.5, 0, 0};

  _mainDeletionQueue.push_function(
      [=, this]() { destroy_buffer(materialConstants); });

  materialResources.dataBuffer = materialConstants.buffer;
  materialResources.dataBufferOffset = 0;

  defaultData = metalRoughMaterial.write_material(
      _device, MaterialPass::MainColor, materialResources,
      globalDescriptorAllocator);

  for (auto &m : testMesh) {
    std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>();
    newNode->mesh = m;

    newNode->localTransform = glm::mat4{1.f};
    newNode->worldTransform = glm::mat4{1.f};

    for (auto &s : newNode->mesh->surfaces) {
      s.material = std::make_shared<GLTFMaterial>(defaultData);
    }

    loadedNodes[m->name] = std::move(newNode);
  }

  _mainDeletionQueue.push_function(
      [&]() { metalRoughMaterial.clear_resources(_device); });
}

void VulkanEngine::immediate_submit(
    std::function<void(VkCommandBuffer cmd)> &&function) {
  VK_CHECK(vkResetFences(_device, 1, &_immFence));
  VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

  VkCommandBuffer cmd = _immCommandBuffer;

  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  function(cmd);

  VK_CHECK(vkEndCommandBuffer(cmd));
  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

  VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::init_imgui() {

  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  VkDescriptorPool imguiPool;
  VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

  ImGui::CreateContext();

  ImGui_ImplSDL2_InitForVulkan(_window);

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = _instance;
  init_info.PhysicalDevice = _chosenGPU;
  init_info.Device = _device;
  init_info.Queue = _graphicsQueue;
  init_info.DescriptorPool = imguiPool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = true;

  init_info.PipelineRenderingCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats =
      &_swapchainImageFormat;

  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info);
  ImGui_ImplVulkan_CreateFontsTexture();

  _mainDeletionQueue.push_function([=, this]() {
    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(_device, imguiPool, nullptr);
  });
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize,
                                            VkBufferUsageFlags usage,
                                            VmaMemoryUsage memoryUsage) {
  VkBufferCreateInfo buffInfo{};
  buffInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffInfo.pNext = nullptr;
  buffInfo.size = allocSize;
  buffInfo.usage = usage;

  VmaAllocationCreateInfo vmaAllocInfo{};
  vmaAllocInfo.usage = memoryUsage;
  vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
  AllocatedBuffer newBuffer;

  VK_CHECK(vmaCreateBuffer(_allocator, &buffInfo, &vmaAllocInfo,
                           &newBuffer.buffer, &newBuffer.allocation,
                           &newBuffer.info));

  return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buff) {
  vmaDestroyBuffer(_allocator, buff.buffer, buff.allocation);
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format,
                                          VkImageUsageFlags usage,
                                          bool mipmapped) {
  AllocatedImage newImg;
  newImg.imageFormat = format;
  newImg.imageExtent = size;

  VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
  if (mipmapped) {
    img_info.mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(size.width, size.height))));
  }

  VmaAllocationCreateInfo allocinfo{};

  allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  allocinfo.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImg.image,
                          &newImg.allocation, nullptr));

  VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;

  if (format == VK_FORMAT_D32_SFLOAT) {
    aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
  }

  VkImageViewCreateInfo view_info =
      vkinit::imageview_create_info(format, newImg.image, aspectFlag);
  view_info.subresourceRange.levelCount = img_info.mipLevels;

  VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImg.imageView));

  return newImg;
}

AllocatedImage VulkanEngine::create_image(void *data, VkExtent3D size,
                                          VkFormat format,
                                          VkImageUsageFlags usage,
                                          bool mipmapped) {
  size_t data_size = size.depth * size.width * size.height * 4;
  AllocatedBuffer uploadBuffer = create_buffer(
      data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  memcpy(uploadBuffer.info.pMappedData, data, data_size);

  AllocatedImage new_img = create_image(
      size, format,
      usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      mipmapped);

  immediate_submit([&](VkCommandBuffer cmd) {
    vkutil::transition_image(cmd, new_img.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = size;

    vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, new_img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);
    vkutil::transition_image(cmd, new_img.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  });

  destroy_buffer(uploadBuffer);

  return new_img;
}

void VulkanEngine::destroy_image(const AllocatedImage &img) {
  vkDestroyImageView(_device, img.imageView, nullptr);

  vmaDestroyImage(_allocator, img.image, img.allocation);
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices,
                                        std::span<Vertex> vertices) {
  const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
  const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

  GPUMeshBuffers newSurface;
  newSurface.vertexBuffer = create_buffer(
      vertexBufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  VkBufferDeviceAddressInfo deviceAdressInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = newSurface.vertexBuffer.buffer};
  newSurface.vertexBufferAddress =
      vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

  newSurface.indexBuffer = create_buffer(indexBufferSize,
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VMA_MEMORY_USAGE_GPU_ONLY);

  AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize,
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VMA_MEMORY_USAGE_CPU_ONLY);

  void *data_buf = staging.allocation->GetMappedData();

  memcpy(data_buf, vertices.data(), vertexBufferSize);
  memcpy((char *)data_buf + vertexBufferSize, indices.data(), indexBufferSize);

  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy vertexCopy{};
    vertexCopy.dstOffset = 0;
    vertexCopy.srcOffset = 0;
    vertexCopy.size = vertexBufferSize;

    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1,
                    &vertexCopy);

    VkBufferCopy indexCopy{};
    indexCopy.dstOffset = 0;
    indexCopy.srcOffset = vertexBufferSize;
    indexCopy.size = indexBufferSize;

    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1,
                    &indexCopy);
  });

  destroy_buffer(staging);

  return newSurface;
}

void VulkanEngine::update_scene() {
  mainDrawContext.OpaqueSurfaces.clear();
  mainCamera->update();

  glm::mat4 model = glm::mat4(1.0f);
  model = glm::translate(model, monkey_model.location);
  model *= monkey_model.createYawPitchRotation();
  loadedNodes["Suzanne"]->Draw(model, mainDrawContext);

  for (int x = -3; x < 3; ++x) {
    const glm::mat4 cubeTranslation = glm::translate(
        glm::mat4{1.0f}, glm::vec3{static_cast<float>(x), 1.0f, 0.0f});
    const glm::mat4 cubeScale = glm::scale(glm::mat4{1.0f}, glm::vec3{0.2f});

    loadedNodes["Cube"]->Draw(cubeTranslation * cubeScale, mainDrawContext);
  }

  // glm::mat4 view = glm::mat4(1.f);
  // view = glm::translate(view, -view_settings.location);
  // view *= view_settings.createYawPitchRotation();
  auto view = mainCamera->getViewMatrix();
  glm::mat4 proj = glm::perspective(
      glm::radians(fov_user),
      (float)_drawExtent.width / (float)_drawExtent.height, 1000.f, 0.1f);

  proj[1][1] *= -1;

  sceneData.proj = proj;
  sceneData.view = mainCamera->getViewMatrix();
  sceneData.projview = proj * view;

  sceneData.ambientColor = glm::vec4(.1f);
  sceneData.sunLightColor = glm::vec4(1.f);
  sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);
  // sceneData.ambientColor = glm::vec4(0.7f, 0.7f, 1.f, 0.1f);
  // sceneData.sunlightDirection = glm::vec4(50.f, 100.f, 25.f, 1.f);
  // sceneData.sunLightColor = glm::vec4(1.f, 0.8f, 0.3f, 1.f);
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {

  ComputeEffect &effect = backgroundEffects[currentBackgroundEffect];
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _gradientPipelineLayout, 0, 1, &_drawImageDescriptors,
                          0, nullptr);

  vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(ComputePushConstants), &effect.data);

  vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0),
                std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {

  AllocatedBuffer gpuSceneDataBuffer =
      create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU);

  get_current_frame()._deletionQueue.push_function(
      [=, this]() { destroy_buffer(gpuSceneDataBuffer); });

  GPUSceneData *sceneUniformData =
      (GPUSceneData *)gpuSceneDataBuffer.allocation->GetMappedData();
  *sceneUniformData = sceneData;

  VkDescriptorSet globalDescriptor =
      get_current_frame()._frameDescriptors.allocate(
          _device, _gpuSceneDataDescriptorLayout);

  DescriptorWriter writer;
  writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  writer.update_set(_device, globalDescriptor);

  VkRenderingAttachmentInfo clrAttachment = vkinit::attachment_info(
      _drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingAttachmentInfo dpthAttachment = vkinit::depth_attachment_info(
      _depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo =
      vkinit::rendering_info(_drawExtent, &clrAttachment, &dpthAttachment);
  vkCmdBeginRendering(cmd, &renderInfo);
  // vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

  VkViewport viewport{};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = _drawExtent.width;
  viewport.height = _drawExtent.height;
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor;
  scissor.offset.x = 0;
  scissor.offset.y = 0;
  scissor.extent.width = _drawExtent.width;
  scissor.extent.height = _drawExtent.height;

  vkCmdSetScissor(cmd, 0, 1, &scissor);

  VkDescriptorSet imageSet = get_current_frame()._frameDescriptors.allocate(
      _device, _singleImageDescriptorLayout);

  {
    DescriptorWriter writer;
    writer.write_image(0, _errorCheckerboardImage.imageView,
                       _defaultSamplerNearest,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(_device, imageSet);
  }

  for (const RenderObject &draw : mainDrawContext.OpaqueSurfaces) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      draw.material->pipeline->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            draw.material->pipeline->layout, 0, 1,
                            &globalDescriptor, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            draw.material->pipeline->layout, 1, 1,
                            &draw.material->materialSet, 0, nullptr);
    vkCmdBindIndexBuffer(cmd, draw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    GPUDrawPushConstants pushConstants;
    pushConstants.vertexBuffer = draw.vertexBufferAddress;
    pushConstants.worldMatrix = draw.transform;

    vkCmdPushConstants(cmd, draw.material->pipeline->layout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(GPUDrawPushConstants), &pushConstants);
    vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, 0, 0);
  }

  vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd,
                              VkImageView targetImageView) {
  VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(
      targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo =
      vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);
  vkCmdBeginRendering(cmd, &renderInfo);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
}

void VulkanEngine::draw() {
  update_scene();
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           1'000'000'000));
  get_current_frame()._deletionQueue.flush();
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  get_current_frame()._deletionQueue.flush();
  get_current_frame()._frameDescriptors.clear_pools(_device);

  uint32_t swapchainImageIndex;
  auto e = vkAcquireNextImageKHR(_device, _swapchain, 1'000'000'000,
                                 get_current_frame()._swapchainSemaphore,
                                 nullptr, &swapchainImageIndex);

  if (e == VK_ERROR_OUT_OF_DATE_KHR) {
    resize_requested = true;
    return;
  } else {
    VK_CHECK(e);
  }

  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  // _drawExtent.width =
  //     std::min(_drawImage.imageExtent.width, _swapchainExtent.width) *
  //     renderScale;
  // _drawExtent.height =
  //     std::min(_drawImage.imageExtent.height, _swapchainExtent.height) *
  //     renderScale;
  _drawExtent.width = _drawImage.imageExtent.width;
  _drawExtent.height = _drawImage.imageExtent.height;

  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex],
  //                          VK_IMAGE_LAYOUT_UNDEFINED,
  //                          VK_IMAGE_LAYOUT_GENERAL);
  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_GENERAL);

  draw_background(cmd);

  // vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex],
  //                          VK_IMAGE_LAYOUT_GENERAL,
  //                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  draw_geometry(cmd);

  vkutil::transition_image(cmd, _drawImage.image,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  vkutil::copy_image_to_image(cmd, _drawImage.image,
                              _swapchainImages[swapchainImageIndex],
                              _drawExtent, _swapchainExtent);

  vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

  vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
  VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
      get_current_frame()._swapchainSemaphore);
  VkSemaphoreSubmitInfo signalInfo =
      vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                    get_current_frame()._renderSemaphore);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit,
                          get_current_frame()._renderFence));

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.pNext = nullptr;
  presentInfo.pSwapchains = &_swapchain;
  presentInfo.swapchainCount = 1;

  presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
  presentInfo.waitSemaphoreCount = 1;

  presentInfo.pImageIndices = &swapchainImageIndex;

  e = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
  if (e == VK_ERROR_OUT_OF_DATE_KHR) {
    resize_requested = true;
  } else {
    VK_CHECK(e);
  }

  _frameNumber++;
}

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT)
        bQuit = true;

      if (e.type == SDL_WINDOWEVENT) {
        if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
          stop_rendering = true;
        }
        if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
          stop_rendering = false;
        }
      }

      mainCamera->processSDLEvent(e);
      ImGui_ImplSDL2_ProcessEvent(&e);
    }

    // do not draw if we are minimized
    if (stop_rendering) {
      // throttle the speed to avoid the endless spinning
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    if (resize_requested) {
      resize_swapchain();
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // ImGui::ShowDemoWindow();
    if (ImGui::Begin("background")) {

      ComputeEffect &selected = backgroundEffects[currentBackgroundEffect];

      ImGui::Text("framerate: %6.2f", ImGui::GetIO().Framerate);

      if (ImGui::BeginCombo("Selected Effect", selected.name)) {
        for (int i = 0; i < backgroundEffects.size(); ++i) {
          if (ImGui::Selectable(backgroundEffects[i].name)) {
            currentBackgroundEffect = i;
          };
        }
        ImGui::EndCombo();
      }

      if (ImGui::CollapsingHeader("bg shader data")) {
        ImGui::SliderFloat4("data1", (float *)&selected.data.color1, 0.f, 1.f);
        ImGui::SliderFloat4("data2", (float *)&selected.data.color2, 0.f, 1.f);
      }
      ImGui::SliderFloat("Render Scale", &this->renderScale, 0.3f, 1.f);
      // if (ImGui::IsItemDeactivatedAfterEdit()) {
      if (ImGui::IsItemEdited()) {
        make_draw_image(true);
      }
      ImGui::End();
    }

    if (ImGui::Begin("foreground")) {
      if (ImGui::CollapsingHeader("Monkey Model",
                                  ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputFloat3("LM", (float *)&monkey_model.location);
        ImGui::SliderFloat3("RM", (float *)&monkey_model.rotation,
                            -glm::radians(180.f), glm::radians(180.f));
      }
      if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        // ImGui::InputFloat3("LC", (float *)&view_settings.location);
        // ImGui::SliderFloat3("RC", (float *)&view_settings.rotation,
        // -glm::radians(180.f), glm::radians(180.f));
        ImGui::Text("Speed", mainCamera->speed);
        ImGui::SliderFloat("FOV", (float *)&fov_user, 0, 180.f);
      }
      ImGui::End();
    }

    ImGui::Render();

    draw();
  }
}
