#pragma once
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

namespace vkutil {

bool load_shader_module(const char *filePath, VkDevice device,
                        VkShaderModule *outShaderModule);

};

class PipelineBuilder {
public:
  std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;

  VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineColorBlendAttachmentState _colorBlendAttachment;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineLayout _pipelineLayout;
  VkPipelineDepthStencilStateCreateInfo _depthStencil;
  VkPipelineRenderingCreateInfo _renderInfo;
  VkFormat _colorAttachmentFormat;

  PipelineBuilder() { clear(); }

  void clear();

  VkPipeline build_pipeline(VkDevice device);

  void set_shaders(VkShaderModule vs, VkShaderModule fs);

  void set_input_topology(VkPrimitiveTopology topology);
};
