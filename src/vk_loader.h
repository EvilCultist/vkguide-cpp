#pragma once
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <vk_types.h>

struct GeoSurface {
  uint32_t startIndex;
  uint32_t count;
};

struct MeshAsset {
  std::string name;
  std::vector<GeoSurface> surfaces;
  GPUMeshBuffers buffers;
};

class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>>
loadGltfMeshes(VulkanEngine *engine, std::filesystem::path filePath);
