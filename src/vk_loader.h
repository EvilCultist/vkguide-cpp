#pragma once
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <vk_types.h>

struct GLTFMaterial {
  MaterialInstance data;
};

struct GeoSurface {
  uint32_t startIndex;
  uint32_t count;
  std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
  std::string name;
  std::vector<GeoSurface> surfaces;
  GPUMeshBuffers buffers;
};

class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>>
loadGltfMeshes(VulkanEngine *engine, std::filesystem::path filePath);
