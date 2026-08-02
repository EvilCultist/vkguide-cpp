
#include "fastgltf/types.hpp"
#include "fastgltf/util.hpp"
#include "fmt/core.h"
#include "fmt/os.h"
#include "iostream"
#include "stb_image.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <vk_loader.h>

#include "glm/gtx/quaternion.hpp"
#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"

#include "fastgltf/core.hpp"
#include "fastgltf/glm_element_traits.hpp"
#include "fastgltf/tools.hpp"

std::optional<std::vector<std::shared_ptr<MeshAsset>>>
loadGltfMeshes(VulkanEngine *engine, std::filesystem::path filePath) {
  fmt::println("loading GLTF at : {}", filePath.string());

  if (!std::filesystem::exists(filePath)) {
    fmt::println("File does not exist: {}", filePath.string());
    return {};
  }

  if (!std::filesystem::is_regular_file(filePath)) {
    fmt::println("Not a regular file: {}", filePath.string());
    return {};
  }

  fastgltf::GltfDataBuffer data;
  // data.FromPath(filePath); 😭😭😭
  data = *fastgltf::GltfDataBuffer::FromPath(filePath);

  constexpr auto opts = fastgltf::Options::LoadExternalBuffers;

  fastgltf::Asset gltf;
  fastgltf::Parser parser{};

  auto load = parser.loadGltfBinary(data, filePath.parent_path(), opts);

  if (load) {
    gltf = std::move(load.get());
  } else {
    fmt::println(stderr, "failed to load GLTF\t:\t{}",
                 fastgltf::to_underlying(load.error()));
    return {};
  }

  std::vector<std::shared_ptr<MeshAsset>> meshes;

  std::vector<uint32_t> indices;
  std::vector<Vertex> vertices;

  for (fastgltf::Mesh &mesh : gltf.meshes) {
    MeshAsset newMesh;
    newMesh.name = mesh.name;

    indices.clear();
    vertices.clear();

    for (auto &&p : mesh.primitives) {
      GeoSurface newSurface;

      newSurface.startIndex = (uint32_t)indices.size();
      newSurface.count =
          (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

      size_t initial_vtx = vertices.size();

      // load indices
      {
        fastgltf::Accessor &indicesAccessor =
            gltf.accessors[p.indicesAccessor.value()];
        indices.reserve(indices.size() + indicesAccessor.count);
        fastgltf::iterateAccessor<uint32_t>(
            gltf, indicesAccessor,
            [&](uint32_t idx) { indices.push_back(idx + initial_vtx); });
      }

      // load vertex positions
      {
        fastgltf::Accessor &posAccessor =
            gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
        vertices.resize(vertices.size() + posAccessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            gltf, posAccessor, [&](glm::vec3 v, size_t index) {
              Vertex newvtx;
              newvtx.position = v;
              newvtx.normal = {1, 0, 0};
              newvtx.color = glm::vec4{1.f};
              newvtx.uv_x = 0;
              newvtx.uv_y = 0;
              vertices[initial_vtx + index] = newvtx;
            });
      }

      // load vertex normals
      auto normals = p.findAttribute("NORMAL");
      if (normals != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            gltf, gltf.accessors[(*normals).accessorIndex],
            [&](glm::vec3 v, size_t index) {
              vertices[initial_vtx + index].normal = v;
            });
      }

      // load UVs
      auto uv = p.findAttribute("TEXCOORD_0");
      if (uv != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec2>(
            gltf, gltf.accessors[(*uv).accessorIndex],
            [&](glm::vec2 v, size_t index) {
              vertices[initial_vtx + index].uv_x = v.x;
              vertices[initial_vtx + index].uv_y = v.y;
            });
      }

      // load vertex colors
      auto colors = p.findAttribute("COLOR_0");
      if (colors != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec4>(
            gltf, gltf.accessors[(*colors).accessorIndex],
            [&](glm::vec4 v, size_t index) {
              vertices[initial_vtx + index].color = v;
            });
      }
      newMesh.surfaces.push_back(newSurface);
    }

    constexpr bool OverrideColors = false;
    if (OverrideColors) {
      for (Vertex &vtx : vertices) {
        vtx.color = glm::vec4(vtx.normal, 1.f);
      }
    }

    newMesh.buffers = engine->uploadMesh(indices, vertices);
    meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
  }

  return meshes;
}
