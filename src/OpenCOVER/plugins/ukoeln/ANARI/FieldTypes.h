#pragma once

// std
#include <array>
#include <vector>

// Structured field type //////////////////////////////////////////////////////
struct StructuredField
{
  std::vector<uint8_t> dataUI8;
  std::vector<uint16_t> dataUI16;
  std::vector<float> dataF32;
  int dimX{0};
  int dimY{0};
  int dimZ{0};
  unsigned bytesPerCell{0};
  struct
  {
    float x, y;
  } dataRange;

  bool empty() const
  {
    if (bytesPerCell == 1 && dataUI8.empty())
      return true;
    if (bytesPerCell == 2 && dataUI16.empty())
      return true;
    if (bytesPerCell == 4 && dataF32.empty())
      return true;
    return false;
  }
};

// AMR field type /////////////////////////////////////////////////////////////
typedef std::array<int, 6> BlockBounds;
struct BlockData
{
  int dims[3];
  std::vector<float> values;
};
struct AMRField
{
  std::vector<float> cellWidth;
  std::vector<int> blockLevel;
  std::vector<BlockBounds> blockBounds;
  std::vector<BlockData> blockData;
  struct
  {
    float x, y;
  } voxelRange;
};

// Unstructured field type ////////////////////////////////////////////////////
struct UnstructuredField
{
  struct vec3f
  {
    float x, y, z;
  };
  std::vector<vec3f> vertexPosition;
  std::vector<float> vertexData;
  std::vector<uint64_t> index;
  bool indexPrefixed{false};
  std::vector<uint64_t> cellIndex;
  // std::vector<float> cellData;
  std::vector<uint8_t> cellType;
  struct
  {
    float x, y;
  } dataRange;

  // unstructured meshes can optionally store
  // vertex-centered grids
  typedef std::array<float, 6> GridDomain;
  struct GridData
  {
    int dims[3];
    std::vector<float> values;
  };
  std::vector<GridDomain> gridDomains;
  std::vector<GridData> gridData;
};
