#pragma once

#include <cstdint>
#include <vector>

struct UnstructuredField {
  struct vec3f { float x,y,z; };
  std::vector<vec3f> vertexPosition;
  std::vector<float> vertexData;
  std::vector<uint64_t> index;
  bool indexPrefixed{false};
  std::vector<uint64_t> cellIndex;
  //std::vector<float> cellData;
  std::vector<uint8_t> cellType;
  struct { float x, y; } dataRange;
};

#ifdef HAVE_VTK
class vtkUnstructuredGrid;
class vtkUnstructuredGridReader;
struct VTKReader {
 ~VTKReader();

  bool open(const char *fileName);
  UnstructuredField getField(int index, bool indexPrefixed = false);

  std::vector<std::string> fieldNames;
  std::vector<UnstructuredField> fields;
  vtkUnstructuredGrid *ugrid{nullptr};
  vtkUnstructuredGridReader *reader{nullptr};
};
#endif
