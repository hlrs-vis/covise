#pragma once

// std
#include <cstdint>
#include <string>
#include <vector>
// ours
#include "FieldTypes.h"

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
