#pragma once

// std
#include <vector>
// ours
#include "FieldTypes.h"

#ifdef HAVE_UMESH
namespace umesh {
class UMesh;
} // namespace umesh

struct UMeshReader
{
  ~UMeshReader();

  bool open(const char *fileName);
  UnstructuredField getField(int index);

  std::vector<UnstructuredField> fields;
  std::shared_ptr<umesh::UMesh> mesh{nullptr};
};
#endif
