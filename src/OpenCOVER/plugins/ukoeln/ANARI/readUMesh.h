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
  // the following assumes the topology is shared between fields:
  bool addFieldFromFile(const char *fileName, int index);
  UnstructuredField getField(int index);

  std::vector<UnstructuredField> fields;
  std::shared_ptr<umesh::UMesh> mesh{nullptr};
 private:
  void initField(UnstructuredField &field, float *scalarValues=nullptr);
};
#endif
