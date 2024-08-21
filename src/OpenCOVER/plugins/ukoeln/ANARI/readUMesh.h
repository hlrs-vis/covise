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

  bool open(const char *fileName, int slotID=0);
  // the following assumes the topology is shared between fields:
  bool addFieldFromFile(const char *fileName, int index, int slotID=0);
  UnstructuredField getField(int index);

 private:
  // fields _after_ umesh/scalar files were combined
  // and the slot assignment was resolved:
  std::vector<UnstructuredField> fields;

  typedef std::shared_ptr<umesh::UMesh> Mesh;
  // one mesh per slot:
  std::vector<Mesh> meshes;

  typedef std::vector<float> Scalars;
  // zero or one scalar lists per slot (outer dim) per field/variable (inner dim)
  std::vector<std::vector<Scalars>> scalars;

  void initField(int index);
};
#endif
