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
  bool addFieldFromFile(const char *fileName, int index, int slotID=0, int timeStep=0);
  UnstructuredField getField(int index, int timeStep=0);

  int numTimeSteps() const
  { return scalars.empty() ? 1 : (int)scalars.size(); } // TODO: should check if mesh has perVertex!

  bool haveTimeStep(int timeStep) const
  { return numTimeSteps() > timeStep && !scalars[timeStep].empty(); }

 private:
  // fields _after_ umesh/scalar files were combined
  // and the slot assignment was resolved, per time step (outer),
  // per field variable (inner):
  std::vector<std::vector<UnstructuredField>> fields;

  typedef std::shared_ptr<umesh::UMesh> Mesh;
  // one mesh per slot:
  std::vector<Mesh> meshes;

  typedef std::vector<float> Scalars;
  // zero or one scalar lists per time step (outer dim)
  // per slot (nested dim) per field/variable (inner dim)
  std::vector<std::vector<std::vector<Scalars>>> scalars;

  void initField(int index, int timeStep=0);
};
#endif
