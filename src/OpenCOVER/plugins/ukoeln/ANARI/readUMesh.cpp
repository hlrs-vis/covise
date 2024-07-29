// std
#include <cassert>
// umesh
#include "umesh/UMesh.h"
// ours
#include "readUMesh.h"

UMeshReader::~UMeshReader() {}

bool UMeshReader::open(const char *fileName)
{
  std::cout << "#mm: loading umesh from " << fileName << std::endl;
  mesh = umesh::UMesh::loadFrom(fileName);
  if (!mesh)
    return false;
  std::cout << "#mm: got umesh w/ " << mesh->toString() << std::endl;
  fields.resize(1);
  return true;
}

UnstructuredField UMeshReader::getField(int index)
{
  assert(mesh);
  assert(index == 0);

  if (fields.empty()) {
    fields.resize(index + 1);
  }

  // vertex.position
  for (size_t i = 0; i < mesh->vertices.size(); ++i) {
    const auto V = mesh->vertices[i];
    fields[index].vertexPosition.push_back({V.x, V.y, V.z});
  }

  // vertex.data
  fields[index].dataRange.x = FLT_MAX;
  fields[index].dataRange.y = -FLT_MAX;

  for (size_t i = 0; i < mesh->vertices.size(); ++i) {
    assert(!mesh->perVertex->values.empty());
    float value = mesh->perVertex->values[i];
    fields[index].vertexData.push_back(value);
    fields[index].dataRange.x = std::min(fields[index].dataRange.x, value);
    fields[index].dataRange.y = std::max(fields[index].dataRange.y, value);
  }

  // cells
  for (size_t i = 0; i < mesh->tets.size(); ++i) {
    fields[index].cellType.push_back(10 /*VKL_TETRAHEDRON*/);
    fields[index].cellIndex.push_back(fields[index].index.size());
    for (int j = 0; j < mesh->tets[i].numVertices; ++j) {
      fields[index].index.push_back((uint64_t)mesh->tets[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->pyrs.size(); ++i) {
    fields[index].cellType.push_back(14 /*VKL_PYRAMID*/);
    fields[index].cellIndex.push_back(fields[index].index.size());
    for (int j = 0; j < mesh->pyrs[i].numVertices; ++j) {
      fields[index].index.push_back((uint64_t)mesh->pyrs[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->wedges.size(); ++i) {
    fields[index].cellType.push_back(13 /*VKL_WEDGE*/);
    fields[index].cellIndex.push_back(fields[index].index.size());
    for (int j = 0; j < mesh->wedges[i].numVertices; ++j) {
      fields[index].index.push_back((uint64_t)mesh->wedges[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->hexes.size(); ++i) {
    fields[index].cellType.push_back(12 /*VKL_HEXAHEDRON*/);
    fields[index].cellIndex.push_back(fields[index].index.size());
    for (int j = 0; j < mesh->hexes[i].numVertices; ++j) {
      fields[index].index.push_back((uint64_t)mesh->hexes[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->grids.size(); ++i) {
    const umesh::Grid &grid = mesh->grids[i];

    UnstructuredField::GridDomain gridDomain;
    gridDomain[0] = grid.domain.lower.x;
    gridDomain[1] = grid.domain.lower.y;
    gridDomain[2] = grid.domain.lower.z;
    gridDomain[3] = grid.domain.upper.x;
    gridDomain[4] = grid.domain.upper.y;
    gridDomain[5] = grid.domain.upper.z;

    size_t numScalars =
        (grid.numCells.x + 1) * size_t(grid.numCells.y + 1) * (grid.numCells.z + 1);

    UnstructuredField::GridData gridData;
    for (int d = 0; d < 3; ++d) {
      gridData.dims[d] = grid.numCells[d] + 1;
    }

    gridData.values.resize(numScalars);
    for (size_t s = 0; s < numScalars; ++s) {
      float value = mesh->gridScalars[grid.scalarsOffset + s];
      gridData.values[s] = value;
      fields[index].dataRange.x = std::min(fields[index].dataRange.x, value);
      fields[index].dataRange.y = std::max(fields[index].dataRange.y, value);
    }

    fields[index].gridData.push_back(gridData);
    fields[index].gridDomains.push_back(gridDomain);
  }

  return fields[index];
}
