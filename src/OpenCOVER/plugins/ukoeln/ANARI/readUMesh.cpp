// std
#include <cassert>
#include <fstream>
#include <vector>
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
  return true;
}

bool UMeshReader::addFieldFromFile(const char *fileName, int index)
{
  if (!mesh)
    return false;

  std::ifstream scalarFile(fileName);
  if (!scalarFile.good())
    return false;

  size_t numScalars = 0;
  scalarFile.seekg(0,scalarFile.end);
  numScalars = scalarFile.tellg()/sizeof(float);
  scalarFile.seekg(0,scalarFile.beg);

  std::vector<float> scalars(numScalars);
  scalarFile.read((char *)scalars.data(),scalars.size()*sizeof(scalars[0]));

  fields.resize(index + 1);
  initField(fields[index], scalars.data());
  return true;
}

UnstructuredField UMeshReader::getField(int index)
{
  assert(mesh);

  if (index < fields.size()) {
    return fields[index];
  }

  fields.resize(index + 1);
  initField(fields[index]);
  return fields[index];
}

void UMeshReader::initField(UnstructuredField &field, float *scalarValues)
{
  // vertex.position
  for (size_t i = 0; i < mesh->vertices.size(); ++i) {
    const auto V = mesh->vertices[i];
    field.vertexPosition.push_back({V.x, V.y, V.z});
  }

  // vertex.data
  field.dataRange.x = FLT_MAX;
  field.dataRange.y = -FLT_MAX;

  for (size_t i = 0; i < mesh->vertices.size(); ++i) {
    float value = 0.f;
    if (scalarValues)
      value = scalarValues[i];
    else if (mesh->perVertex->values.size() > i)
      value = mesh->perVertex->values[i];
    else
      throw std::runtime_error("value not present!");
    field.vertexData.push_back(value);
    field.dataRange.x = std::min(field.dataRange.x, value);
    field.dataRange.y = std::max(field.dataRange.y, value);
  }

  // cells
  for (size_t i = 0; i < mesh->tets.size(); ++i) {
    field.cellType.push_back(10 /*VKL_TETRAHEDRON*/);
    field.cellIndex.push_back(field.index.size());
    for (int j = 0; j < mesh->tets[i].numVertices; ++j) {
      field.index.push_back((uint64_t)mesh->tets[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->pyrs.size(); ++i) {
    field.cellType.push_back(14 /*VKL_PYRAMID*/);
    field.cellIndex.push_back(field.index.size());
    for (int j = 0; j < mesh->pyrs[i].numVertices; ++j) {
      field.index.push_back((uint64_t)mesh->pyrs[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->wedges.size(); ++i) {
    field.cellType.push_back(13 /*VKL_WEDGE*/);
    field.cellIndex.push_back(field.index.size());
    for (int j = 0; j < mesh->wedges[i].numVertices; ++j) {
      field.index.push_back((uint64_t)mesh->wedges[i][j]);
    }
  }

  for (size_t i = 0; i < mesh->hexes.size(); ++i) {
    field.cellType.push_back(12 /*VKL_HEXAHEDRON*/);
    field.cellIndex.push_back(field.index.size());
    for (int j = 0; j < mesh->hexes[i].numVertices; ++j) {
      field.index.push_back((uint64_t)mesh->hexes[i][j]);
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
      field.dataRange.x = std::min(field.dataRange.x, value);
      field.dataRange.y = std::max(field.dataRange.y, value);
    }

    field.gridData.push_back(gridData);
    field.gridDomains.push_back(gridDomain);
  }
}
