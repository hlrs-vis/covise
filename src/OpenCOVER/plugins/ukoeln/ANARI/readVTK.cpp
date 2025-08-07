#include <cassert>
#include <vtkCellIterator.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include "readVTK.h"

// uint8_t VKL_TETRAHEDRON = 10;
// uint8_t VKL_HEXAHEDRON = 12;
// uint8_t VKL_WEDGE = 13;
// uint8_t VKL_PYRAMID = 14;
inline uint8_t toTypeEnum(vtkIdType id) { // for the moment, as in ospray..
  if (id == 4) // tet
    return 10;
  else if (id == 5) // pyr
    return 12;
  else if (id == 6) // wedge
    return 13;
  else if (id == 8) // hex
    return 14;
  else
    return uint8_t(-1);
}

template <typename Reader>
void initFields(const Reader &reader,
                std::vector<std::string> &fieldNames,
                std::vector<UnstructuredField> &fields)
{
}

VTKReader::~VTKReader()
{
}

bool VTKReader::open(const char *fn)
{
  std::string fileName(fn);
  std::string extension = "";
  if (fileName.find_last_of(".") != std::string::npos)
  {
    extension = fileName.substr(fileName.find_last_of("."));
  }

  if (extension == ".vtu") {
    readerXML = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    readerXML->SetFileName(fn);
    readerXML->Update();
    ugrid = readerXML->GetOutput();
  } else if (extension == ".vtk") {
    reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(fn);
    reader->Update();
    ugrid = reader->GetOutput();
  }
  // ugrid->Print(cout);

  return ugrid != nullptr;
}

UnstructuredField VTKReader::getField()
{
  vtkIdType numPoints = ugrid->GetNumberOfPoints();
  vtkIdType numCells = ugrid->GetNumberOfCells();

  // vertex.position
  for (vtkIdType i=0;i<numPoints; ++i) {
    double *pt = ugrid->GetPoint(i);
    field.vertexPosition.push_back({(float)pt[0],(float)pt[1],(float)pt[2]});
  }

  // connectivity
  for (vtkIdType i=0; i<numCells; ++i) {
    vtkCell *cell = ugrid->GetCell(i);
    int n = cell->GetNumberOfPoints();
    if (n < 4 || n > 8) {
      std::cerr << "Unsupported cell type with " << n << " points\n";
      continue;
    }

    field.cellIndex.push_back((uint32_t)field.index.size());
    field.cellType.push_back(toTypeEnum(n));
    for (vtkIdType j=0;j<n;++j) {
      field.index.push_back((uint32_t)cell->GetPointId(j));
    }
  }

  auto copyFloatArray = [](std::vector<float> &dest,
                           float &minValue, float &maxValue,
                           vtkDataArray *array, vtkIdType count) {
    int numComponents = array->GetNumberOfComponents();
    if (numComponents > 1) {
      std::cerr << "readVTK: only single-component arrays supported"
                << " using only first one!\n";
    }
    dest.resize(count);
    for (vtkIdType i=0; i<count; ++i) {
      float f = static_cast<float>(array->GetComponent(i, 0));
      dest[i] = f;
      minValue = std::min(minValue, f);
      maxValue = std::max(maxValue, f);
    }
  };

  // vertex.data
  vtkPointData *pointData = ugrid->GetPointData();
  uint32_t numPointArrays = pointData->GetNumberOfArrays();

  field.vertexData.resize(std::min(1u, numPointArrays));

  for (uint32_t i=0; i<std::min(1u, numPointArrays); ++i) {
    field.vertexData[i].range.x = FLT_MAX;
    field.vertexData[i].range.y = -FLT_MAX;

    vtkDataArray *array = pointData->GetArray(i);
    copyFloatArray(field.vertexData[i].array,
                   field.vertexData[i].range.x,
                   field.vertexData[i].range.y,
                   array,
                   numPoints);
  }

  // cell.data
  vtkCellData *cellData = ugrid->GetCellData();
  uint32_t numCellArrays = cellData->GetNumberOfArrays();

  field.cellData.resize(std::min(1u, numCellArrays));

  for (uint32_t i=0; i<std::min(1u, numCellArrays); ++i) {
    field.cellData[i].range.x = FLT_MAX;
    field.cellData[i].range.y = -FLT_MAX;

    vtkDataArray *array = cellData->GetArray(i);
    copyFloatArray(field.cellData[i].array,
                   field.cellData[i].range.x,
                   field.cellData[i].range.y,
                   array,
                   numCells);
  }

  return field;
}
