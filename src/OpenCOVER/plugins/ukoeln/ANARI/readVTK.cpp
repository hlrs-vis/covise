#include <cassert>
#include <vtkCellIterator.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
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

VTKReader::~VTKReader()
{
  if (reader)
    reader->Delete();
}

bool VTKReader::open(const char *fileName)
{
  reader = vtkUnstructuredGridReader::New();
  reader->SetFileName(fileName);

  if (!reader->IsFileUnstructuredGrid())
    return false;

  reader->Update();

  ugrid = reader->GetOutput();
  //ugrid->Print(cout);

  int numFields = reader->GetNumberOfScalarsInFile();
  fieldNames.resize(numFields);
  fields.resize(numFields);

  std::cout << "Variables found:\n";
  for (int i=0;i<numFields;++i) {
    fieldNames[i] = std::string(reader->GetScalarsNameInFile(i));
    std::cout << reader->GetScalarsNameInFile(i) << '\n';
  }

  return ugrid != nullptr;
}

UnstructuredField VTKReader::getField(int index, bool indexPrefixed)
{
  int numFields = fields.size();

  for (int f=0;f<numFields;++f) {
    std::cout << "Reading field \"" << fieldNames[0] << "\"\n";

    fields[f].indexPrefixed = indexPrefixed;

    vtkIdType numPoints = ugrid->GetNumberOfPoints();

    // vertex.position
    vtkPoints *points = ugrid->GetPoints();

    for (vtkIdType i=0;i<numPoints;++i) {
      double pt[3];
      points->GetPoint(i, pt);
      fields[f].vertexPosition.push_back({(float)pt[0],(float)pt[1],(float)pt[2]});
    }

    // vertex.data
    vtkDataArray *data = ugrid->GetPointData()->GetArray("data");

    fields[f].dataRange.x = FLT_MAX;
    fields[f].dataRange.y = -FLT_MAX;

    for (vtkIdType i=0;i<numPoints;++i) {
      float value = data->GetTuple1(i);
      fields[f].vertexData.push_back(value);
      fields[f].dataRange.x = std::min(fields[f].dataRange.x, value);
      fields[f].dataRange.y = std::max(fields[f].dataRange.y, value);
    }

    // cells
    vtkCellIterator *iter = ugrid->NewCellIterator();

    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextCell()) {
      vtkIdList *pointIDs = iter->GetPointIds();
      vtkIdType type = pointIDs->GetNumberOfIds();
      assert(type >= 4 && type <= 8);

      fields[f].cellIndex.push_back(fields[f].index.size());
      if (indexPrefixed) {
        fields[f].index.push_back((uint64_t)type);
        for (vtkIdType id=0;id<type;++id) {
          fields[f].index.push_back((uint64_t)pointIDs->GetId(id));
        }
      } else {
        fields[f].cellType.push_back(toTypeEnum(type));
        for (vtkIdType id=0;id<type;++id) {
          fields[f].index.push_back((uint64_t)pointIDs->GetId(id));
        }
      }
    }

    iter->Delete();
  }

  assert(index < numFields);
  return fields[index];
}
