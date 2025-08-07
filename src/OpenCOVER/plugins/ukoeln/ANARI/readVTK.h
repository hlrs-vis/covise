#pragma once

// std
#include <cstdint>
#include <string>
#include <vector>
// vtk
#ifdef HAVE_VTK
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#endif
// ours
#include "FieldTypes.h"

#ifdef HAVE_VTK
struct VTKReader {
 ~VTKReader();

  bool open(const char *fileName);
  UnstructuredField getField();

  UnstructuredField field;

  vtkSmartPointer<vtkUnstructuredGridReader> reader;
  vtkSmartPointer<vtkXMLUnstructuredGridReader> readerXML;
  vtkUnstructuredGrid *ugrid{nullptr};
};
#endif
