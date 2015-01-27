/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Write unstructured grid in Unstructured format
  Module:    $RCSfile: vtkWriteUnstructured.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkWriteUnstructured - writes unstructured grid
// .SECTION Description
// vtkWriteUnstructured is a writer that writes an unstructured grid to a file

#ifndef __vtkWriteUnstructured_h
#define __vtkWriteUnstructured_h

#include "vtkUnstructuredGridAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_WriteUnstructured_EXPORT VTK_EXPORT
#else
#include "vtkWriteUnstructuredConfigure.h"
#endif

class vtkDataArray;

class VTK_WriteUnstructured_EXPORT vtkWriteUnstructured : public vtkUnstructuredGridAlgorithm
{
public:
    static vtkWriteUnstructured *New();
    vtkTypeRevisionMacro(vtkWriteUnstructured, vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    // Description:
    // Get/Set the name of the output file.
    vtkSetStringMacro(FileName);
    vtkGetStringMacro(FileName);

    // Description:
    // Write data
    void Write();

protected:
    vtkWriteUnstructured();
    ~vtkWriteUnstructured();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    char *FileName;

private:
    vtkWriteUnstructured(const vtkWriteUnstructured &); // Not implemented.
    void operator=(const vtkWriteUnstructured &); // Not implemented.
};

#endif
