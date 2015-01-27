/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vortex Core Lines according to Parallel Vectors
  Module:    $RCSfile: vtkVortexCores.h,v $

  Copyright (c) Ronald Peikert, Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkVortexCores - computes vortex core lines
// .SECTION Description
// vtkVortexCores is a filter that extracts vortex core lines of vector data
// defined on the nodes of an unstructured grid.

#ifndef __vtkVortexCores_h
#define __vtkVortexCores_h

#include "vtkPolyDataAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_VortexCores_EXPORT VTK_EXPORT
#else
#include "vtkVortexCoresConfigure.h"
#endif

class vtkDataArray;

class VTK_VortexCores_EXPORT vtkVortexCores : public vtkPolyDataAlgorithm
{
public:
    static vtkVortexCores *New();
    vtkTypeRevisionMacro(vtkVortexCores, vtkPolyDataAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetMacro(Method, int);
    vtkGetMacro(Method, int);

    vtkSetMacro(Variant, int);
    vtkGetMacro(Variant, int);

    vtkSetClampMacro(MinimumNumberOfVertices, int, 1, VTK_INT_MAX);
    vtkGetMacro(MinimumNumberOfVertices, int);

    vtkSetClampMacro(MaximumNumberOfExceptions, int, 1, VTK_INT_MAX);
    vtkGetMacro(MaximumNumberOfExceptions, int);

    vtkSetClampMacro(MinStrength, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(MinStrength, float);

    vtkSetClampMacro(MaxAngle, float, 0.0, 180.0);
    vtkGetMacro(MaxAngle, float);

protected:
    vtkVortexCores();
    ~vtkVortexCores();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int Method;
    int Variant;
    int MinimumNumberOfVertices;
    int MaximumNumberOfExceptions;
    float MinStrength;
    float MaxAngle;

private:
    vtkVortexCores(const vtkVortexCores &); // Not implemented.
    void operator=(const vtkVortexCores &); // Not implemented.
};

#endif
