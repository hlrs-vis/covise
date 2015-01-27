/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vortex Criteria
  Module:    $RCSfile: vtkVortexCriteria.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkVortexCriteria - computes vortex criteria
// .SECTION Description
// vtkVortexCriteria is a filter that computes vortex criteria from velocity data
// defined on the nodes of an unstructured grid.

#ifndef __vtkVortexCriteria_h
#define __vtkVortexCriteria_h

#include "vtkUnstructuredGridAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_VortexCriteria_EXPORT VTK_EXPORT
#else
#include "vtkVortexCriteriaConfigure.h"
#endif

class vtkDataArray;

class VTK_VortexCriteria_EXPORT vtkVortexCriteria : public vtkUnstructuredGridAlgorithm
{
public:
    static vtkVortexCriteria *New();
    vtkTypeRevisionMacro(vtkVortexCriteria, vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetMacro(Quantity, int);
    vtkGetMacro(Quantity, int);

    vtkSetClampMacro(SmoothingRange, int, 1, VTK_INT_MAX);
    vtkGetMacro(SmoothingRange, int);

protected:
    vtkVortexCriteria();
    ~vtkVortexCriteria();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int Quantity;
    int SmoothingRange;

private:
    vtkVortexCriteria(const vtkVortexCriteria &); // Not implemented.
    void operator=(const vtkVortexCriteria &); // Not implemented.
};

#endif
