/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   LocalizedFlow
  Module:    $RCSfile: vtkLocalizedFlow.h,v $

  Copyright (c) Alexander Wiebel, BSV - University of Leipzig
            and Christoph Garth, IDAV - UC Davis
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkLocalizedFlow - computes localized flow and related fields
// .SECTION Description
// vtkLocalizedFlow is a filter that computes localized flow and related
// fields from velocity data defined on the nodes of an unstructured grid.

#ifndef __vtkLocalizedFlow_h
#define __vtkLocalizedFlow_h

#include "vtkUnstructuredGridAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_LocalizedFlow_EXPORT VTK_EXPORT
#else
#include "vtkLocalizedFlowConfigure.h"
#endif

class vtkDataArray;

class VTK_LocalizedFlow_EXPORT vtkLocalizedFlow : public vtkUnstructuredGridAlgorithm
{
public:
    static vtkLocalizedFlow *New();
    vtkTypeRevisionMacro(vtkLocalizedFlow, vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetMacro(MaxIter, int);
    vtkGetMacro(MaxIter, int);

    vtkSetMacro(Residual, double);
    vtkGetMacro(Residual, double);

    //   vtkSetClampMacro(SmoothingRange, int, 1, VTK_INT_MAX);
    //   vtkGetMacro(SmoothingRange, int);

protected:
    vtkLocalizedFlow();
    ~vtkLocalizedFlow();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int MaxIter;
    double Residual;

private:
    vtkLocalizedFlow(const vtkLocalizedFlow &); // Not implemented.
    void operator=(const vtkLocalizedFlow &); // Not implemented.
};

#endif
