/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Convert Line Field to Polylines
  Module:    $RCSfile: vtkFieldToLines.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkFieldToLines - converts line field to polylines
// .SECTION Description
// vtkFieldToLines is a filter that converts line fields to polylines
// defined on the nodes of a structured grid.

#ifndef __vtkFieldToLines_h
#define __vtkFieldToLines_h

#include "vtkPolyDataAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_FieldToLines_EXPORT VTK_EXPORT
#else
#include "vtkFieldToLinesConfigure.h"
#endif

class vtkDataArray;

class VTK_FieldToLines_EXPORT vtkFieldToLines : public vtkPolyDataAlgorithm
{
public:
    static vtkFieldToLines *New();
    vtkTypeRevisionMacro(vtkFieldToLines, vtkPolyDataAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetClampMacro(NodesX, int, 1, VTK_INT_MAX);
    vtkGetMacro(NodesX, int);

    vtkSetClampMacro(NodesY, int, 1, VTK_INT_MAX);
    vtkGetMacro(NodesY, int);

    vtkSetClampMacro(NodesZ, int, 1, VTK_INT_MAX);
    vtkGetMacro(NodesZ, int);

    vtkSetClampMacro(Stride, int, 1, VTK_INT_MAX);
    vtkGetMacro(Stride, int);

    vtkSetMacro(PassData, int);
    vtkGetMacro(PassData, int);

protected:
    vtkFieldToLines();
    ~vtkFieldToLines();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int NodesX;
    int NodesY;
    int NodesZ;
    int Stride;
    int PassData;

private:
    vtkFieldToLines(const vtkFieldToLines &); // Not implemented.
    void operator=(const vtkFieldToLines &); // Not implemented.
};

#endif
