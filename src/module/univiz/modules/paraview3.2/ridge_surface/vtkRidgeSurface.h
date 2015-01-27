/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Ridge Surface
  Module:    $RCSfile: vtkRidgeSurface.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkRidgeSurface - extract ridge surfaces
// .SECTION Description
// vtkRidgeSurface is a filter that extracts ridge surfaces from scalar data
// defined on the nodes of an unstructured grid.

#ifndef __vtkRidgeSurface_h
#define __vtkRidgeSurface_h

#include "vtkPolyDataAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_RidgeSurface_EXPORT VTK_EXPORT
#else
#include "vtkRidgeSurfaceConfigure.h"
#endif

class vtkDataArray;

class VTK_RidgeSurface_EXPORT vtkRidgeSurface : public vtkPolyDataAlgorithm
{
public:
    static vtkRidgeSurface *New();
    vtkTypeRevisionMacro(vtkRidgeSurface, vtkPolyDataAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetClampMacro(SmoothingRange, int, 1, VTK_INT_MAX);
    vtkGetMacro(SmoothingRange, int);

    vtkSetMacro(Mode, int);
    vtkGetMacro(Mode, int);

    vtkSetMacro(Extremum, int);
    vtkGetMacro(Extremum, int);

    vtkSetMacro(ExcludeFLT_MAX, int);
    vtkGetMacro(ExcludeFLT_MAX, int);

    vtkSetMacro(ExcludeLonelyNodes, int);
    vtkGetMacro(ExcludeLonelyNodes, int);

    vtkSetClampMacro(HessExtrEigenvalMin, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(HessExtrEigenvalMin, float);

    vtkSetClampMacro(PCAsubdomMaxPerc, float, 0.0, 1.0);
    vtkGetMacro(PCAsubdomMaxPerc, float);

    vtkSetClampMacro(ScalarMin, float, -VTK_FLOAT_MAX, VTK_FLOAT_MAX);
    vtkGetMacro(ScalarMin, float);

    vtkSetClampMacro(ScalarMax, float, -VTK_FLOAT_MAX, VTK_FLOAT_MAX);
    vtkGetMacro(ScalarMax, float);

    vtkSetClampMacro(ClipScalarMin, float, -VTK_FLOAT_MAX, VTK_FLOAT_MAX);
    vtkGetMacro(ClipScalarMin, float);

    vtkSetClampMacro(ClipScalarMax, float, -VTK_FLOAT_MAX, VTK_FLOAT_MAX);
    vtkGetMacro(ClipScalarMax, float);

    vtkSetClampMacro(MinSize, int, 1, VTK_INT_MAX);
    vtkGetMacro(MinSize, int);

    vtkSetMacro(FilterByCell, int);
    vtkGetMacro(FilterByCell, int);

    vtkSetMacro(CombineExceptions, int);
    vtkGetMacro(CombineExceptions, int);

    vtkSetClampMacro(MaxExceptions, int, 1, 3);
    vtkGetMacro(MaxExceptions, int);

    vtkSetMacro(GenerateNormals, int);
    vtkGetMacro(GenerateNormals, int);

protected:
    vtkRidgeSurface();
    ~vtkRidgeSurface();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int SmoothingRange;
    int Mode;
    int Extremum;
    int ExcludeFLT_MAX;
    int ExcludeLonelyNodes;
    float HessExtrEigenvalMin;
    float PCAsubdomMaxPerc;
    float ScalarMin;
    float ScalarMax;
    float ClipScalarMin;
    float ClipScalarMax;
    int MinSize;
    int FilterByCell;
    int CombineExceptions;
    int MaxExceptions;
    int GenerateNormals;

private:
    vtkRidgeSurface(const vtkRidgeSurface &); // Not implemented.
    void operator=(const vtkRidgeSurface &); // Not implemented.
};

#endif
