/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Flow Topology
  Module:    $RCSfile: vtkFlowTopo.h,v $

  Copyright (c) Ronald Peikert, Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkFlowTopo - computes critical points
// .SECTION Description
// vtkFlowTopo is a filter that extracts critical points of vector data
// defined on the nodes of an unstructured grid.

#ifndef __vtkFlowTopo_h
#define __vtkFlowTopo_h

#include "vtkPolyDataAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_FlowTopo_EXPORT VTK_EXPORT
#else
#include "vtkFlowTopoConfigure.h"
#endif

class vtkDataArray;

class VTK_FlowTopo_EXPORT vtkFlowTopo : public vtkPolyDataAlgorithm
{
public:
    static vtkFlowTopo *New();
    vtkTypeRevisionMacro(vtkFlowTopo, vtkPolyDataAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetMacro(DivideByWallDist, int);
    vtkGetMacro(DivideByWallDist, int);

    vtkSetMacro(InteriorCritPts, int);
    vtkGetMacro(InteriorCritPts, int);

    vtkSetMacro(BoundaryCritPts, int);
    vtkGetMacro(BoundaryCritPts, int);

    vtkSetMacro(GenerateSeeds, int);
    vtkGetMacro(GenerateSeeds, int);

    vtkSetClampMacro(SeedsPerCircle, int, 1, VTK_INT_MAX);
    vtkGetMacro(SeedsPerCircle, int);

    vtkSetClampMacro(Radius, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(Radius, float);

    vtkSetClampMacro(Offset, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(Offset, float);

    vtkSetClampMacro(GlyphRadius, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(GlyphRadius, float);

protected:
    vtkFlowTopo();
    ~vtkFlowTopo();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    int DivideByWallDist;
    int InteriorCritPts;
    int BoundaryCritPts;
    int GenerateSeeds;
    int SeedsPerCircle;
    float Radius;
    float Offset;
    float GlyphRadius;

private:
    vtkFlowTopo(const vtkFlowTopo &); // Not implemented.
    void operator=(const vtkFlowTopo &); // Not implemented.
};

#endif
