/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Convert Line Field to Polylines
  Module:    $RCSfile: vtkFieldToLines.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkFieldToLines.h"

#include "vtkCell.h"
#include "vtkFloatArray.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPolyData.h"

#include "linalg.h"
#include "unifield.h"
#include "unigeom.h"
#include "unisys.h"

#include "paraview_ext.h"
#include "field_to_lines_impl.cpp" // ### including .cpp

#define USE_CACHE 0

vtkCxxRevisionMacro(vtkFieldToLines, "$Revision: 0.01$");
vtkStandardNewMacro(vtkFieldToLines);

vtkFieldToLines::vtkFieldToLines()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
}

vtkFieldToLines::~vtkFieldToLines()
{
}

int vtkFieldToLines::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector)
{
    // get the info objects
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the input and ouptut
    vtkStructuredGrid *input = vtkStructuredGrid::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input)
    {
        return 1;
    }

    // #### Float HACK TODO
    vtkFloatArray *inVectors = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);

    // Initialize
    vtkDebugMacro(<< "Converting line field to lines");

    // Check input
    if (!inVectors)
    {
        printf("no vector data\n");
        vtkDebugMacro(<< "No vector data");
        return 1;
    }

    // system wrapper
    UniSys us = UniSys(this);

    // create UniField wrapper for input
    UniField *unif = new UniField(input);

    // setup geometry wrapper for output
    UniGeom ugeom = UniGeom(output);

    // compute
    std::vector<int> usedNodes;
    std::vector<int> usedNodesVertCnt;
    if (!field_to_lines_impl(&us,
                             unif,
                             NodesX,
                             NodesY,
                             NodesZ,
                             Stride,
                             &ugeom,
                             &usedNodes,
                             &usedNodesVertCnt))
    {
        delete unif;
        return 0;
    }

    // pass point data
    if (PassData)
    {
        passLineFieldVertexData(input,
                                &usedNodes,
                                &usedNodesVertCnt,
                                output);
    }

    if (unif)
        delete unif;

    return 1;
}

int vtkFieldToLines::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkFieldToLines::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
