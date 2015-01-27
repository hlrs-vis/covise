/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vortex Core Lines according to Parallel Vectors
  Module:    $RCSfile: vtkVortexCores.cxx,v $

  Copyright (c) Ronald Peikert, Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkVortexCores.h"

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
#include "unstructured.h"
#include "unigeom.h"
#include "unisys.h"

#include "paraview_ext.h"
#include "computeVortexCores.h"
#include "ucd_vortex_cores_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst = NULL;

vtkCxxRevisionMacro(vtkVortexCores, "$Revision: 0.11 $");
vtkStandardNewMacro(vtkVortexCores);

vtkVortexCores::vtkVortexCores()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
}

vtkVortexCores::~vtkVortexCores()
{
}

int vtkVortexCores::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector)
{
    // get the info objects
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the input and ouptut
    vtkUnstructuredGrid *input = vtkUnstructuredGrid::SafeDownCast(
        inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input)
    {
        return 1;
    }

    vtkIdType numCells = input->GetNumberOfCells();

    // #### Float HACK TODO
    vtkFloatArray *inVectors = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);

    // Initialize
    vtkDebugMacro(<< "Computing vortex cores");

    // Check input
    if (numCells < 1)
    {
        printf("no cells\n");
        vtkErrorMacro("No cells");
        return 1;
    }

    if (!inVectors)
    {
        printf("no vector data\n");
        vtkDebugMacro(<< "No vector data");
        return 1;
    }

    // compute
    {
        // system wrapper
        UniSys us = UniSys(this);

// create unstructured wrapper for input
#if USE_CACHE
        if (us.inputChanged("ucd", 0))
        {
            if (unst)
                delete unst;
            std::vector<vtkFloatArray *> vvec;
            vvec.push_back(inVectors);
            unst = new Unstructured(input, NULL, &vvec);
        }
#else
        std::vector<vtkFloatArray *> vvec;
        vvec.push_back(inVectors);
        unst = new Unstructured(input, NULL, &vvec);
#endif

        // setup geometry wrapper for output
        UniGeom ugeom = UniGeom(output);

        // compute
        ucd_vortex_cores_impl(&us,
                              unst, 0,
                              Method + 1,
                              Variant + 1,
                              MinimumNumberOfVertices,
                              MaximumNumberOfExceptions,
                              MinStrength,
                              MaxAngle,
                              &ugeom);

        // pass point data (interpolate from input)
        passInterpolatePointData(input, output);
    }

#if !USE_CACHE
    if (unst)
        delete unst;
#endif

    return 1;
}

int vtkVortexCores::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkVortexCores::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
