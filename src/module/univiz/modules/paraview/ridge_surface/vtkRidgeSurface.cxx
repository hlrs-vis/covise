/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vortex Core Lines according to Parallel Vectors
  Module:    $RCSfile: vtkRidgeSurface.cxx,v $

  Copyright (c) Ronald Peikert, Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkRidgeSurface.h"

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
#include "ridge_surface_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst = NULL;
static Unstructured *temp = NULL;
static bool *excludeNodes = NULL;

vtkCxxRevisionMacro(vtkRidgeSurface, "$Revision: 0.01$");
vtkStandardNewMacro(vtkRidgeSurface);

vtkRidgeSurface::vtkRidgeSurface()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::SCALARS);
}

vtkRidgeSurface::~vtkRidgeSurface()
{
}

int vtkRidgeSurface::RequestData(
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
    vtkFloatArray *inScalars = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);

    // Initialize
    vtkDebugMacro(<< "Extracting ridge surfaces");

    // Check input
    if (numCells < 1)
    {
        printf("no cells\n");
        vtkErrorMacro("No cells");
        return 1;
    }

    if (!inScalars)
    {
        printf("no scalar data\n");
        vtkDebugMacro(<< "No scalar data");
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
            std::vector<vtkFloatArray *> svec;
            svec.push_back(inScalars);
            unst = new Unstructured(input, &svec, NULL);
        }
#else
        std::vector<vtkFloatArray *> svec;
        svec.push_back(inScalars);
        unst = new Unstructured(input, &svec, NULL);
#endif

        // setup geometry wrapper for output
        UniGeom ugeom = UniGeom(output);

        // compute
        if (!ridge_surface_impl(&us,
                                unst,
                                0, // compScalar
                                -1, // compClipScalar TODO
                                &temp,
                                &excludeNodes,
                                //*level,
                                0.0,
                                SmoothingRange,
                                Mode + 1,
                                Extremum + 1,
                                UseBisection,
                                ExcludeFLT_MAX,
                                ExcludeLonelyNodes,
                                HessExtrEigenvalMin,
                                PCAsubdomMaxPerc,
                                ScalarMin,
                                ScalarMax,
                                ClipScalarMin,
                                ClipScalarMax,
                                MinSize,
                                FilterByCell,
                                CombineExceptions,
                                MaxExceptions,
                                -FLT_MAX, // min x
                                FLT_MAX, // max x
                                -FLT_MAX, // min y
                                FLT_MAX, // max y
                                -FLT_MAX, // min z
                                FLT_MAX, // max z
                                //clip_lower_data,
                                0,
                                //clip_higher_data,
                                0,
                                GenerateNormals,
                                &ugeom))
        {
            return 0;
        }

        // pass point data (interpolate from input)
        passInterpolatePointData(input, output);
    }

#if !USE_CACHE
    if (temp)
    {
        delete temp;
        temp = NULL;
    }
    if (unst)
    {
        delete unst;
        unst = NULL;
    }
    if (excludeNodes)
    {
        delete excludeNodes;
        excludeNodes = NULL;
    }
#endif

    return 1;
}

int vtkRidgeSurface::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkRidgeSurface::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
