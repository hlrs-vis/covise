/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Statistics
  Module:    $RCSfile: vtkStatistics.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkStatistics.h"

// TODO: cleanup, some are not used
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
#include "unisys.h"

// ##### for test
#include "vtkCellArray.h"
#include "vtkPolyData.h"

#ifdef WIN32
#define rint(a) floor(a + 0.5)
#endif

#include <algorithm>
using namespace vtkstd;
#include "statistics_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst_in = NULL;

vtkCxxRevisionMacro(vtkStatistics, "$Revision: 0.01$");
vtkStandardNewMacro(vtkStatistics);

vtkStatistics::vtkStatistics()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::SCALARS);
}

vtkStatistics::~vtkStatistics()
{
}

int vtkStatistics::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputScalar,
    vtkInformationVector *outputVector)
{
    // get the info objects
    vtkInformation *inInfo = inputScalar[0]->GetInformationObject(0);
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the input and ouptut
    vtkUnstructuredGrid *input = vtkUnstructuredGrid::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataSet *inputDS = vtkDataSet::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataSet *output = vtkDataSet::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input)
    {
        return 1;
    }

    vtkIdType numCells = input->GetNumberOfCells();

    // #### Float HACK TODO
    vtkFloatArray *inScalars = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputScalar);

    // Initialize
    vtkDebugMacro(<< "Computing statistics");

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
            if (unst_in)
                delete unst_in;
            std::vector<vtkFloatArray *> vvec;
            vvec.push_back(inVectors);
            unst_in = new Unstructured(input, NULL, &vvec);
        }
#else
        std::vector<vtkFloatArray *> svec;
        svec.push_back(inScalars);
        unst_in = new Unstructured(input, &svec, NULL);
#endif

        // compute
        statistics_impl(&us, unst_in, 0);

// copy existing data
//output->GetPointData()->PassData(input->GetPointData());
//output->GetCellData()->PassData(input->GetCellData());

#if !USE_CACHE
        if (unst_in)
            delete unst_in;
#endif
    }

    return 1;
}

int vtkStatistics::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkStatistics::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
