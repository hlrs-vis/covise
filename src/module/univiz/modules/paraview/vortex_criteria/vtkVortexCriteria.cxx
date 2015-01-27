/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vortex Criteria
  Module:    $RCSfile: vtkVortexCriteria.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkVortexCriteria.h"

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

#include "vortex_criteria_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst_in = NULL;

vtkCxxRevisionMacro(vtkVortexCriteria, "$Revision: 0.01$");
vtkStandardNewMacro(vtkVortexCriteria);

vtkVortexCriteria::vtkVortexCriteria()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
}

vtkVortexCriteria::~vtkVortexCriteria()
{
}

int vtkVortexCriteria::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector)
{
    // get the info objects
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
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
    vtkFloatArray *inVectors = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);

    // Initialize
    vtkDebugMacro(<< "Computing vortex criteria");

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
            if (unst_in)
                delete unst_in;
            std::vector<vtkFloatArray *> vvec;
            vvec.push_back(inVectors);
            unst_in = new Unstructured(input, NULL, &vvec);
        }
#else
        std::vector<vtkFloatArray *> vvec;
        vvec.push_back(inVectors);
        unst_in = new Unstructured(input, NULL, &vvec);
#endif

#if 0 // ############### debug
    unst_in->saveAs("/pub/flowvis/sadlof/unst_debug.unst");
#endif

        // unstructured wrapper for output
        Unstructured *unst_out;
        vtkFloatArray *scalars = NULL;
        {
            // alloc output, ### TODO future: do it inside Unstructured

            // copy geometry
            output->CopyStructure(inputDS);

            // alloc data
            scalars = vtkFloatArray::New();
            vtkIdType numPts = input->GetNumberOfPoints();
            scalars->SetNumberOfComponents(1);
            scalars->SetNumberOfTuples(numPts);
            //scalars->SetName("vortex criterion");

            std::vector<vtkFloatArray *> svec;
            svec.push_back(scalars);
            unst_out = new Unstructured(input, &svec, NULL);
        }

        // compute gradient
        if (us.inputChanged("ucd", 0) || us.parameterChanged("smoothingRange"))
        {
            us.moduleStatus("computing gradient", 5);
            unst_in->gradient(0 /*###*/, false, SmoothingRange);
            us.moduleStatus("computing gradient", 50);
        }

        // compute
        char quantity_name[256] = "";
        vortex_criteria_impl(&us,
                             unst_in, 0, // ### component HACK
                             NULL, 0, // TODO
                             unst_out,
                             Quantity + 1,
                             SmoothingRange,
                             quantity_name);

        scalars->SetName(quantity_name);

        // copy existing data
        output->GetPointData()->PassData(input->GetPointData());
        output->GetCellData()->PassData(input->GetCellData());

        // add scalar output
        //output->GetPointData()->SetScalars(scalars);
        output->GetPointData()->AddArray(scalars);

        // reference is counted, we can delete
        scalars->Delete();

        // delete output wrapper
        delete unst_out;

#if !USE_CACHE
        if (unst_in)
            delete unst_in;
#endif
    }

    return 1;
}

int vtkVortexCriteria::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkVortexCriteria::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
