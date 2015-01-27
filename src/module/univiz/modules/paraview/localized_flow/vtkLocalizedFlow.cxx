/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Localized Flow
  Module:    $RCSfile: vtkLocalizedFlow.cxx,v $

  Copyright (c) Alexander Wiebel, BSV - University of Leipzig
            and Christoph Garth, IDAV - UC Davis
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkLocalizedFlow.h"

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

#include "localized_flow_impl.cpp" // ### including .cpp

static Unstructured *unst_in = NULL;

vtkCxxRevisionMacro(vtkLocalizedFlow, "$Revision: 0.01$");
vtkStandardNewMacro(vtkLocalizedFlow);

vtkLocalizedFlow::vtkLocalizedFlow()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
}

vtkLocalizedFlow::~vtkLocalizedFlow()
{
}

int vtkLocalizedFlow::RequestData(
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
    vtkDebugMacro(<< "Computing localized flow");

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

        std::vector<vtkFloatArray *> vvec;
        vvec.push_back(inVectors);
        unst_in = new Unstructured(input, NULL, &vvec);

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
            //scalars->SetName("BLA");

            std::vector<vtkFloatArray *> svec;
            svec.push_back(scalars);
            unst_out = new Unstructured(input, &svec, NULL);
        }

        // compute
        char quantity_name[256] = "Potential (Localized Flow)";
        localized_flow_impl(&us,
                            unst_in, 0, // ### component HACK
                            unst_out,
                            Residual,
                            MaxIter,
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

        if (unst_in)
            delete unst_in;
    }

    return 1;
}

int vtkLocalizedFlow::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkLocalizedFlow::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
