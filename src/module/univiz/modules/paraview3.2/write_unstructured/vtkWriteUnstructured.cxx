/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Write Unstructured Grid
  Module:    $RCSfile: vtkWriteUnstructured.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkWriteUnstructured.h"

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

#include "paraview_ext.h"
#include "write_unstructured_impl.cpp" // ### including .cpp

#define USE_CACHE 0

vtkCxxRevisionMacro(vtkWriteUnstructured, "$Revision: 0.01$");
vtkStandardNewMacro(vtkWriteUnstructured);

//----------------------------------------------------------------------------
vtkWriteUnstructured::vtkWriteUnstructured()
{
    this->FileName = NULL;
}

//----------------------------------------------------------------------------
vtkWriteUnstructured::~vtkWriteUnstructured()
{
    this->SetFileName(0);
}

//----------------------------------------------------------------------------
void vtkWriteUnstructured::Write()
{
    // ### HACK: actually writing inside RequestData()
    //this->WriteToStream(0);
    //printf("should write\n");
}

//----------------------------------------------------------------------------
int vtkWriteUnstructured::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *vtkNotUsed(outputVector))
{
    // system wrapper
    UniSys us = UniSys(this);

    //this->SetErrorCode(vtkErrorCode::NoError);

    int len = inputVector[0]->GetNumberOfInformationObjects();

    if (len > 1)
    {
        printf("attention: writing only last information object\n");
    }

    for (int cc = 0; cc < len; cc++)
    {

        std::vector<vtkFloatArray *> svec;
        std::vector<vtkFloatArray *> vvec;

        vtkInformation *inInfo = inputVector[0]->GetInformationObject(cc);
        vtkUnstructuredGrid *input = vtkUnstructuredGrid::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));

        if (!input)
        {
            continue;
        }

        for (int i = 0; i < input->GetPointData()->GetNumberOfArrays(); i++)
        {

            // #### Float HACK TODO
            vtkFloatArray *inArray = (vtkFloatArray *)input->GetPointData()->GetArray(i);

            switch (input->GetPointData()->GetArray(i)->GetNumberOfComponents())
            {
            case 1: // scalar
                svec.push_back(inArray);
                break;
            case 3: // vector
                vvec.push_back(inArray);
                break;
            default: // ignored
                break;
            }
        }

        Unstructured *unst;
        if (svec.size() > 0 && vvec.size() == 0)
        {
            unst = new Unstructured(input, &svec, NULL);
        }
        else if (svec.size() == 0 && vvec.size() > 0)
        {
            unst = new Unstructured(input, NULL, &vvec);
        }
        else if (svec.size() > 0 && vvec.size() > 0)
        {
            unst = new Unstructured(input, &svec, &vvec);
        }

        // ### TODO: own file name for each set (actually overwritten by last)
        write_unstructured_impl(&us, unst, FileName);
    }

    return 1;
}

//----------------------------------------------------------------------------
int vtkWriteUnstructured::FillInputPortInformation(int port, vtkInformation *info)
{
    if (!this->Superclass::FillInputPortInformation(port, info))
    {
        return 0;
    }
    info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
    return 1;
}
//----------------------------------------------------------------------------
void vtkWriteUnstructured::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);

    os << indent << "File Name: "
       << (this->FileName ? this->FileName : "(none)") << "\n";
}
