/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Finite Lyapunov Exponents
  Module:    $RCSfile: vtkFLE.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich, 2007
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkFLE.h"

// TODO: cleanup, some are not used
#include "vtkCell.h"
#include "vtkFloatArray.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkStructuredGrid.h"

#include "linalg.h"
#include "unstructured.h"
#include "unifield.h"
#include "unisys.h"
#include "paraview_ext.h"
#include "util.h"

#include "FLE_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst_in = NULL;

vtkCxxRevisionMacro(vtkFLE, "$Revision: 0.01$");
vtkStandardNewMacro(vtkFLE);

vtkFLE::vtkFLE()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
    this->VelocityFile = NULL;
}

vtkFLE::~vtkFLE()
{
    if (this->VelocityFile)
        delete[] this->VelocityFile;
    this->VelocityFile = NULL;
}

int vtkFLE::RequestData(
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

    //vtkUnstructuredGrid *outputUG =
    //vtkUnstructuredGrid::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
    //vtkDataSet *outputUGDS =
    //vtkDataSet::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkMultiBlockDataSet *output = vtkMultiBlockDataSet::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input)
    {
        return 1;
    }

    int UG_Id = 0;
    {
        vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::New();
        output->SetDataSet(UG_Id, 0, ugrid);
        ugrid->Delete();
    }
    vtkUnstructuredGrid *outputUG = vtkUnstructuredGrid::SafeDownCast(output->GetDataSet(UG_Id, 0));
    vtkDataSet *outputUGDS = vtkDataSet::SafeDownCast(output->GetDataSet(UG_Id, 0));

    int SG_Id = 1;
    {
        vtkStructuredGrid *sgrid = vtkStructuredGrid::New();
        output->SetDataSet(SG_Id, 0, sgrid);
        sgrid->Delete();
    }
    vtkStructuredGrid *outputSG = vtkStructuredGrid::SafeDownCast(output->GetDataSet(SG_Id, 0));
    vtkDataSet *outputSGDS = vtkDataSet::SafeDownCast(output->GetDataSet(SG_Id, 0));

    vtkIdType numCells = input->GetNumberOfCells();

    // #### Float HACK TODO
    vtkFloatArray *inVectors = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);

    // Initialize
    vtkDebugMacro(<< "Computing finite Lyapunov exponent");

    // system wrapper
    UniSys us = UniSys(this);

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

    if (Unsteady && (Mode + 1 == 2))
    {
        us.warning("FLLE makes no sense for transient data (\"nearby trajectories are not nearby\") !!");
    }

    if (IntegrationTime <= 0.0)
    {
        us.error("integration time must be larger than zero");
        return 1;
    }

    if (Unsteady && !fileReadable(VelocityFile))
    {
        us.error("could not open transient data descriptor: %s", VelocityFile);
        return 1;
    }

// create Unstructured wrapper for velocity input
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

    // create Unstructured wrapper for sampling grid
    Unstructured *unst_samplingGrid;
    vtkUnstructuredGrid *sgrid;
    {
        sgrid = generateUniformUSG("samplingGrid",
                                   Origin[0],
                                   Origin[1],
                                   Origin[2],
                                   Cells[0],
                                   Cells[1],
                                   Cells[2],
                                   CellSize);
        unst_samplingGrid = new Unstructured(sgrid, NULL, NULL);
    }

    // compute
    {
        // unstructured wrapper for output
        Unstructured *unst_out;
        vtkFloatArray *FLE = NULL;
        vtkFloatArray *eigenvalMax = NULL;
        vtkFloatArray *eigenvalMed = NULL;
        vtkFloatArray *eigenvalMin = NULL;
        vtkFloatArray *integrationSize = NULL;
        vtkFloatArray *map = NULL;
        {
            // alloc output, TODO future: do it inside Unstructured

            // copy geometry
            outputUGDS->CopyStructure(sgrid);

            // alloc data
            {
                vtkIdType numPts = sgrid->GetNumberOfPoints();

                FLE = vtkFloatArray::New();
                FLE->SetNumberOfComponents(1);
                FLE->SetNumberOfTuples(numPts);
                FLE->SetName("FLE");

                eigenvalMax = vtkFloatArray::New();
                eigenvalMax->SetNumberOfComponents(1);
                eigenvalMax->SetNumberOfTuples(numPts);
                eigenvalMax->SetName("FLE eigenval max");

                eigenvalMed = vtkFloatArray::New();
                eigenvalMed->SetNumberOfComponents(1);
                eigenvalMed->SetNumberOfTuples(numPts);
                eigenvalMed->SetName("FLE eigenval med");

                eigenvalMin = vtkFloatArray::New();
                eigenvalMin->SetNumberOfComponents(1);
                eigenvalMin->SetNumberOfTuples(numPts);
                eigenvalMin->SetName("FLE eigenval min");

                integrationSize = vtkFloatArray::New();
                integrationSize->SetNumberOfComponents(1);
                integrationSize->SetNumberOfTuples(numPts);
                integrationSize->SetName("FLE integration time/length");

                map = vtkFloatArray::New();
                map->SetNumberOfComponents(3);
                map->SetNumberOfTuples(numPts);
                map->SetName("Flow map");
            }

            std::vector<vtkFloatArray *> svec;
            svec.push_back(FLE);
            svec.push_back(eigenvalMax);
            svec.push_back(eigenvalMed);
            svec.push_back(eigenvalMin);
            svec.push_back(integrationSize);
            std::vector<vtkFloatArray *> vvec;
            vvec.push_back(map);
            unst_out = new Unstructured(sgrid, &svec, &vvec);
        }

        // setup Unifield for trajectory output
        UniField *unif_traj = NULL;
        {
            // UniField wrapper
            unif_traj = new UniField(outputSG);

            // alloc trajectory field
            int dims[3] = { IntegStepsMax + 1, sgrid->GetNumberOfPoints(), 1 };
            int compVecLens[1] = { 2 };
            const char *compNames[1] = { "trajectory data" };
            if ((unif_traj)->allocField(2 /*ndims*/, dims, 3 /*nspace*/,
                                        false /*regular*/, 1, compVecLens, compNames, UniField::DT_FLOAT) == false)
            {
                us.error("out of memory");
            }
        }

        // compute
        if (Execute)
        {
            FLE_impl(&us, unst_in, 0, Unsteady, VelocityFile, StartTime,
                     Mode + 1,
                     Ln, DivT, IntegrationTime, IntegrationLength,
                     TimeIntervals, SepFactorMin, IntegStepsMax, Forward,
                     unst_out, SmoothingRange,
                     OmitBoundaryCells, GradNeighDisabled,
                     unif_traj);
        }

        // pass point data (interpolate from input)
        passInterpolatePointData(input, outputUG);
        // an option, it takes very long and needs very much memory
        if (ResampleOnTrajectories)
        {
            us.info("resampling field variables on trajectory output");
            passInterpolatePointData(input, outputSG);
        }

        // add output
        outputUGDS->GetPointData()->AddArray(FLE);
        outputUGDS->GetPointData()->AddArray(eigenvalMax);
        outputUGDS->GetPointData()->AddArray(eigenvalMed);
        outputUGDS->GetPointData()->AddArray(eigenvalMin);
        outputUGDS->GetPointData()->AddArray(integrationSize);
        outputUGDS->GetPointData()->AddArray(map);

        // reference is counted, we can delete
        FLE->Delete();
        eigenvalMax->Delete();
        eigenvalMed->Delete();
        eigenvalMin->Delete();
        integrationSize->Delete();
        map->Delete();

        // delete wrapper (but not the field)
        delete unif_traj;

        // delete unstructured output wrapper
        delete unst_out;
    }

#if !USE_CACHE
    if (unst_in)
        delete unst_in;
#endif

    delete unst_samplingGrid;
    sgrid->Delete();

    return 1;
}

int vtkFLE::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkFLE::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
