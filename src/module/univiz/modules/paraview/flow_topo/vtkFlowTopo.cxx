/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Vector Field Topology
  Module:    $RCSfile: vtkFlowTopo.cxx,v $

  Copyright (c) Ronald Peikert, Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
#include "vtkFlowTopo.h"

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
#include "unifield.h"

// #### for test
#include "vtkCellArray.h"
#include "vtkPolyData.h"

#include "paraview_ext.h"
#include "flow_topo_impl.cpp" // ### including .cpp

#define USE_CACHE 0

static Unstructured *unst = NULL;

vtkCxxRevisionMacro(vtkFlowTopo, "$Revision: 0.06$");
vtkStandardNewMacro(vtkFlowTopo);

vtkFlowTopo::vtkFlowTopo()
{
    this->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::VECTORS);
    this->SetInputArrayToProcess(1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                 vtkDataSetAttributes::SCALARS);
}

vtkFlowTopo::~vtkFlowTopo()
{
}

void generateCross(double posX, double posY, double posZ,
                   double glyphRadius,
                   vtkPoints *outputPoints,
                   vtkCellArray *outputLines,
                   int &numPoints)
{
    // cross of three orthogonal lines
    // -> each line is represented by a VTK cell that contains 2 points

    for (int line = 0; line < 3; line++)
    {

        outputLines->InsertNextCell(2);
        for (int v = 0; v < 2; v++)
        {

            double pnt[3] = { posX, posY, posZ };
            if (v == 0)
                pnt[line] -= glyphRadius / 2.0;
            else
                pnt[line] += glyphRadius / 2.0;

            // store point
            outputPoints->InsertNextPoint(pnt);

            // insert point ID into cell (connectivity info)
            outputLines->InsertCellPoint(numPoints++);
        }
    }
}

int vtkFlowTopo::RequestData(
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
    //vtkPoints *output =
    //vtkPoints::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input)
    {
        return 1;
    }

    vtkIdType numCells = input->GetNumberOfCells();

    // #### Float HACK TODO
    vtkFloatArray *inVectors = (vtkFloatArray *)this->GetInputArrayToProcess(0, inputVector);
    vtkFloatArray *inScalars = (vtkFloatArray *)this->GetInputArrayToProcess(1, inputVector);

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
#if 0 // ### REPLACED BY WORK-AROUND, RE-ACTIVATE WHEN PARAVIEW HAS BUG FIXED
      if (inScalars)
#else
            if (arrayExists(inScalars, inVectors))
#endif
            {
                std::vector<vtkFloatArray *> svec;
                svec.push_back(inScalars);
                unst = new Unstructured(input, &svec, &vvec);
            }
            else
            {
                unst = new Unstructured(input, NULL, &vvec);
            }
        }
#else
        std::vector<vtkFloatArray *> vvec;
        vvec.push_back(inVectors);
#if 0 // ### REPLACED BY WORK-AROUND, RE-ACTIVATE WHEN PARAVIEW HAS BUG FIXED
    if (inScalars)
#else
        if (arrayExists(inScalars, inVectors))
#endif
        {
            std::vector<vtkFloatArray *> svec;
            svec.push_back(inScalars);
            unst = new Unstructured(input, &svec, &vvec);
        }
        else
        {
            unst = new Unstructured(input, NULL, &vvec);
        }
#endif

#if 0 // ### DEBUG
    {
      vec3 v;
      unst->getVector3(100, v);
      printf("vec at node 0:\n");
      vec3dump(v, stdout);

      printf("scal at node 0: %g\n", unst->getScalar(100));

      unst->saveAs("/pub/scratch/sadlof/debug.unst");
    }
#endif

        // compute
        //UniField *unif = new UniField(NULL, NULL);
        UniField *unif = new UniField(NULL);

        flow_topo_impl(&us,
                       unst,
                       unst->getVectorNodeDataComponent(),
                       unst->getScalarNodeDataComponent(),
                       //0, -1 /*compWallDist TODO (when paraview solves the bug)*/,
                       &unif,
                       DivideByWallDist,
                       InteriorCritPts,
                       BoundaryCritPts,
                       GenerateSeeds,
                       SeedsPerCircle,
                       Radius,
                       Offset);

        { // ### HACK
            vtkPoints *outputPoints;
            vtkCellArray *outputLines;

            outputPoints = vtkPoints::New();
            outputLines = vtkCellArray::New();

            int numPoints = 0;

            for (int i = 0; i < unif->getDim(0); i++)
            {
                //printf("pos[%d] = (%g, %g, %g)\n",
                //     i,
                //     unif->vtkPositions[i*3 + 0],
                //     unif->vtkPositions[i*3 + 1],
                //     unif->vtkPositions[i*3 + 2]
                //     );

                vec3 pos;
                unif->getCoord(i, pos);

                generateCross(pos[0], pos[1], pos[2],
                              GlyphRadius,
                              outputPoints, outputLines, numPoints);
            }

            // set geometry
            output->SetPoints(outputPoints);
            output->SetLines(outputLines);

            // pass point data (interpolate from input)
            passInterpolatePointData(input, output);

            // reference is counted, we can delete
            outputPoints->Delete();
            outputLines->Delete();
        }

        // delete field
        unif->freeField();

        // delete field wrapper
        delete unif;

#if !USE_CACHE
        if (unst)
            delete unst;
#endif
    }

    return 1;
}

int vtkFlowTopo::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

void vtkFlowTopo::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
