/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Read unstructured grid in Unstructured format
  Module:    $RCSfile: vtkReadUnstructured.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkReadUnstructured.h"
#include "vtkDataArraySelection.h"
#include "vtkErrorCode.h"
#include "vtkUnstructuredGrid.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkFieldData.h"
#include "vtkPointData.h"
#include "vtkByteSwap.h"
#include "vtkIdTypeArray.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkByteSwap.h"

#include "unstructured.h"

#include "vtkCallbackCommand.h"
#include "vtkCompositeDataPipeline.h"
#include "vtkDataArrayCollection.h"
#include "vtkDataArraySelection.h"

#include "vtkUnstructuredGridAlgorithm.h"

vtkCxxRevisionMacro(vtkReadUnstructured, "$Revision: 0.01 $");
vtkStandardNewMacro(vtkReadUnstructured);

//----------------------------------------------------------------------------
vtkReadUnstructured::vtkReadUnstructured()
{
    this->FileName = NULL;
    this->NumberOfNodeFields = 0;
    this->NumberOfFields = 0;
    this->NumberOfNodeComponents = 0;
    this->NumberOfNodes = 0;
    this->NumberOfCells = 0;

    this->NodeDataInfo = NULL;
    this->PointDataArraySelection = vtkDataArraySelection::New();

    // Setup the selection callback to modify this object when an array
    // selection is changed.
    this->SelectionObserver = vtkCallbackCommand::New();
    this->SelectionObserver->SetCallback(&vtkReadUnstructured::SelectionModifiedCallback);
    this->SelectionObserver->SetClientData(this);
    this->PointDataArraySelection->AddObserver(vtkCommand::ModifiedEvent,
                                               this->SelectionObserver);
    this->SelectionModifiedDoNotCallModified = 0;

    this->SetNumberOfInputPorts(0);
}

//----------------------------------------------------------------------------
vtkReadUnstructured::~vtkReadUnstructured()
{
    if (this->FileName)
    {
        delete[] this->FileName;
    }
    if (this->NodeDataInfo)
    {
        delete[] this->NodeDataInfo;
    }

    this->PointDataArraySelection->RemoveObserver(this->SelectionObserver);
    this->SelectionObserver->Delete();

    this->PointDataArraySelection->Delete();
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::SelectionModifiedCallback(vtkObject *,
                                                    unsigned long,
                                                    void *clientdata,
                                                    void *)
{
    static_cast<vtkReadUnstructured *>(clientdata)->SelectionModified();
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::SelectionModified()
{
    if (!this->SelectionModifiedDoNotCallModified)
    {
        this->Modified();
    }
}

//----------------------------------------------------------------------------
int vtkReadUnstructured::RequestInformation(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **vtkNotUsed(inputVector),
    vtkInformationVector *vtkNotUsed(outputVector))
{
    // TODO: read only once, this is inefficient (RequestData reads again)
    Unstructured *unst = new Unstructured(FileName);
    if (!unst)
    {
        vtkErrorMacro("error reading Unstructured file");
        return 0;
    }

    this->NumberOfNodes = unst->nNodes;
    this->NumberOfCells = unst->nCells;
    this->NumberOfNodeFields = unst->getNodeVecLenTot();
    this->NumberOfNodeComponents = unst->getNodeCompNb();

    for (int c = 0; c < NumberOfNodeComponents; c++)
    {
        this->PointDataArraySelection->AddArray(unst->getNodeCompLabel(c));
    }

    delete unst;
    return 1;
}

//----------------------------------------------------------------------------
int vtkReadUnstructured::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **vtkNotUsed(inputVector),
    vtkInformationVector *outputVector)
{
    // get the info object
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the ouptut
    vtkUnstructuredGrid *output = vtkUnstructuredGrid::SafeDownCast(
        outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkDebugMacro(<< "Reading Unstructured file");

    Unstructured *unst_all = new Unstructured(FileName);
    if (!unst_all)
    {
        vtkErrorMacro("error reading Unstructured file");
        return 0;
    }

    this->Convert(unst_all, output);

    delete unst_all;

    return 1;
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::Convert(Unstructured *unst_all, vtkUnstructuredGrid *output)
{
    vtkFloatArray *coords = vtkFloatArray::New();
    coords->SetNumberOfComponents(3);
    coords->SetNumberOfTuples(unst_all->nNodes);

    vtkIdTypeArray *listcells = vtkIdTypeArray::New();
    // this array contains a list of NumberOfCells tuples
    // each tuple is 1 integer, i.e. the number of indices following it (N)
    // followed by these N integers
    listcells->SetNumberOfValues(unst_all->nCells + unst_all->getNodeListSize());

    vtkCellArray *cells = vtkCellArray::New();
    vtkPoints *points = vtkPoints::New();

    int *types = new int[unst_all->nCells];
    if (!types)
    {
        printf("allocation failed\n");
    }

    // set coordinates and connectivity
    {
        vtkIdType *clist = listcells->GetPointer(0);
        float *ptr = coords->GetPointer(0);
        int n = 0;
        int c = 0;

        for (int n = 0; n < unst_all->nNodes; n++)
        {
            vec3 pos;
            unst_all->getCoords(n, pos);
            ptr[n * 3 + 0] = pos[0];
            ptr[n * 3 + 1] = pos[1];
            ptr[n * 3 + 2] = pos[2];
        }

        int cpos = 0;
        for (int c = 0; c < unst_all->nCells; c++)
        {
            int *cellNodesAVS = unst_all->getCellNodesAVS(c);
            int type = unst_all->getCellType(c);
            int cellNodes[8];
            Unstructured::nodeOrderAVStoVTK(type, cellNodesAVS, cellNodes);
            int nvert = nVertices[type];
            clist[cpos++] = nvert;
            for (int v = 0; v < nvert; v++)
            {
                clist[cpos++] = cellNodes[v];
            }
            switch (type)
            {
            case Unstructured::CELL_TET:
                types[c] = VTK_TETRA;
                break;
            case Unstructured::CELL_PYR:
                types[c] = VTK_PYRAMID;
                break;
            case Unstructured::CELL_PRISM:
                types[c] = VTK_WEDGE;
                break;
            case Unstructured::CELL_HEX:
                types[c] = VTK_HEXAHEDRON;
                break;

            default:
                vtkErrorMacro("unsupported type");
            }
        }
    }

    cells->SetCells(unst_all->nCells, listcells);
    output->SetCells(types, cells);
    points->SetData(coords);
    output->SetPoints(points);

    delete[] types;
    listcells->Delete();
    cells->Delete();
    coords->Delete();
    points->Delete();

    // set data
    for (int c = 0; c < NumberOfNodeComponents; c++)
    {
        if (PointDataArraySelection->GetArraySetting(c))
        {
            vtkFloatArray *scalars = vtkFloatArray::New();
            scalars->SetNumberOfComponents(unst_all->getNodeCompVecLen(c));
            scalars->SetNumberOfTuples(NumberOfNodes);
            scalars->SetName(PointDataArraySelection->GetArrayName(c));

            float *ptr = scalars->GetPointer(0);
            // ### inefficient, TODO: use memcpy
            if (unst_all->getNodeCompVecLen(c) == 1)
            {
                for (int n = 0; n < NumberOfNodes; n++)
                {
                    ptr[n] = unst_all->getScalar(n, c);
                }
            }
            else
            {
                // TODO: assuming vec3
                for (int n = 0; n < NumberOfNodes; n++)
                {
                    vec3 v;
                    unst_all->getVector3(n, c, v);
                    ptr[n * 3 + 0] = v[0];
                    ptr[n * 3 + 1] = v[1];
                    ptr[n * 3 + 2] = v[2];
                }
            }

            output->GetPointData()->AddArray(scalars);
            if (!output->GetPointData()->GetScalars())
            {
                output->GetPointData()->SetScalars(scalars);
            }
            scalars->Delete();
        }
    }
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);

    os << indent << "File Name: "
       << (this->FileName ? this->FileName : "(none)") << "\n";

    os << indent << "Number Of Nodes: " << this->NumberOfNodes << endl;
    os << indent << "Number Of Node Fields: "
       << this->NumberOfNodeFields << endl;
    os << indent << "Number Of Node Components: "
       << this->NumberOfNodeComponents << endl;

    os << indent << "Number Of Cells: " << this->NumberOfCells << endl;

    os << indent << "Number of Fields: " << this->NumberOfFields << endl;
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::GetNodeDataRange(int nodeComp, int index, float *min, float *max)
{
    if (index >= this->NodeDataInfo[nodeComp].veclen || index < 0)
    {
        index = 0; // if wrong index, set it to zero
    }
    // *min = this->NodeDataInfo[nodeComp].min[index];
    // *max = this->NodeDataInfo[nodeComp].max[index];
    // #### TODO
    *min = 0;
    *max = 1;
}

//----------------------------------------------------------------------------
const char *vtkReadUnstructured::GetPointArrayName(int index)
{
    return this->PointDataArraySelection->GetArrayName(index);
}

//----------------------------------------------------------------------------
int vtkReadUnstructured::GetPointArrayStatus(const char *name)
{
    return this->PointDataArraySelection->ArrayIsEnabled(name);
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::SetPointArrayStatus(const char *name, int status)
{
    if (status)
    {
        this->PointDataArraySelection->EnableArray(name);
    }
    else
    {
        this->PointDataArraySelection->DisableArray(name);
    }
}

//----------------------------------------------------------------------------
int vtkReadUnstructured::GetNumberOfPointArrays()
{
    return this->PointDataArraySelection->GetNumberOfArrays();
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::EnableAllPointArrays()
{
    this->PointDataArraySelection->EnableAllArrays();
}

//----------------------------------------------------------------------------
void vtkReadUnstructured::DisableAllPointArrays()
{
    this->PointDataArraySelection->DisableAllArrays();
}

//----------------------------------------------------------------------------
int vtkReadUnstructured::GetNumberOfCellArrays()
{
    //return this->CellDataArraySelection->GetNumberOfArrays();
    return 0;
}
