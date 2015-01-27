/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Read unstructured grid in Unstructured format
  Module:    $RCSfile: vtkDumpCFX.cxx,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkDumpCFX.h"
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
#include "unisys.h"
#include "values.h"

#include "vtkCallbackCommand.h"
#include "vtkCompositeDataPipeline.h"
#include "vtkDataArrayCollection.h"
#include "vtkDataArraySelection.h"

#include "vtkUnstructuredGridAlgorithm.h"

#include "cfx_export_lib.h"
#include "dump_cfx_impl.cpp" // ### including .cpp

#define OUTPUT_ENABLE 0

vtkCxxRevisionMacro(vtkDumpCFX, "$Revision: 0.01 $");
vtkStandardNewMacro(vtkDumpCFX);

// #### for work-around
static bool cfxLibInitialized = false;

//----------------------------------------------------------------------------
vtkDumpCFX::vtkDumpCFX()
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
    this->SelectionObserver->SetCallback(&vtkDumpCFX::SelectionModifiedCallback);
    this->SelectionObserver->SetClientData(this);
    this->PointDataArraySelection->AddObserver(vtkCommand::ModifiedEvent,
                                               this->SelectionObserver);
    this->SelectionModifiedDoNotCallModified = 0;

    this->SetNumberOfInputPorts(0);

    this->OutputPath = NULL;
}

//----------------------------------------------------------------------------
vtkDumpCFX::~vtkDumpCFX()
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

    if (this->OutputPath)
        delete[] this->OutputPath;
    this->OutputPath = NULL;
}

//----------------------------------------------------------------------------
void vtkDumpCFX::SelectionModifiedCallback(vtkObject *,
                                           unsigned long,
                                           void *clientdata,
                                           void *)
{
    static_cast<vtkDumpCFX *>(clientdata)->SelectionModified();
}

//----------------------------------------------------------------------------
void vtkDumpCFX::SelectionModified()
{
    if (!this->SelectionModifiedDoNotCallModified)
    {
        this->Modified();
    }
}

//----------------------------------------------------------------------------
int vtkDumpCFX::getCFXInfo(int **node_components,
                           char **node_component_labels,
                           int &nnodes,
                           int &ncells,
                           int &num_node_components,
                           int &nodeVecLenTot,
                           int &timeStepCnt)
{
    *node_components = NULL;
    *node_component_labels = NULL;

    int num_tetra, num_pyra, num_prisms, num_hexa, node_veclen, num_boundaries;
    float timeVal = 0.0;

    if (cfx_getInfo((const char *)FileName,
                    1, //levelOfInterest.getValue(), //level_of_interest
                    0, //domain.getValue(), //domain
                    0, //crop
                    -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop

                    1, //firstTimeStep.getValue(), //timestep
                    1, // timestep_by_idx
                    &num_tetra, &num_pyra, &num_prisms, &num_hexa, &nnodes,
                    NULL /*components_to_read*/,
                    NULL /*delimiter*/,
                    0, //output_zone_id
                    &node_veclen, &num_node_components,
                    0, //output_boundary_nodes /*output boundary*/,
                    &num_boundaries,
                    &timeVal, /*timeval*/
                    &timeStepCnt,
                    0, //allow_zone_rotation
                    NULL,
                    !cfxLibInitialized, //reopen||!exportInitialized /*exportInit*/, // ########## cfxLibInitialized: work around for cfxLib bug
                    0 //0/*exportDone*/
                    ))
    {
        //us->error("error reading description");
        return 0;
    }

    cfxLibInitialized = true;

    ncells = num_tetra + num_pyra + num_prisms + num_hexa;
    int nodeListSize = num_tetra * 4 + num_pyra * 5 + num_prisms * 6 + num_hexa * 8;

    *node_component_labels = (char *)malloc(num_node_components * 256);
    *node_components = (int *)malloc(num_node_components * sizeof(int));

    if (cfx_getData((const char *)FileName,
                    1, //levelOfInterest.getValue(), //level_of_interest,
                    0, //domain.getValue(), //domain /*zone*/,
                    0, //crop,
                    -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop
                    1, //firstTimeStep.getValue(), //timestep /* timestep */,
                    1, //timestep_by_idx,
                    NULL /*x*/, NULL /*y*/, NULL /*z*/,
                    REQ_TYPE_ALL /*required_cell_type*/,
                    NULL /*node_list*/,
                    NULL /*cell_types*/, NULL /*components_to_read*/,
                    NULL /*delimiter*/, NULL /*node_data*/,
                    *node_components, /*node_components*/
                    *node_component_labels,
                    1, //fix_boundary_nodes,
                    0, //output_zone_id,
                    0, //output_boundary_nodes/*output boundary*/,
                    NULL /*boundary_node_label*/,
                    NULL /*boundary_node_labels*/,
                    NULL, //search_string,
                    0, //allow_zone_rotation, // allowZoneRotation
                    0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    NULL,
                    0 /*exportInit*/,
                    0 /*exportDone*/
                    ))
    {
        //us->error("error reading component info");
        return 0;
    }

    nodeVecLenTot = 0;
    for (int c = 0; c < num_node_components; c++)
    {
        nodeVecLenTot += (*node_components)[c];
    }

    return 1;
}

//----------------------------------------------------------------------------
int vtkDumpCFX::RequestInformation(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **vtkNotUsed(inputVector),
    vtkInformationVector *vtkNotUsed(outputVector))
{
    // system wrapper
    UniSys us = UniSys(this);

    int *node_components = NULL;
    char *node_component_labels = NULL;
    int nnodes, ncells, num_node_components, nodeVecLenTot, timeStepCnt;
    if (!getCFXInfo(&node_components, &node_component_labels,
                    nnodes, ncells, num_node_components, nodeVecLenTot, timeStepCnt))
    {
        us.error("error reading info");
        return 0;
    }

    this->NumberOfNodes = nnodes;
    this->NumberOfCells = ncells;
    this->NumberOfNodeFields = nodeVecLenTot;
    this->NumberOfNodeComponents = num_node_components;

    for (int c = 0; c < NumberOfNodeComponents; c++)
    {
        char buf[256];
        getNodeComponentLabel(node_component_labels, c, buf);
        this->PointDataArraySelection->AddArray(buf);
    }

    free(node_component_labels);
    free(node_components);

    return 1;
}

//----------------------------------------------------------------------------
int vtkDumpCFX::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **vtkNotUsed(inputVector),
    vtkInformationVector *outputVector)
{
    // get the info object
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the ouptut
    vtkUnstructuredGrid *output = vtkUnstructuredGrid::SafeDownCast(
        outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkDebugMacro(<< "Reading CFX file for dump");

    // system wrapper
    UniSys us = UniSys(this);

    int *node_components = NULL;
    char *node_component_labels = NULL;
    int nnodes, ncells, num_node_components, nodeVecLenTot, timeStepCnt;
    if (!getCFXInfo(&node_components, &node_component_labels,
                    nnodes, ncells, num_node_components, nodeVecLenTot, timeStepCnt))
    {
        us.error("error reading info");
        return 0;
    }

    int vecCnt = 0;
    int otherCnt = 0;
    int selectedComponent = -1;
    for (int c = 0; c < NumberOfNodeComponents; c++)
    {
        if (PointDataArraySelection->GetArraySetting(c))
        {
            if (node_components[c] == 3)
            {
                vecCnt++;
                selectedComponent = c;
            }
            else
            {
                otherCnt++;
            }
        }
    }
    if ((vecCnt != 1) || (otherCnt != 0))
    {
        us.error("must select only a single vector variable");
        free(node_components);
        free(node_component_labels);
        return 0;
    }

    if (strlen(OutputPath) == 0)
    {
        free(node_components);
        free(node_component_labels);
        return 0;
    }

    float *node_data_interleaved = new float[nnodes * 3];

    int timeStepsToProcess;
    if (TimeStepNb > 0)
    {
        timeStepsToProcess = TimeStepNb;
    }
    else
    {
        timeStepsToProcess = timeStepCnt;
    }

    char dumpFileNames[timeStepsToProcess][256];

    for (int step = FirstTimeStep; step < FirstTimeStep + timeStepsToProcess; step++)
    {

        // only for getting the time
        int num_tetra, num_pyra, num_prisms, num_hexa, node_veclen, num_boundaries;
        float timeVal = 0.0;
        int timeStepCnt;
        if (cfx_getInfo((const char *)FileName,
                        LevelOfInterest, //level_of_interest
                        Domain, //domain
                        0, //crop
                        -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop

                        step, //timestep
                        1, // timestep_by_idx
                        &num_tetra, &num_pyra, &num_prisms, &num_hexa, &nnodes,
                        NULL /*components_to_read*/,
                        NULL /*delimiter*/,
                        0, //output_zone_id
                        &node_veclen, &num_node_components,
                        0, //output_boundary_nodes /*output boundary*/,
                        &num_boundaries,
                        &timeVal, /*timeval*/
                        &timeStepCnt,
                        0, //allow_zone_rotation
                        NULL,
                        !cfxLibInitialized, //reopen||!exportInitialized /*exportInit*/, // ########## cfxLibInitialized: work around for cfxLib bug
                        0 //0/*exportDone*/
                        ))
        {
            us.error("error reading description for step %d", step);
            free(node_components);
            free(node_component_labels);
            delete[] node_data_interleaved;
            return 0;
        }

        // read data
        char selectedLabel[256];
        getNodeComponentLabel(node_component_labels, selectedComponent, selectedLabel);
        if (cfx_getData((const char *)FileName,
                        LevelOfInterest, //level_of_interest,
                        Domain, //domain /*zone*/,
                        0, //crop,
                        -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop
                        step, //timestep /* timestep */,
                        1, //timestep_by_idx,
#if OUTPUT_ENABLE
                        coordX, coordY, coordZ,
#else
                        NULL, NULL, NULL,
#endif
                        REQ_TYPE_ALL /*required_cell_type*/,
#if OUTPUT_ENABLE
                        cornerList /*node_list*/,
                        cell_types /*cell_types*/,
#else
                        NULL,
                        NULL,
#endif
                        selectedLabel /*components_to_read*/,
                        ";" /*delimiter*/,
                        node_data_interleaved /*node_data*/,
                        node_components, /*node_components*/
                        node_component_labels,
                        1, //fix_boundary_nodes,
                        0, //output_zone_id,
                        0, //output_boundary_nodes/*output boundary*/,
                        NULL /*boundary_node_label*/,
                        NULL /*boundary_node_labels*/,
                        NULL, //search_string,
                        0, //allow_zone_rotation, // allowZoneRotation
                        0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL,
                        0 /*exportInit*/,
                        0 //1/*exportDone*/ // ########## work around (see cfxLibInitialized)
                        ))
        {
            us.error("error reading data at step %d", step);
            free(node_components);
            free(node_component_labels);
            delete[] node_data_interleaved;
            return 0;
        }

        // dump
        {
            int relStep = step - FirstTimeStep;
            FILE *fp;
            char name[256];
            sprintf(name, "%s/%.6f", OutputPath, timeVal);
            sprintf(dumpFileNames[relStep], "%.6f", timeVal);
            fp = fopen(name, "wb");
            if (!fp)
            {
                us.error("error opening output file %s", name);
                free(node_components);
                free(node_component_labels);
                delete[] node_data_interleaved;
                return 0;
            }

            fwrite(node_data_interleaved, nnodes * 3 * sizeof(float), 1, fp);

            fclose(fp);
        }
    }

    // generate mmap file(s) and descriptor file
    if (GenerateMMapFiles)
    {
        generateMMapFile(timeStepsToProcess, dumpFileNames,
                         nnodes,
                         MMapFileSizeMax,
                         OutputPath);
    }

    // cleanup (delete) dump files
    if (DeleteDumpFiles)
    {
        for (int i = 0; i < timeStepsToProcess; i++)
        {
            char name[256];
            sprintf(name, "%s/%s", OutputPath, dumpFileNames[i]);
            remove(name);
        }
    }

    free(node_components);
    free(node_component_labels);
    delete[] node_data_interleaved;

    return 1;
}

//----------------------------------------------------------------------------
void vtkDumpCFX::PrintSelf(ostream &os, vtkIndent indent)
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
void vtkDumpCFX::GetNodeDataRange(int nodeComp, int index, float *min, float *max)
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
const char *vtkDumpCFX::GetPointArrayName(int index)
{
    return this->PointDataArraySelection->GetArrayName(index);
}

//----------------------------------------------------------------------------
int vtkDumpCFX::GetPointArrayStatus(const char *name)
{
    return this->PointDataArraySelection->ArrayIsEnabled(name);
}

//----------------------------------------------------------------------------
void vtkDumpCFX::SetPointArrayStatus(const char *name, int status)
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
int vtkDumpCFX::GetNumberOfPointArrays()
{
    return this->PointDataArraySelection->GetNumberOfArrays();
}

//----------------------------------------------------------------------------
void vtkDumpCFX::EnableAllPointArrays()
{
    this->PointDataArraySelection->EnableAllArrays();
}

//----------------------------------------------------------------------------
void vtkDumpCFX::DisableAllPointArrays()
{
    this->PointDataArraySelection->DisableAllArrays();
}

//----------------------------------------------------------------------------
int vtkDumpCFX::GetNumberOfCellArrays()
{
    //return this->CellDataArraySelection->GetNumberOfArrays();
    return 0;
}
