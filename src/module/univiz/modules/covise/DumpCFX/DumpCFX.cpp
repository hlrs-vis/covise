/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// DumpCFX
// Filip Sadlo 2008
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "DumpCFX.h"

#include "linalg.h"

#include "unstructured.h"
#include "unisys.h"

#include "cfx_export_lib.h"
#include "dump_cfx_impl.cpp" // ### including .cpp

// #### for work-around
static bool cfxLibInitialized = false;

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::postInst()
{
}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // system wrapper
    us = UniSys(this);

    int num_node_components;
    int num_tetra, num_pyra, num_prisms, num_hexa, nnodes, node_veclen,
        num_boundaries;
    char *node_component_labels = NULL;
    float timeVal = 0.0;
    int timeStepCnt;

    if (cfx_getInfo((const char *)fileName->getValue(),
                    levelOfInterest.getValue(), //level_of_interest
                    domain.getValue(), //domain
                    0, //crop
                    -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop

                    firstTimeStep.getValue(), //timestep
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
        us.error("error reading description");
        return FAIL;
    }

    cfxLibInitialized = true;

    int ncells = num_tetra + num_pyra + num_prisms + num_hexa;
    int nodeListSize = num_tetra * 4 + num_pyra * 5 + num_prisms * 6 + num_hexa * 8;

    node_component_labels = (char *)malloc(num_node_components * 256);
    int *node_components = (int *)malloc(num_node_components * sizeof(int));

    if (cfx_getData((const char *)fileName->getValue(),
                    levelOfInterest.getValue(), //level_of_interest,
                    domain.getValue(), //domain /*zone*/,
                    0, //crop,
                    -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, // crop
                    firstTimeStep.getValue(), //timestep /* timestep */,
                    1, //timestep_by_idx,
                    NULL /*x*/, NULL /*y*/, NULL /*z*/,
                    REQ_TYPE_ALL /*required_cell_type*/,
                    NULL /*node_list*/,
                    NULL /*cell_types*/, NULL /*components_to_read*/,
                    NULL /*delimiter*/, NULL /*node_data*/,
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
                    0 /*exportDone*/
                    ))
    {
        us.error("error reading data");
        free(node_components);
        free(node_component_labels);
        return FAIL;
    }

    // adapt component selectors
    //int nodeCompScal = -1;
    int nodeCompVec = -1;
    //int scalCnt = 0;
    int vecCnt = 0;
    //char *nodeScalarLabels[1024];
    char *nodeVectorLabels[1024];
    //int selectedScal = -1;
    int selectedVec = -1;
    {
        for (int c = 0; c < num_node_components; c++)
        {

#if 0
      if (node_components[c] == 1) {

        if (scalarComponent->getValue() == scalCnt) {
          selectedScal = scalCnt;
          nodeCompScal = c;
        }

        nodeScalarLabels[scalCnt] = new char[256];
        //strcpy(nodeScalarLabels[scalCnt], unst_in->getNodeCompLabel(c));
        getNodeComponentLabel(node_component_labels, c, nodeScalarLabels[scalCnt]);

        scalCnt++;
      }
      else
#endif
            if (node_components[c] == 3)
            {

                if (vectorComponent->getValue() == vecCnt)
                {
                    selectedVec = vecCnt;
                    nodeCompVec = c;
                }

                nodeVectorLabels[vecCnt] = new char[256];
                //strcpy(nodeVectorLabels[vecCnt], unst_in->getNodeCompLabel(c));
                getNodeComponentLabel(node_component_labels, c, nodeVectorLabels[vecCnt]);

                vecCnt++;
            }
            else
            {
                //us.error("unsupported vector length");
            }
        }

#if 0
    if (scalCnt > 0) {
      scalarComponent->updateValue(scalCnt, nodeScalarLabels, selectedScal);
    }
    else {
      scalarComponent->setValue(1, defaultChoice, 0);
    }
#endif

        if (vecCnt > 0)
        {
            vectorComponent->updateValue(vecCnt, nodeVectorLabels, selectedVec);
        }
        else
        {
            vectorComponent->setValue(1, defaultChoice, 0);
        }
    }

    if (strlen(outputPath->getValue()) == 0)
    {
        free(node_components);
        free(node_component_labels);
        return FAIL;
    }

    // generate output
    {
#if OUTPUT_ENABLE
        coDoUnstructuredGrid *gridData = NULL;

        // ### TODO: check
        float *coordX = new float[nnodes];
        float *coordY = new float[nnodes];
        float *coordZ = new float[nnodes];
        int *elemList = new int[ncells];
        int *typeList = new int[ncells];
        int *cornerList = new int[nodeListSize];

        unsigned char *cell_types = new unsigned char[ncells];
#endif
        float *node_data_interleaved = new float[nnodes * 3];

        int timeStepsToProcess;
        if (timeStepNb.getValue() > 0)
        {
            timeStepsToProcess = timeStepNb.getValue();
        }
        else
        {
            timeStepsToProcess = timeStepCnt;
        }

        char dumpFileNames[timeStepsToProcess][256];

        for (int step = firstTimeStep.getValue(); step < firstTimeStep.getValue() + timeStepsToProcess; step++)
        {

            // only for getting the time
            if (cfx_getInfo((const char *)fileName->getValue(),
                            levelOfInterest.getValue(), //level_of_interest
                            domain.getValue(), //domain
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
                return FAIL;
            }

            // read data
            if (cfx_getData((const char *)fileName->getValue(),
                            levelOfInterest.getValue(), //level_of_interest,
                            domain.getValue(), //domain /*zone*/,
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
                            nodeVectorLabels[selectedVec] /*components_to_read*/,
                            (char *)";" /*delimiter*/,
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
                return FAIL;
            }

            // dump
            {
                int relStep = step - firstTimeStep.getValue();
                FILE *fp;
                char name[256];
                sprintf(name, "%s/%.6f", outputPath->getValue(), timeVal);
                sprintf(dumpFileNames[relStep], "%.6f", timeVal);
                fp = fopen(name, "wb");
                if (!fp)
                {
                    us.error("error opening output file %s", name);
                    free(node_components);
                    free(node_component_labels);
                    return FAIL;
                }

                fwrite(node_data_interleaved, nnodes * 3 * sizeof(float), 1, fp);

                fclose(fp);
            }
        }

        // generate mmap file(s) and descriptor file
        if (generateMMapFiles->getValue())
        {
            generateMMapFile(timeStepsToProcess, dumpFileNames,
                             nnodes,
                             mmapFileSizeMax.getValue(),
                             outputPath->getValue());
        }

        // cleanup (delete) dump files
        if (deleteDumpFiles->getValue())
        {
            for (int i = 0; i < timeStepsToProcess; i++)
            {
                char name[256];
                sprintf(name, "%s/%s", outputPath->getValue(), dumpFileNames[i]);
                remove(name);
            }
        }

#if OUTPUT_ENABLE

        // data
        //coDoFloat *scalarData = NULL;
        coDoVec3 *vectorData = NULL;
        {
#if 0
      if (nodeCompScal >= 0) {
        scalarData = new coDoFloat(scalar->getObjName(), nnodes);

        float *wp;
        scalarData->getAddress(&wp);
        for (int n=0; n<nnodes; n++) {
          wp[n] = unst_in->getScalar(n, nodeCompScal);
        }
      }
#endif
            if (nodeCompVec >= 0)
            {
                vectorData = new coDoVec3(vector->getObjName(), nnodes);

                float *up, *vp, *wp;
                vectorData->getAddresses(&up, &vp, &wp);
                for (int n = 0; n < nnodes; n++)
                {
                    up[n] = node_data_interleaved[n * 3 + 0];
                    vp[n] = node_data_interleaved[n * 3 + 1];
                    wp[n] = node_data_interleaved[n * 3 + 2];
                }
            }
        }

        // cells
        int cornerListCnt = 0;
        for (int c = 0; c < ncells; c++)
        {

            // type
            switch (cell_types[c])
            {
            case REQ_TYPE_TETRA:
            {
                typeList[c] = TYPE_TETRAHEDER;
            }
            break;
            case REQ_TYPE_PYRAM:
            {
                typeList[c] = TYPE_PYRAMID;
            }
            break;
            case REQ_TYPE_WEDGE:
            {
                typeList[c] = TYPE_PRISM;
            }
            break;
            case REQ_TYPE_HEXA:
            {
                typeList[c] = TYPE_HEXAEDER;
            }
            break;
            }

            // nodes
            // ######### HACK: assumes cell types identical between AVS and Covise
            int *cellNodesAVS = &cornerList[cornerListCnt];
            int cellNodes[8];
            Unstructured::nodeOrderAVStoCovise(typeList[c], cellNodesAVS, cellNodes);
            int nvert = nVertices[typeList[c]];
            memcpy(cornerList + cornerListCnt, cellNodes, nvert * sizeof(int));
            elemList[c] = cornerListCnt;
            cornerListCnt += nvert;
        }

        gridData = new coDoUnstructuredGrid(grid->getObjName(),
                                            ncells,
                                            nodeListSize,
                                            nnodes,
                                            elemList,
                                            cornerList,
                                            coordX, coordY, coordZ,
                                            typeList);
#endif

        delete[] node_data_interleaved;
#if OUTPUT_ENABLE
        delete[] cell_types;
        delete[] coordX;
        delete[] coordY;
        delete[] coordZ;
        delete[] elemList;
        delete[] typeList;
        delete[] cornerList;
#endif

        //for (int i=0; i<scalCnt; i++) {
        //  delete [] nodeScalarLabels[i];
        //}
        for (int i = 0; i < vecCnt; i++)
        {
            delete[] nodeVectorLabels[i];
        }

// assign data to ports
#if OUTPUT_ENABLE
        grid->setCurrentObject(gridData);
        // if (nodeCompScal >= 0) scalar->setCurrentObject(scalarData);
        if (nodeCompVec >= 0)
            vector->setCurrentObject(vectorData);
#endif
    }

    free(node_components);
    free(node_component_labels);

    return SUCCESS;
}
