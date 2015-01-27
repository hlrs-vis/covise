/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description:                                                             **
**                                                                          **
** Name:     ReadABAQUSfil                                                  **
** Category: I/0 Module                                                     **
**                                                                          **
** Author: Ralf Schneider	                                             **
**                                                                          **
** History:  								     **
**                					       		     **
**                                                                          **
**                                                                          **
\****************************************************************************/

// Standard headers
#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include <stdint.h>
#else
#include <inttypes.h>
#endif
#include <sys/stat.h>
// COVISE data types
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
// this includes our own class's headers
#include "ReadABAQUSfil.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static const char *sigma[] = { "sigma_xx", "sigma_yy", "sigma_zz",
                               "sigma_xy", "sigma_xz", "sigma_yz" };
static const char *epsilon[] = { "epsilon_xx", "epsilon_yy", "epsilon_zz",
                                 "epsilon_xy", "epsilon_xz", "epsilon_yz" };
static const char *equivalence[] = { "von_Mises" };

ReadABAQUSfil::ReadABAQUSfil(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Read ABAQUS .fil result file")
{

    // Parameters
    p_filFile = addFileBrowserParam("p_filFile", "ABAQUS .fil result File");
    p_filFile->setValue("xxx.fil", "*.fil");

    const char *no_info[] = { "No info" };

    p_elemres = addChoiceParam("p_elem_res", "Select element results to be loaded");
    p_elemres->setValue(1, no_info, 0);

    // Ports
    p_gridOutPort = addOutputPort("p_gridOutPort", "UnstructuredGrid", "Read unstructured grid");
    p_eresOutPort = addOutputPort("p_eresOutPort", "Float", "Loaded element results");

    // not yet in compute call
    computeRunning = false;

    // Init .fil file storage
    fil_array = NULL;
    data_length = 0;

    // Set initial .fil filename to nothing
    fil_name = "xxx.fil";
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  param() is called every time a parameter is changed (really ??)
// ++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadABAQUSfil::param(const char *paramName, bool in_map_loading)
{
#ifdef WIN32
    struct _stat attribut;
#else
    struct stat attribut;
#endif
    int64_t fil_size;

    FILE *fd_fil;
    int ii, jj, cp;

    int rec_struct[3000];
    int rec_length, rec_type, no_recs;

    int file_offset = 4;

    double *tmp_d;
    int *tmp_i;

    tmp_d = (double *)fil_array;
    tmp_i = (int *)fil_array;

    // *************************************************************************
    // If param is called in case of fil file selection ************************
    if ((0 == strcmp(p_filFile->getName(), paramName)) &&
#ifdef _WIN32
        (0 == _stat(p_filFile->getValue(), &attribut)) &&
#else
        (0 == stat(p_filFile->getValue(), &attribut)) &&
#endif
        (0 != strcmp(p_filFile->getValue(), fil_name)) && (!computeRunning))
    {

        // Allocate fil array and open fil file ***
        fil_size = ((int)attribut.st_size - 8) / 8;

        printf("Need %ld 8-byte elements to store file \n", (long int)fil_size);

        // In case a .fil file was already loaded free memory ***
        if (fil_array != NULL)
        {
            free(fil_array);
        }

        fil_array = (int64_t *)malloc(fil_size * sizeof(int64_t));

        // Open ABAQUS result file *************************************
        if ((fd_fil = fopen(p_filFile->getValue(), "rb")) == NULL)
        {
            sendError("Failed to open ABAQUS result file '%s'",
                      p_filFile->getValue());
        }
        else
        {

            // Skip first record control word in .fil file *****************
            fseek(fd_fil, file_offset, SEEK_SET);

            fread(fil_array, sizeof(int64_t), fil_size, fd_fil);

            // Close ABAQUS result file ************************************
            fclose(fd_fil);

            // Eliminate record control words from fil_array ***************
            ii = 513;
            cp = 512;

            while (ii < fil_size)
            {
                for (jj = 1; jj <= 512; ++jj)
                {
                    fil_array[cp] = fil_array[ii];
                    cp = cp + 1;
                    ii = ii + 1;
                };
                ii = ii + 1;
            };

            data_length = cp - 1;

            // *************************************************************
            // Analyze record structure ************************************

            // Init record count ************
            for (ii = 0; ii <= 2999; ++ii)
            {
                rec_struct[ii] = 0;
            }

            no_recs = 0;
            jj = 0;

            // Count records *********************************
            while (jj < data_length)
            {

                rec_length = fil_array[jj];
                rec_type = fil_array[jj + 1];

                rec_struct[rec_type] = rec_struct[rec_type] + 1;

                // Parse the job-header record *****************
                if (rec_type == 1921)
                {

                    jobhead.version[8] = 0;
                    *(int64_t *)jobhead.version = fil_array[jj + 2];

                    // Parse date and time of job ****
                    // jobhead.date =
                    // jobhead.time =

                    jobhead.typical_el_length = float(fil_array[jj + 8]);
                }

                jj = jj + rec_length;

                no_recs = no_recs + 1;
            }

            jobhead.no_nodes = rec_struct[1901];
            jobhead.no_elems = rec_struct[1900];

            jobhead.no_node_sets = rec_struct[1931];
            jobhead.no_elem_sets = rec_struct[1933];

            jobhead.no_steps = rec_struct[2000];

            // **************************************************
            // Setup drop down list for element results *********
            int ii_choiseVals = 0;

            if (rec_struct[11] > 0)
            {
                ii_choiseVals = ii_choiseVals + 6;
            }
            if (rec_struct[21] > 0)
            {
                ii_choiseVals = ii_choiseVals + 6;
            }
            // Equivalence Stresses ************
            if (ii_choiseVals > 0)
            {
                // Von Mises *********************
                ii_choiseVals = ii_choiseVals + 1;
            }

            const char **choiseVals = new const char *[ii_choiseVals];
            ii_choiseVals = 0;

            // Stresses *********************************
            if (rec_struct[11] > 0)
            {
                for (ii = 0; ii < 6; ++ii)
                {
                    choiseVals[ii_choiseVals] = sigma[ii];
                    ii_choiseVals = ii_choiseVals + 1;
                }
            }

            // Strains **********************************
            if (rec_struct[21] > 0)
            {
                for (ii = 0; ii < 6; ++ii)
                {
                    choiseVals[ii_choiseVals] = epsilon[ii];
                    ii_choiseVals = ii_choiseVals + 1;
                }
            }

            // Equivalence Stresses *********************
            if (ii_choiseVals > 0)
            {

                // Von Mises ******************************
                choiseVals[ii_choiseVals] = equivalence[0];

                ii_choiseVals = ii_choiseVals + 1;
            }

            p_elemres->setValue(ii_choiseVals, choiseVals, 0);

            // Log ***************************************
            printf("%s\n", "Rec # - No.");
            for (ii = 0; ii <= 2999; ++ii)
            {
                if (rec_struct[ii] > 0)
                {
                    printf("%5d - %d\n", ii, rec_struct[ii]);
                }
            }

            // ***********************************************************************
            // Look through elements to set up connection list ***********************
            //
            // Element types as enum see covise/src/kernel/do/coDoUnstructuredGrid.h
            // TYPE_HEXAGON = 7,
            // TYPE_HEXAEDER = 7,
            // TYPE_PRISM = 6,
            // TYPE_PYRAMID = 5,
            // TYPE_TETRAHEDER = 4,
            // TYPE_QUAD = 3,
            // TYPE_TRIANGLE = 2,
            // TYPE_BAR = 1,
            // TYPE_NONE = 0,
            // TYPE_POINT = 10

            jobhead.no_conn = 0;
            jobhead.no_sup_elems = 0;
            jj = 0;

            while (jj < data_length)
            {

                rec_length = fil_array[jj];
                rec_type = fil_array[jj + 1];

                if (rec_type == 1900)
                {

                    switch (fil_array[jj + 3])
                    {

                    case 2314885531223470915: // "C3D8    "
                        jobhead.no_conn = jobhead.no_conn + 8;
                        break;

                    case 2314885530821735251: // "S3R     "
                        jobhead.no_conn = jobhead.no_conn + 3;
                        break;

                    case 2314885531156362051: // "C3D4    "
                        jobhead.no_conn = jobhead.no_conn + 4;
                        break;

                    case 2314885531189916483: // "C3D6    "
                        jobhead.no_conn = jobhead.no_conn + 6;
                        break;

                    case 2314885530819572546: // "B31     "
                        //jobhead.no_conn  = jobhead.no_conn  + 2;
                        jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
                        break;

                    case 2324217284263039059: // "SPRINGA "
                        //jobhead.no_conn  = jobhead.no_conn  + 2;
                        jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
                        break;

                    case 2314885531122807636: // "T3D2    "
                        //jobhead.no_conn  = jobhead.no_conn  + 2;
                        jobhead.no_sup_elems = jobhead.no_sup_elems - 1;
                        break;

                    default:

                        sendError("While counting connections : Unknown element type '%ld'", (long int)fil_array[jj + 3]);

                        char temp[9];
                        temp[8] = 0;
                        *(int64_t *)temp = fil_array[jj + 3];
                        printf("%8s - %ld\n", temp, (long int)fil_array[jj + 3]);

                        break;
                    };

                    jobhead.no_sup_elems = jobhead.no_sup_elems + 1;
                };

                jj = jj + rec_length;
            }
            delete[] choiseVals;

            if ((jobhead.no_elems - jobhead.no_sup_elems) > 0)
            {
                sendWarning("Found %d unsupported elements", jobhead.no_elems - jobhead.no_sup_elems);
            }
        }
    }
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadABAQUSfil::compute(const char *port)
{
    (void)port;

    int ii, jj;

    // We have to specify element types
    int hasTypes = 1;

    // Cast pointers for fil_array ***
    double *tmp_d;
    int *tmp_i;

    tmp_d = (double *)fil_array;
    tmp_i = (int *)fil_array;

    computeRunning = true;

    //===========================================================================

    // Declare grid Pointers ****************************************************
    int *outElemList, *outConnList, *outTypeList;
    float *outXCoord, *outYCoord, *outZCoord;

    // allocate new Unstructured grid *******************************************
    coDoUnstructuredGrid *outGrid = new coDoUnstructuredGrid(p_gridOutPort->getObjName(),
                                                             jobhead.no_sup_elems,
                                                             jobhead.no_conn,
                                                             jobhead.no_nodes,
                                                             hasTypes);
    // if object was not properly allocated *************************************
    if (!outGrid->objectOk())
    {
        sendError("Failed to create object '%s' for port '%s'",
                  p_gridOutPort->getObjName(), p_gridOutPort->getName());

        return FAIL;
    }

    outGrid->getAddresses(&outElemList, &outConnList,
                          &outXCoord, &outYCoord, &outZCoord);

    outGrid->getTypeList(&outTypeList);

    // ***************************************************************************
    // Load coordinates, elements and their external numbers *********************

    int ii_nodes = 0;
    int ii_Elems = 0;
    int ii_Conn = 0;

    jj = 0;

    int *ext_nn;
    ext_nn = (int *)malloc(jobhead.no_nodes * sizeof(int));
    int max_ext_nn = -1;

    while ((jj < data_length) && (fil_array[jj + 1] != 2000))
    {

        // Parse node records **********************
        if (fil_array[jj + 1] == 1901)
        {

            ext_nn[ii_nodes] = (int)fil_array[jj + 2] - 1;
            if (max_ext_nn < (int)fil_array[jj + 2] - 1)
                max_ext_nn = (int)fil_array[jj + 2] - 1;

            outXCoord[ii_nodes] = float(tmp_d[jj + 3]);
            outYCoord[ii_nodes] = float(tmp_d[jj + 4]);
            outZCoord[ii_nodes] = float(tmp_d[jj + 5]);

            ii_nodes = ii_nodes + 1;
        };

        jj = jj + fil_array[jj];
    }

    // set up cross reference array for external node numbers *******************
    // WARNING !! It is assumed that the node numbers are in accending order
    int *cref_nodes;
    cref_nodes = (int *)malloc(max_ext_nn * sizeof(int));
    for (ii = 0; ii < max_ext_nn; ++ii)
    {
        cref_nodes[ii] = -1;
    }
    for (ii = 0; ii < jobhead.no_nodes; ++ii)
    {
        cref_nodes[ext_nn[ii]] = ii;
    }

    jj = 0;

    int *ext_en;
    ext_en = (int *)malloc(jobhead.no_sup_elems * sizeof(int));
    int max_ext_en = -1;

    while ((jj < data_length) && (fil_array[jj + 1] != 2000))
    {
        // Parse element records **********************
        if (fil_array[jj + 1] == 1900)
        {

            ext_en[ii_Elems] = (int)fil_array[jj + 2];

            outElemList[ii_Elems] = ii_Conn;

            switch (fil_array[jj + 3])
            {

            case 2314885531223470915: // "C3D8    "
                for (ii = 0; ii < 8; ++ii)
                {
                    outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
                ii_Conn = ii_Conn + 8;
                outTypeList[ii_Elems] = 7;
                if (max_ext_en < (int)fil_array[jj + 2])
                    max_ext_en = (int)fil_array[jj + 2];
                break;

            case 2314885530821735251: // "S3R     "
                for (ii = 0; ii < 3; ++ii)
                {
                    outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
                ii_Conn = ii_Conn + 3;
                outTypeList[ii_Elems] = 2;
                if (max_ext_en < (int)fil_array[jj + 2])
                    max_ext_en = (int)fil_array[jj + 2];
                break;

            case 2314885531156362051: // "C3D4    "
                for (ii = 0; ii < 4; ++ii)
                {
                    outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
                ii_Conn = ii_Conn + 4;
                outTypeList[ii_Elems] = 4;
                if (max_ext_en < (int)fil_array[jj + 2])
                    max_ext_en = (int)fil_array[jj + 2];
                break;

            case 2314885531189916483: // "C3D6    "
                for (ii = 0; ii < 6; ++ii)
                {
                    outConnList[ii_Conn + ii] = cref_nodes[fil_array[jj + 4 + ii] - (int64_t)1];
                }
                ii_Conn = ii_Conn + 6;
                outTypeList[ii_Elems] = 6;
                if (max_ext_en < (int)fil_array[jj + 2])
                    max_ext_en = (int)fil_array[jj + 2];
                break;

            case 2314885530819572546: // "B31     "
                // for(ii=0; ii<2; ++ii) {
                //   outConnList[ii_Conn+ii]=fil_array[jj+4+ii]-(int64_t)1;
                // }
                // ii_Conn  = ii_Conn  + 2;
                // outTypeList[ii_Elems] = 1;
                ii_Elems = ii_Elems - 1;
                break;

            case 2324217284263039059: // "SPRINGA "
                // for(ii=0; ii<2; ++ii) {
                //   outConnList[ii_Conn+ii]=fil_array[jj+4+ii]-(int64_t)1;
                // }
                // ii_Conn  = ii_Conn  + 2;
                // outTypeList[ii_Elems] = 1;
                ii_Elems = ii_Elems - 1;
                break;

            case 2314885531122807636: // "T3D2    "
                ii_Elems = ii_Elems - 1;
                break;

            default:

                sendError("While reading Elements : Unknown element type '%ld'", (long int)fil_array[jj + 3]);

                char temp[9];
                temp[8] = 0;
                *(int64_t *)temp = fil_array[jj + 3];
                printf("%8s - %ld\n", temp, (long int)fil_array[jj + 3]);

                return FAIL;

                break;
            };

            ii_Elems = ii_Elems + 1;
        };

        jj = jj + fil_array[jj];
    }

    // set up cross reference array for external elem numbers *******************
    // WARNING !! It is assumed that the elem numbers are in accending order
    int *cref_elems;
    cref_elems = (int *)malloc((max_ext_en + 1) * sizeof(int));
    for (ii = 0; ii < max_ext_en + 1; ++ii)
    {
        cref_elems[ii] = -1;
    }
    for (ii = 0; ii < jobhead.no_sup_elems; ++ii)
    {
        cref_elems[ext_en[ii]] = ii;
    }

    // **************************************************************************
    // Load data ****************************************************************
    float *dataList;

    //Allocate Output Port Data Objects *****************************************
    // coDoTensor *outData = new coDoTensor(p_valuesOutPort->getObjName(),
    // 				       numElems, coDoTensor::S3D);
    coDoFloat *data = new coDoFloat(p_eresOutPort->getObjName(),
                                    jobhead.no_sup_elems);

    // if objects were not properly allocated ***********************************
    if (!data->objectOk())
    {
        sendError("Failed to create object '%s' for port '%s'",
                  p_eresOutPort->getObjName(), p_eresOutPort->getName());
        return FAIL;
    }

    data->getAddress(&dataList);

    for (ii = 0; ii < jobhead.no_sup_elems; ii++)
    {
        dataList[ii] = 0.;
    }

    int ii_dat = 0;

    printf("requested result: %s\n", p_elemres->getActLabel());

    const char *activeLabel = p_elemres->getActLabel();

    while ((jj < data_length))
    {

        // Element results ********************************************************
        if ((fil_array[jj + 1] == 1) && (cref_elems[(int)fil_array[jj + 2]] != -1))
        {

            // Check for element averaged result values *****************************
            if (tmp_i[(jj + 3) * 2] != 0)
            {
                sendWarning("This module supports only element averaged result values");
                sendWarning("Element results not loaded *****************************");
                return SUCCESS;
                ;
            }

            ii_dat = (int)fil_array[jj + 2];

            jj = jj + fil_array[jj];

            // stress tensor element *****************************************
            if (fil_array[jj + 1] == 11)
            {

                if (strcmp(activeLabel, sigma[0]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 2]);
                }
                if (strcmp(activeLabel, sigma[1]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 3]);
                }
                if (strcmp(activeLabel, sigma[2]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 4]);
                }
                if (strcmp(activeLabel, sigma[3]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 5]);
                }
                if (strcmp(activeLabel, sigma[4]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 6]);
                }
                if (strcmp(activeLabel, sigma[5]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 7]);
                }

                if (strcmp(activeLabel, equivalence[0]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = sqrt(
                        float(tmp_d[jj + 2]) * float(tmp_d[jj + 2]) + float(tmp_d[jj + 3]) * float(tmp_d[jj + 3]) + float(tmp_d[jj + 4]) * float(tmp_d[jj + 4]) + float(tmp_d[jj + 2]) * float(tmp_d[jj + 3]) + float(tmp_d[jj + 2]) * float(tmp_d[jj + 4]) + float(tmp_d[jj + 3]) * float(tmp_d[jj + 4]) + 3. * (float(tmp_d[jj + 5]) * float(tmp_d[jj + 5]) + float(tmp_d[jj + 6]) * float(tmp_d[jj + 6]) + float(tmp_d[jj + 7]) * float(tmp_d[jj + 7])));
                };
            };

            // Strain Tensor ********************************************************
            if (fil_array[jj + 1] == 21)
            {

                if (strcmp(activeLabel, epsilon[0]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 2]);
                }
                if (strcmp(activeLabel, epsilon[1]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 3]);
                }
                if (strcmp(activeLabel, epsilon[2]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 4]);
                }
                if (strcmp(activeLabel, epsilon[3]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 5]);
                }
                if (strcmp(activeLabel, epsilon[4]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 6]);
                }
                if (strcmp(activeLabel, epsilon[5]) == 0)
                {
                    dataList[cref_elems[ii_dat]] = float(tmp_d[jj + 7]);
                }
            };
        };

        jj = jj + fil_array[jj];
    }

    //===========================================================================

    computeRunning = false;

    return SUCCESS;
}

MODULE_MAIN(IO, ReadABAQUSfil)
