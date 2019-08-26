/******************************************************************
 *
 *    READ WRF CHEM
 *
 *
 *  Description: Read NetCDF-like files from WRF-Chem Simulations
 *  Date: 04.06.19
 *  Author: Leyla Kern
 *
 *******************************************************************/



#include "ReadECMWF.h"

#include <iostream>
#include <netcdfcpp.h>
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"
#include <do/coDoSet.h>

using namespace covise;

const char *NoneChoices[] = { "none" };
// Lets assume no more than 100 variables per file
int varIds[100];
char *VarDisplayList[100];
char *AxisChoices[100];

// -----------------------------------------------------------------------------
// constructor
// -----------------------------------------------------------------------------
ReadECMWF::ReadECMWF(int argc, char *argv[])
    : coModule(argc, argv, "ECMWF Reader")
{
    ncDataFile = NULL;

    // define parameters

    // File browser
    p_fileBrowser = addFileBrowserParam("NC_file", "NC File");
    p_fileBrowser->setValue("/data/openforecast/testdata_OF.nc",
                            "*.nc");

    // Variables to visualise
    for (int i = 0; i < numParams; i++)
    {
        char namebuf[50];
        sprintf(namebuf, "Variable%d", i);
        p_variables[i] = addChoiceParam(namebuf, namebuf);
        p_variables[i]->setValue(1, NoneChoices, 0);
    }

    // Vertical scale box
    p_verticalScale = addFloatParam("VerticalScale", "VerticalScale");
    p_verticalScale->setValue(1.0);

    // define ports

    p_unigrid_out = addOutputPort("uniGridPort", "StructuredGrid", "Uniform grid output");

    // Data ports
    for (int i = 0; i < numParams; i++)
    {
        char namebuf[50];
        sprintf(namebuf, "dataOutPort%d", i);
        p_data_outs[i] = addOutputPort(namebuf, "Float", namebuf);
    }
}

// -----------------------------------------------------------------------------
// destructor
// -----------------------------------------------------------------------------
ReadECMWF::~ReadECMWF()
{
    if (ncDataFile)
    {
        delete ncDataFile;
    }
}

// -----------------------------------------------------------------------------
// change of parameters callback
// -----------------------------------------------------------------------------
void ReadECMWF::param(const char *paramName, bool inMapLoading)
{
    sendInfo("param callback");

    if (openNcFile())
    {
        // enable parameter menus for selection
        p_verticalScale->enable();
        for (int i = 0; i < numParams; i++)
            p_variables[i]->enable();

        // Create "Variable" menu entries for all 2D and 3D variables
        NcVar *var;
        int num2d3dVars = 0;
        for (int i = 0; i < ncDataFile->num_vars(); i++)
        {
            var = ncDataFile->get_var(i);
            if (var->num_dims() >= 0)
            {
                varIds[num2d3dVars] = i;

                // A list of info to display for each variable
                char *dispListEntry = new char[50];
                if (var->num_dims() > 0)
                {
                    sprintf(dispListEntry, "%s (%dD) : [", var->name(),
                            var->num_dims());
                    for (int i = 0; i < var->num_dims() - 1; i++)
                        sprintf(dispListEntry, "%s %s,", dispListEntry,
                                var->get_dim(i)->name());
                    sprintf(dispListEntry, "%s %s ]", dispListEntry,
                            var->get_dim(var->num_dims() - 1)->name());
                }
                else
                    sprintf(dispListEntry, "%s (%dD)", var->name(), var->num_dims());
                VarDisplayList[num2d3dVars] = dispListEntry;

                num2d3dVars++;
            }
        }
        // Fill the menu. Default selection = last selected. (#1 initially).
        for (int i = 0; i < numParams; i++)
            p_variables[i]->setValue(num2d3dVars, VarDisplayList,
                                     p_variables[i]->getValue());

        // Create "Axis" menu entries for 2D variables only
        int num2dVars = 0;
        for (int i = 0; i < ncDataFile->num_vars(); ++i)
        {
            var = ncDataFile->get_var(i);
            if (var->num_dims() > 0)
            {
                char *newEntry = new char[50];
                strcpy(newEntry, var->name());
                AxisChoices[num2dVars] = newEntry;
                num2dVars++;
            }
        }


    }
    else
    {
        // No nc file, disable parameters
        for (int i = 0; i < numParams; i++)
            p_variables[i]->disable();
        p_verticalScale->disable();
    }
}

// -----------------------------------------------------------------------------
// compute callback
// -----------------------------------------------------------------------------
int ReadECMWF::compute(const char *)
{
    char buf[128];
    sendInfo("compute call");
    has_timesteps = 0;

    if (ncDataFile->is_valid())
    {
        // get variable name from choice parameters
        NcVar *var;
        long *edges;
        int numdims = 0;
        // Find the variable with most dimensions and record its size
        for (int i = 0; i < numParams; i++)
        {
            var = ncDataFile->get_var(varIds[p_variables[i]->getValue()]);
            if (var->num_dims() > numdims)
            {
                numdims = var->num_dims();
                edges = var->edges();
                printf("%s is %ld", var->name(), edges[0]);
                for (int j = 1; j < numdims; j++)
                    printf(" x %ld", edges[j]);
                printf("\n");
            }
        }


        if (edges[0] > 1)
        {
            has_timesteps = 1;
        }

        sendInfo("Found %ld time steps\n", edges[0]);

        int nx = edges[numdims - 2], ny = edges[numdims - 1], nz = 1, nTime = edges[0];
        if (has_timesteps > 0)
        {
            if (numdims > 3)
            {
                nz = edges[numdims - 3];
            }
        }else if (numdims >=3){
           // nz = edges[numdims - 3]; //for 4D data only
        }

        // THE GRID
        coDistributedObject **time_grid;
        coDistributedObject **time_unigrid;
        //float *x_coord, *y_coord, *z_coord;

        float scale = p_verticalScale->getValue();
        int m, n = 0;

        float *x_unicoord, *y_unicoord, *z_unicoord;
        if (has_timesteps > 0)
        {
            time_unigrid = new coDistributedObject *[nTime + 1];
            time_unigrid[nTime] = NULL;

            for (int t = 0; t < nTime; ++t)
            {
                sprintf(buf, "%s_%d", p_unigrid_out->getObjName(), t);
                coDoStructuredGrid *outUniGrid = new coDoStructuredGrid(buf, nz, nx, ny);
                time_unigrid[t] = outUniGrid;
                outUniGrid->getAddresses(&z_unicoord, &x_unicoord, &y_unicoord);

                n = 0;
                for (int k = 0; k < nz; k++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        for (int j = 0; j < ny; j++, n++)
                         {

                                     x_unicoord[n] = i;
                                     y_unicoord[n] = j;
                                     z_unicoord[n] = k;
                          }
                     }
                }


            }
            coDoSet *time_outUniGrid = new coDoSet(p_unigrid_out->getObjName(), time_unigrid);
            sprintf(buf, "1 %d", nTime);
            time_outUniGrid->addAttribute("TIMESTEP", buf);

            delete [] time_unigrid;

        }else
        {
            coDoStructuredGrid *outUniGrid = new coDoStructuredGrid(
                p_unigrid_out->getObjName(), nx, ny, nz);

            outUniGrid->getAddresses(&x_unicoord, &y_unicoord, &z_unicoord);

            n = 0;

            for (int i = 0; i < nx; i++)
            {
                m = 0;
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++, m++, n++)
                    {
                        x_unicoord[n] = i;
                        y_unicoord[n] = j;
                        z_unicoord[n] = k;
                    }
            }
        }


        // Variables
        coDistributedObject **time_data;
        if (has_timesteps > 0)
        {
            time_data = new coDistributedObject *[nTime + 1];
            time_data[nTime] = NULL;

            coDoFloat *dataOut[numParams];
            bool time_dependent = false;
            int num_vals;

            for (int i = 0; i < numParams; ++i)
            {
                time_dependent = false;
                num_vals = 1;
                var = ncDataFile->get_var(varIds[p_variables[i]->getValue()]);
                long* cur = var->edges();

                for (int d = 0; d < var->num_dims(); ++d)
                {
                    NcDim *dim = var->get_dim(d);
                    if(strcmp(dim->name(), "time") == 0)
                    {
                        time_dependent = true;
                    }else {
                         num_vals = num_vals * dim->size();
                    }
                    cur[d] = 0;
                }

                if(time_dependent)
                {
                    long* edges_red = var->edges();
                    edges_red[0] = 1;

                    for(int t = 0; t < nTime; ++t)
                    {
                        float *floatData;
                        cur[0]=t;
                        sprintf(buf, "%s_%d", p_data_outs[i]->getObjName(), t);
                        coDoFloat *outdata = new coDoFloat(buf, num_vals);
                        time_data[t] = outdata;
                        outdata->getAddress(&floatData);
                        var->set_cur(cur);
                        var->get(floatData, edges_red);
                    }

                    coDoSet *time_outData = new coDoSet(p_data_outs[i]->getObjName(), time_data);
                    sprintf(buf, "1 %d", nTime);
                    time_outData->addAttribute("TIMESTEP", buf);

                }else
                {
                    //static
                    float *floatData;
                    var = ncDataFile->get_var(varIds[p_variables[i]->getValue()]);
                    dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), num_vals);
                    dataOut[i]->getAddress(&floatData);
                    var->get(floatData, var->edges());
                }



            }
            delete [] time_data;

        }else {

            coDoFloat *dataOut[numParams];
            float *floatData;
            for (int i = 0; i < numParams; i++)
            {
                var = ncDataFile->get_var(varIds[p_variables[i]->getValue()]);
                dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), var->num_vals());
                dataOut[i]->getAddress(&floatData);
                var->get(floatData, var->edges());
            }
        }

        // interaction info for COVER
        coFeedback feedback("ReadECMWFPlugin");
        feedback.addPara(p_fileBrowser);

        for (int i = 0; i < numParams; ++i)
        {
            feedback.addPara(p_variables[i]);
        }

    }

    return CONTINUE_PIPELINE;
}

// -----------------------------------------------------------------------------
// open the nc file
// -----------------------------------------------------------------------------
bool ReadECMWF::openNcFile()
{
    string sFileName = p_fileBrowser->getValue();

    if (sFileName.empty())
    {
        sendInfo("ECMWF filename is empty!");
        return false;
    }
    else
    {
        ncDataFile = new NcFile(sFileName.c_str(), NcFile::ReadOnly);

        if (!ncDataFile->is_valid())
        {
            sendInfo("Couldn't open ECMWF file!");
            return false;
        }
        else
        {
            return true;
        }
    }
}

// -----------------------------------------------------------------------------
// main: create and start the module
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    covise::coModule *application = new ReadECMWF(argc, argv);
    application->start(argc, argv);

    return 0;
}
