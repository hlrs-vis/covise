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



/*  TODO:
*       add true earth curvature
*/
#include "ReadECMWF.h"

#include <iostream>
#include <netcdfcpp.h>
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"
#include <do/coDoSet.h>

#define PI 3.14159265

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

    const char * coordTypeList[] = {"pressure level","ocean depth"};
    ncDataFile = NULL;

    // File browser
    p_fileBrowser = addFileBrowserParam("NC_file", "NC File");
    p_fileBrowser->setValue("/data/openforecast/testdata_OF.nc",
                            "*.nc");

    // Variables
    for (int i = 0; i < numParams; i++)
    {
        char namebuf[50];
        sprintf(namebuf, "Variable%d", i);
        p_variables[i] = addChoiceParam(namebuf, namebuf);
        p_variables[i]->setValue(1, NoneChoices, 0);
    }
    //coordinate
    p_coord_type = addChoiceParam("CoordType","Coordinate type for grid z component");
    p_coord_type->setValue(2,coordTypeList,1);

    p_grid_lat = addChoiceParam("GridLat", "Latitude");
    p_grid_lat->setValue(1, NoneChoices, 1);
    p_grid_lon = addChoiceParam("GridLon", "Longitude");
    p_grid_lon->setValue(1, NoneChoices, 2);
    p_grid_pressure_level = addChoiceParam("GridPLevel", "Pressure Level");
    p_grid_pressure_level->setValue(1, NoneChoices, 3);
    p_grid_depth = addChoiceParam("GridDepth", "Ocean depth");
    p_grid_depth->setValue(1, NoneChoices, 3);
    p_grid_depth->disable();
    p_grid_depth->hide();

    // Vertical scale
    p_verticalScale = addFloatParam("VerticalScale", "VerticalScale");
    p_verticalScale->setValue(1.0);

    // define grid ports
    p_grid_out = addOutputPort("outPort", "StructuredGrid", "Grid output");
    p_unigrid_out = addOutputPort("uniGridPort", "StructuredGrid", "Uniform grid output");

    p_numTimesteps = addInt32Param("numTimesteps", "number of timesteps");
    p_numTimesteps->setValue(1);

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
        p_coord_type->enable();
        p_grid_lat->enable();
        p_grid_lon->enable();
        p_grid_pressure_level->enable();
        p_grid_depth->enable();
        p_numTimesteps->enable();

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
                    for (int j = 0; j < var->num_dims() - 1; j++)
                        sprintf(dispListEntry, "%s %s,", dispListEntry,
                                var->get_dim(j)->name());
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
        // Fill axis menus
        p_grid_lat->setValue(num2dVars,
                                  AxisChoices, p_grid_lat->getValue());
        p_grid_lon->setValue(num2dVars,
                                  AxisChoices, p_grid_lon->getValue());
        p_grid_pressure_level->setValue(num2dVars,
                                  AxisChoices, p_grid_pressure_level->getValue());
        p_grid_depth->setValue(num2dVars,
                                  AxisChoices, p_grid_depth->getValue());
        switch (p_coord_type->getValue())
        {
        case PRESSURE:
            p_grid_pressure_level->show();
            p_grid_depth->disable();
            p_grid_depth->hide();
            break;
        case DEPTH:
            p_grid_depth->show();
            p_grid_pressure_level->disable();
            p_grid_pressure_level->hide();
            break;

        }
    }
    else
    {
        // No nc file, disable parameters
        for (int i = 0; i < numParams; i++)
            p_variables[i]->disable();
        p_verticalScale->disable();
        p_grid_lat->disable();
        p_grid_lon->disable();
        p_grid_pressure_level->disable();
        p_numTimesteps->disable();
    }
}


float ReadECMWF::pressureAltitude(float p)
{
    return (44.30769396 * (1.0 - std::pow(p/1013.25, 0.190284)));
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
            nz = edges[numdims - 3]; //for 4D data only
        }



        /********************************\
        * THE SPHERICAL GRID
        \********************************/
        coDistributedObject **time_grid;
        coDistributedObject **time_unigrid;
        float *x_coord, *y_coord, *z_coord;
        float scale = p_verticalScale->getValue();

        int numTimesteps = p_numTimesteps->getValue();
        if (numTimesteps < nTime )
            nTime = numTimesteps;

        // Find the variable corresponding to the users choice for each axis
        NcVar *varLat = ncDataFile->get_var(
            AxisChoices[p_grid_lat->getValue()]);
        NcVar *varLon = ncDataFile->get_var(
            AxisChoices[p_grid_lon->getValue()]);
        NcVar *varPLevel;
        if (p_coord_type->getValue() == PRESSURE)
        {
            varPLevel = ncDataFile->get_var(AxisChoices[p_grid_pressure_level->getValue()]);
        }else
        {
            varPLevel = ncDataFile->get_var(AxisChoices[p_grid_depth->getValue()]);
        }


        if (has_timesteps > 0)
        {

            //
            float *xVals = new float[varLat->num_vals()];
            float *yVals = new float[varLon->num_vals()];
            varLat->get(xVals,varLat->edges());
            varLon->get(yVals,varLon->edges());
            float *zVals;
            if (nz<=1)
            {
                zVals = new float[1];
                zVals[0] = 0;
            }else{
                zVals = new float[varPLevel->num_vals()];
                varPLevel->get(zVals, varPLevel->edges());
            }


            time_grid = new coDistributedObject *[nTime + 1];
            time_grid[nTime] = NULL;
            for (int t = 0; t < nTime; ++t)
            {
                sprintf(buf, "%s_%d", p_grid_out->getObjName(), t);
                coDoStructuredGrid *outGrid = new coDoStructuredGrid(buf, nz, nx, ny);
                outGrid->getAddresses(&z_coord, &x_coord, &y_coord);
                time_grid[t] = outGrid;


                int n = 0;
                float A = 6380; //actually N(lat)

                switch (p_coord_type->getValue())
                {
                case PRESSURE:
                    //conversion to ECEF coordinates

                    for (int k = 0; k < nz; k++)
                    {
                         A = (6380 + scale*pressureAltitude(zVals[k]))*1000;
                        for (int i = 0; i < nx; i++)
                        {
                            for (int j = 0; j < ny; j++, n++)
                             {

                                         x_coord[n] = A*cos(xVals[i]*PI/180)*cos(yVals[j]*PI/180);
                                         y_coord[n] = A*cos(xVals[i]*PI/180)*sin(yVals[j]*PI/180);
                                         z_coord[n] = A*sin(xVals[i]*PI/180);
                              }
                         }
                    }
                    break;
                case DEPTH:
                    //conversion to ECEF coordinates
                    for (int k = 0; k < nz; k++)
                    {
                        A = (6380*1000 - scale*zVals[k]);
                        for (int i = 0; i < nx; i++)
                        {
                            for (int j = 0; j < ny; j++, n++)
                             {

                                         x_coord[n] = A*cos(xVals[i*ny+j]*PI/180)*cos(yVals[i*ny+j]*PI/180);
                                         y_coord[n] = A*cos(xVals[i*ny+j]*PI/180)*sin(yVals[i*ny+j]*PI/180);
                                         z_coord[n] = A*sin(xVals[i*ny+j]*PI/180);
                              }
                         }
                    }
                    break;
                }


            }

            coDoSet *time_outGrid = new coDoSet(p_grid_out->getObjName(), time_grid);
            sprintf(buf, "1 %d", nTime);
            time_outGrid->addAttribute("TIMESTEP", buf);
            p_grid_out->setCurrentObject(time_outGrid);

            delete [] time_grid;
            delete[] xVals;
            delete[] yVals;
            delete[] zVals;

        }else {

            coDoStructuredGrid *outGrid = new coDoStructuredGrid(p_grid_out->getObjName(), nz, nx, ny);

            // Get the addresses of the 3 coord arrays (each nz*ny*nx long)
            outGrid->getAddresses(&z_coord, &x_coord, &y_coord);
            float *xVals= new float[varLat->num_vals()];
            float *yVals= new float[varLon->num_vals()];
            float *zVals = new float[varPLevel->num_vals()];
            varLat->get(xVals, varLat->edges());
            varLon->get(yVals, varLon->edges());
            varPLevel->get(zVals, varPLevel->edges());
            float A = 6380; //actually N(lat)
            int n = 0;
          /*  for (int i = 0; i < nx; i++)
            {
                int m = 0;
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++, m++, n++)
                    {
                        x_coord[n] = A*cos(xVals[n])*cos(yVals[n]);
                        y_coord[n] = A*cos(xVals[n])*sin(yVals[n]);
                        z_coord[n] = A*sin(xVals[n]);
                    }
            }*/
            for (int k = 0; k < nz; k++)
            {
                 A = (6380 + scale*(zVals[k]))*1000;//(6380 + scale*pressureAltitude(zVals[k]))*1000;
                for (int i = 0; i < nx; i++)
                {
                    for (int j = 0; j < ny; j++, n++)
                     {

                                 x_coord[n] = A*cos(xVals[i]*PI/180)*cos(yVals[j]*PI/180);
                                 y_coord[n] = A*cos(xVals[i]*PI/180)*sin(yVals[j]*PI/180);
                                 z_coord[n] = A*sin(xVals[i]*PI/180);
                      }
                 }
            }
            delete [] xVals;
            delete [] yVals;
            delete [] zVals;
       }



        /*************************\
        // THE UNIFORM (flat) GRID
        \*************************/

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

        /******************\
        // VARIABLES
        \******************/
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
                //check that there is indeed timesteps
                for (int d = 0; d < var->num_dims(); ++d)
                {
                    NcDim *dim = var->get_dim(d);
                    if((strcmp(dim->name(), "time") == 0)||(strcmp(dim->name(),"time_counter") == 0))
                    {
                        time_dependent = true;
                        if (nTime > dim->size())
                         {

                            nTime = dim->size();
                            sendInfo("Max. available time steps is %d", nTime);
                        }

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
                    sendInfo("Could not find timesteps for selected variable");
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
