/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadNetCDF.h"

#include <iostream>
#include <netcdfcpp.h>
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"

using namespace covise;

const char *NoneChoices[] = { "none" };
// Lets assume no more than 100 variables per file
int varIds[100];
char *VarDisplayList[100];
char *AxisChoices[100];

// -----------------------------------------------------------------------------
// constructor
// -----------------------------------------------------------------------------
ReadNetCDF::ReadNetCDF(int argc, char *argv[])
    : coModule(argc, argv, "NetCDF Reader")
{
    ncDataFile = NULL;

    // define parameters

    // File browser
    p_fileBrowser = addFileBrowserParam("NC_file", "NC File");
    p_fileBrowser->setValue("/data/hpcsdela/harmonie/fc-0000.nc",
                            "*.nc");

    // Variables to visualise
    for (int i = 0; i < numParams; i++)
    {
        char namebuf[50];
        sprintf(namebuf, "Variable%d", i);
        p_variables[i] = addChoiceParam(namebuf, namebuf);
        p_variables[i]->setValue(1, NoneChoices, 0);
    }

    // Choice of "coordinate variables" for axes
    p_grid_choice_x = addChoiceParam("GridOutX", "Grid x");
    p_grid_choice_x->setValue(1, NoneChoices, 0);
    p_grid_choice_y = addChoiceParam("GridOutY", "Grid y");
    p_grid_choice_y->setValue(1, NoneChoices, 0);
    p_grid_choice_z = addChoiceParam("GridOutZ", "Grid z");
    p_grid_choice_z->setValue(1, NoneChoices, 0);

    // Vertical scale box
    p_verticalScale = addFloatParam("VerticalScale", "VerticalScale");
    p_verticalScale->setValue(1.0);

    // define ports

    // 3D grid
    p_grid_out = addOutputPort("outPort", "StructuredGrid", "Grid output");
    // 2D Surface
    p_surface_out = addOutputPort("surfaceOut", "Polygons", "2D Grid output");

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
ReadNetCDF::~ReadNetCDF()
{
    if (ncDataFile)
    {
        delete ncDataFile;
    }
}

// -----------------------------------------------------------------------------
// change of parameters callback
// -----------------------------------------------------------------------------
void ReadNetCDF::param(const char *paramName, bool inMapLoading)
{
    sendInfo("param callback");

    if (openNcFile())
    {
        // enable parameter menus for selection
        p_grid_choice_x->enable();
        p_grid_choice_y->enable();
        p_grid_choice_z->enable();
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
            { // FIXME: what will we do here?

                // A list of variable names (unaltered)
                /*char* newEntry = new char[50]; 
				strcpy(newEntry,var->name());
				VarChoices[num2d3dVars] = newEntry; // FIXME: Redundant. An int array will do.*/
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
            if (var->num_dims() == 2 || var->num_dims() == 1)
            {
                char *newEntry = new char[50];
                strcpy(newEntry, var->name());
                AxisChoices[num2dVars] = newEntry;
                num2dVars++;
            }
        }

        // Fill axis menus
        p_grid_choice_x->setValue(num2dVars,
                                  AxisChoices, p_grid_choice_x->getValue());
        p_grid_choice_y->setValue(num2dVars,
                                  AxisChoices, p_grid_choice_y->getValue());
        p_grid_choice_z->setValue(num2dVars,
                                  AxisChoices, p_grid_choice_z->getValue());
    }
    else
    {
        // No nc file, disable parameters
        for (int i = 0; i < numParams; i++)
            p_variables[i]->disable();
        p_grid_choice_x->disable();
        p_grid_choice_y->disable();
        p_grid_choice_z->disable();
        p_verticalScale->disable();
    }
}

// -----------------------------------------------------------------------------
// compute callback
// -----------------------------------------------------------------------------
int ReadNetCDF::compute(const char *)
{
    sendInfo("compute call");

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
        // FIXME: For now I choose the last three dimensions. Ok in general??
        // We choose the order of (x,y,z) to match the data
        int nx = 1, ny = edges[numdims - 2], nz = edges[numdims - 1];
        if (numdims >= 3)
            nx = edges[numdims - 3];

        // create a structured grid
        // This allocates memory for the coordinates. (3 "3D" arrays)
        coDoStructuredGrid *outGrid = new coDoStructuredGrid(
            p_grid_out->getObjName(), nx, ny, nz);

        // Get the addresses of the 3 coord arrays (each nz*ny*nx long)
        float *x_coord, *y_coord, *z_coord;
        outGrid->getAddresses(&x_coord, &y_coord, &z_coord);
        // Find the variable corresponding to the users choice for each axis
        NcVar *varX = ncDataFile->get_var(
            AxisChoices[p_grid_choice_x->getValue()]);
        NcVar *varY = ncDataFile->get_var(
            AxisChoices[p_grid_choice_y->getValue()]);
        NcVar *varZ = ncDataFile->get_var(
            AxisChoices[p_grid_choice_z->getValue()]);

        // Read the grid point values from the file
        float *xVals = new float[varX->num_vals()];
        float *yVals = new float[varY->num_vals()];
        float *zVals = new float[varZ->num_vals()];
        varX->get(xVals, varX->edges());
        varY->get(yVals, varY->edges());
        varZ->get(zVals, varZ->edges());

        // Fill the _coord arrays for all grid points (memcpy faster?)
        // FIXME: Should depend on var?->num_dims().
        float scale = p_verticalScale->getValue();
        if(varX->num_vals()==nx)
        {
         int m, n = 0;
         for (int i = 0; i < nx; i++)
         {
             m = 0;
             for (int j = 0; j < ny; j++)
                 for (int k = 0; k < nz; k++, m++, n++)
                 {
                     x_coord[n] = xVals[i];
                     y_coord[n] = yVals[j];
                     z_coord[n] = zVals[k] * scale;
                 }
         }
        }
        else
        {
         int m, n = 0;
         for (int i = 0; i < nx; i++)
         {
             m = 0;
             for (int j = 0; j < ny; j++)
                 for (int k = 0; k < nz; k++, m++, n++)
                 {
                     x_coord[n] = xVals[m];
                     y_coord[n] = yVals[m];
                     z_coord[n] = zVals[i] * scale;
                 }
         }
        }

        // Now for the 2D variables, we create a surface
        int numPolygons = (ny - 1) * (nz - 1);
        coDoPolygons *outSurface = new coDoPolygons(p_surface_out->getObjName(),
                                                    ny * nz, numPolygons * 4, numPolygons);
        int *vl, *pl;
        outSurface->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &pl);
        // Fill the _coord arrays (memcpy faster?)
        // FIXME: Should depend on var?->num_dims(). (fix with pointers?)
        int n = 0;
        n = 0;
        if(varY->num_vals()==ny)
        {
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++, n++)
            {
                x_coord[n] = xVals[j];
                y_coord[n] = yVals[k];
                z_coord[n] = 0; //zVals[0];
            }
        }
        else
        {
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++, n++)
            {
                x_coord[n] = xVals[n];
                y_coord[n] = yVals[n];
                z_coord[n] = 0; //zVals[0];
            }
        }
        // Fill the vertex list
        n = 0;
        for (int j = 1; j < ny; j++)
            for (int k = 1; k < nz; k++)
            {
                vl[n++] = (j - 1) * nz + (k - 1);
                vl[n++] = j * nz + (k - 1);
                vl[n++] = j * nz + k;
                vl[n++] = (j - 1) * nz + k;
            }
        // Fill the polygon list
        for (int p = 0; p < numPolygons; p++)
            pl[p] = p * 4;

        // Delete buffers from grid replication
        delete[] xVals;
        delete[] yVals;
        delete[] zVals;

        // Create output object pointers
        coDoFloat *dataOut[numParams];
        float *floatData;
        // FIXME: Assuming all data to be floats
        // For each output variable, create an output object and fill it with data
        for (int i = 0; i < numParams; i++)
        {
            var = ncDataFile->get_var(varIds[p_variables[i]->getValue()]);
            dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), var->num_vals());
            dataOut[i]->getAddress(&floatData);
            // FIXME: should make sure only to read a 3D subset
            var->get(floatData, var->edges());
        }

        // interaction info for COVER
        coFeedback feedback("ReadNetCDFPlugin");
        feedback.addPara(p_fileBrowser);

        feedback.addPara(p_grid_choice_x);
        feedback.addPara(p_grid_choice_y);
        feedback.addPara(p_grid_choice_z);

        for (int i = 0; i < numParams; ++i)
        {
            feedback.addPara(p_variables[i]);
        }

        // FIXME: check for data first
        //feedback.apply(outGrid);
        feedback.apply(outSurface);
    }

    return CONTINUE_PIPELINE;
}

// -----------------------------------------------------------------------------
// open the nc file
// -----------------------------------------------------------------------------
bool ReadNetCDF::openNcFile()
{
    string sFileName = p_fileBrowser->getValue();

    if (sFileName.empty())
    {
        sendInfo("NetCDF filename is empty!");
        return false;
    }
    else
    {
        ncDataFile = new NcFile(sFileName.c_str(), NcFile::ReadOnly);

        if (!ncDataFile->is_valid())
        {
            sendInfo("Couldn't open NetCDF file!");
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
    covise::coModule *application = new ReadNetCDF(argc, argv);
    application->start(argc, argv);

    return 0;
}

// -----------------------------------------------------------------------------
