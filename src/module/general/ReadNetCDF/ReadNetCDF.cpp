/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadNetCDF.h"

#include <iostream>
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"

using namespace covise;

const char *NoneChoices[] = { "none" };
// Lets assume no more than 100 variables per file
std::vector<std::string> varNames;
char *VarDisplayList[100];
std::vector<std::string> AxisChoices;

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
        int num2d3dVars = 0;
        std::multimap<std::string, NcVar> allVars = ncDataFile->getVars();
        for (const auto& var : allVars)
        {
            if (var.second.getDimCount() >= 0)
            { // FIXME: what will we do here?

                // A list of variable names (unaltered)
                /*char* newEntry = new char[50]; 
				strcpy(newEntry,var->name());
				VarChoices[num2d3dVars] = newEntry; // FIXME: Redundant. An int array will do.*/
                varNames.push_back(var.first);

                // A list of info to display for each variable
                char *dispListEntry = new char[50];
                if (var.second.getDimCount() > 0)
                {
                    sprintf(dispListEntry, "%s (%dD) : [", var.first.c_str(),
                        var.second.getDimCount());
                    for (int j = 0; j < var.second.getDimCount() - 1; j++)
                        sprintf(dispListEntry, "%s %s,", dispListEntry, var.second.getDim(j).getName().c_str());
                    sprintf(dispListEntry, "%s %s ]", dispListEntry,
                        var.second.getDim(var.second.getDimCount() - 1).getName().c_str());
                }
                else
                    sprintf(dispListEntry, "%s (%dD)", var.first.c_str(), var.second.getDimCount());
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
        for (const auto& var : allVars)
        {
            if (var.second.getDimCount() == 2 || var.second.getDimCount() == 1)
            {
                char* newEntry = new char[var.first.length()];
                strcpy(newEntry, var.first.c_str());
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

    if (ncDataFile->getVarCount() > 0)
    {
        // get variable name from choice parameters
        NcVar var;
        std::vector<NcDim> dims;
        int numdims = 0;
        // Find the variable with most dimensions and record its size
        for (int i = 0; i < numParams; i++)
        {
            var = ncDataFile->getVar(varNames[p_variables[i]->getValue()]);
            if (var.getDimCount() > numdims)
            {
                numdims = var.getDimCount();
                dims = var.getDims();
            }
        }
        // FIXME: For now I choose the last three dimensions. Ok in general??
        // We choose the order of (x,y,z) to match the data
        int nx = 1, ny = dims[numdims - 2].getSize(), nz = dims[numdims - 1].getSize();
        if (numdims >= 3)
            nx = dims[numdims - 3].getSize();

        // create a structured grid
        // This allocates memory for the coordinates. (3 "3D" arrays)
        coDoStructuredGrid *outGrid = new coDoStructuredGrid(
            p_grid_out->getObjName(), nx, ny, nz);

        // Get the addresses of the 3 coord arrays (each nz*ny*nx long)
        float *x_coord, *y_coord, *z_coord;
        outGrid->getAddresses(&x_coord, &y_coord, &z_coord);
        // Find the variable corresponding to the users choice for each axis
        NcVar varX = ncDataFile->getVar(
            AxisChoices[p_grid_choice_x->getValue()]);
        NcVar varY = ncDataFile->getVar(
            AxisChoices[p_grid_choice_y->getValue()]);
        NcVar varZ = ncDataFile->getVar(
            AxisChoices[p_grid_choice_z->getValue()]);

        // Read the grid point values from the file
        float *xVals = new float[varX.getDim(0).getSize()];
        float *yVals = new float[varY.getDim(0).getSize()];
        float *zVals = new float[varZ.getDim(0).getSize()];
        varX.getVar(xVals);
        varY.getVar(yVals);
        varZ.getVar(zVals);

        // Fill the _coord arrays for all grid points (memcpy faster?)
        // FIXME: Should depend on var?->num_dims().
        float scale = p_verticalScale->getValue();
		if (varX.getDim(0).getSize() == nx)
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
        if(varY.getDim(0).getSize()==ny)
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
            var = ncDataFile->getVar(varNames[p_variables[i]->getValue()]);
            int size = 0;
            for (int n = 0; n < var.getDimCount(); n++)
            {
                size *= var.getDim(n).getSize();
            }
            dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), size);
            dataOut[i]->getAddress(&floatData);
            // FIXME: should make sure only to read a 3D subset
            var.getVar(floatData);
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
        try {
            ncDataFile = new NcFile(sFileName.c_str(), NcFile::read);
        }
        catch (...)
        {
            sendInfo("Couldn't open WRFChem file!");
            return false;
        }
        return true;
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
