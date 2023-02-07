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
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"
#include <do/coDoSet.h>

#define PI 3.14159265

using namespace covise;

const char *NoneChoices[] = { "none" };
// Lets assume no more than 100 variables per file
std::vector<std::string> varNames;
char* VarDisplayList[100];
std::vector<std::string> AxisChoices;

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
        //NcVar *var;
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
                char* dispListEntry = new char[50];
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
    return (float)((44.30769396 * (1.0 - std::pow(p/1013.25, 0.190284))));
}

// -----------------------------------------------------------------------------
// compute callback
// -----------------------------------------------------------------------------
int ReadECMWF::compute(const char *)
{
    char buf[128];
    sendInfo("compute call");
    has_timesteps = 0;

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


        if (dims[0].getSize() > 1)
        {
            has_timesteps = 1;
        }

        sendInfo("Found %ld time steps\n", dims[0].getSize());

        size_t nx = dims[numdims - 2].getSize(), ny = dims[numdims - 1].getSize(), nz = 1, nTime = dims[0].getSize();
        if (has_timesteps > 0)
        {
            if (numdims > 3)
            {
                nz = dims[numdims - 3].getSize();
            }
        }else if (numdims >=3){
            nz = dims[numdims - 3].getSize(); //for 4D data only
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
        NcVar varLat = ncDataFile->getVar(
            AxisChoices[p_grid_lat->getValue()]);
        NcVar varLon = ncDataFile->getVar(
            AxisChoices[p_grid_lon->getValue()]);
        NcVar varPLevel;
        if (p_coord_type->getValue() == PRESSURE)
        {
            varPLevel = ncDataFile->getVar(AxisChoices[p_grid_pressure_level->getValue()]);
        }else
        {
            varPLevel = ncDataFile->getVar(AxisChoices[p_grid_depth->getValue()]);
        }


        if (has_timesteps > 0)
        {

            //
            float *xVals = new float[varLat.getDim(0).getSize()];
            float *yVals = new float[varLon.getDim(0).getSize()];
            varLat.getVar(xVals);
            varLon.getVar(yVals);
            float *zVals;
            if (nz<=1)
            {
                zVals = new float[1];
                zVals[0] = 0;
            }else{
                zVals = new float[varPLevel.getDim(0).getSize()];
                varPLevel.getVar(zVals);
            }


            time_grid = new coDistributedObject *[nTime + 1];
            time_grid[nTime] = NULL;
            for (int t = 0; t < nTime; ++t)
            {
                sprintf(buf, "%s_%d", p_grid_out->getObjName(), t);
                coDoStructuredGrid *outGrid = new coDoStructuredGrid(buf, (int)nz, (int)nx, (int)ny);
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

                                         x_coord[n] = (float)(A*cos(xVals[i]*PI/180)*cos(yVals[j]*PI/180));
                                         y_coord[n] = (float)(A*cos(xVals[i]*PI/180)*sin(yVals[j]*PI/180));
                                         z_coord[n] = (float)(A*sin(xVals[i]*PI/180));
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

                                         x_coord[n] = (float)(A*cos(xVals[i*ny+j]*PI/180)*cos(yVals[i*ny+j]*PI/180));
                                         y_coord[n] = (float)(A*cos(xVals[i*ny+j]*PI/180)*sin(yVals[i*ny+j]*PI/180));
                                         z_coord[n] = (float)(A*sin(xVals[i*ny+j]*PI/180));
                              }
                         }
                    }
                    break;
                }


            }

            coDoSet *time_outGrid = new coDoSet(p_grid_out->getObjName(), time_grid);
            sprintf(buf, "1 %zd", nTime);
            time_outGrid->addAttribute("TIMESTEP", buf);
            p_grid_out->setCurrentObject(time_outGrid);

            delete [] time_grid;
            delete[] xVals;
            delete[] yVals;
            delete[] zVals;

        }else {

            coDoStructuredGrid *outGrid = new coDoStructuredGrid(p_grid_out->getObjName(), (int)nz, (int)nx, (int)ny);

            // Get the addresses of the 3 coord arrays (each nz*ny*nx long)
            outGrid->getAddresses(&z_coord, &x_coord, &y_coord);
            float *xVals= new float[varLat.getDim(0).getSize()];
            float *yVals= new float[varLon.getDim(0).getSize()];
            float *zVals = new float[varPLevel.getDim(0).getSize()];
            varLat.getVar(xVals);
            varLon.getVar(yVals);
            varPLevel.getVar(zVals);
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

                                 x_coord[n] = (float)(A*cos(xVals[i]*PI/180)*cos(yVals[j]*PI/180));
                                 y_coord[n] = (float)(A*cos(xVals[i]*PI/180)*sin(yVals[j]*PI/180));
                                 z_coord[n] = (float)(A*sin(xVals[i]*PI/180));
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
                coDoStructuredGrid *outUniGrid = new coDoStructuredGrid(buf, (int)nz, (int)nx, (int)ny);
                time_unigrid[t] = outUniGrid;
                outUniGrid->getAddresses(&z_unicoord, &x_unicoord, &y_unicoord);

                n = 0;
                for (int k = 0; k < nz; k++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        for (int j = 0; j < ny; j++, n++)
                         {

                                     x_unicoord[n] = float(i);
                                     y_unicoord[n] = float(j);
                                     z_unicoord[n] = float(k);
                          }
                     }
                }


            }
            coDoSet *time_outUniGrid = new coDoSet(p_unigrid_out->getObjName(), time_unigrid);
            sprintf(buf, "1 %zd", nTime);
            time_outUniGrid->addAttribute("TIMESTEP", buf);

            delete [] time_unigrid;

        }else
        {
            coDoStructuredGrid *outUniGrid = new coDoStructuredGrid(
                p_unigrid_out->getObjName(), (int)nx, (int)ny, (int)nz);

            outUniGrid->getAddresses(&x_unicoord, &y_unicoord, &z_unicoord);

            n = 0;

            for (int i = 0; i < nx; i++)
            {
                m = 0;
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++, m++, n++)
                    {
                        x_unicoord[n] = float(i);
                        y_unicoord[n] = float(j);
                        z_unicoord[n] = float(k);
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
                var = ncDataFile->getVar(varNames[p_variables[i]->getValue()]);
                std::vector<size_t> start = { 0,0,0,0 };
                std::vector<size_t> size = { 0,0,0,0 };
                //check that there is indeed timesteps
                for (int d = 0; d < var.getDimCount(); ++d)
                {
                    NcDim dim = var.getDim(d);
                    size[d] = (int)dim.getSize();
                    start[d] = 0;
                    if(dim.getName()=="time"|| dim.getName() == "time_counter")
                    {
                        time_dependent = true;
                        if (nTime > (int)dim.getSize())
                         {

                            nTime = (int)dim.getSize();
                            sendInfo("Max. available time steps is %zu", nTime);
                        }

                    }else {
                         num_vals = (int)(num_vals * dim.getSize());
                    }
                }

                if(time_dependent)
                {

                    for(int t = 0; t < nTime; ++t)
                    {
                        float *floatData;
                        sprintf(buf, "%s_%d", p_data_outs[i]->getObjName(), t);
                        coDoFloat *outdata = new coDoFloat(buf, num_vals);
                        time_data[t] = outdata;
                        outdata->getAddress(&floatData);
                        start[0] = t;
                        size[0] = 1;
                        var.getVar(start,size,floatData);
                    }

                    coDoSet *time_outData = new coDoSet(p_data_outs[i]->getObjName(), time_data);
                    sprintf(buf, "1 %zd", nTime);
                    time_outData->addAttribute("TIMESTEP", buf);

                }else
                {
                    //static
                    sendInfo("Could not find timesteps for selected variable");
                    float *floatData;
                    var = ncDataFile->getVar(varNames[p_variables[i]->getValue()]);
                    dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), num_vals);
                    dataOut[i]->getAddress(&floatData);
                    var.getVar(floatData);
                }



            }
            delete [] time_data;

        }else {

            coDoFloat *dataOut[numParams];
            float *floatData;
            for (int i = 0; i < numParams; i++)
            {
                var = ncDataFile->getVar(varNames[p_variables[i]->getValue()]);
                dataOut[i] = new coDoFloat(p_data_outs[i]->getObjName(), int(nx*ny*nz));
                dataOut[i]->getAddress(&floatData);
                var.getVar(floatData);
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
    covise::coModule *application = new ReadECMWF(argc, argv);
    application->start(argc, argv);

    return 0;
}
