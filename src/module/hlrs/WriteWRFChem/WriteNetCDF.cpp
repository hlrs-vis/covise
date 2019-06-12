/******************************************************************\
 *
 *  WriteNetCDF
 *
 *  Description: Write data to NC file
 *  Date: 02.06.19
 *  Author: Leyla Kern
 *
 *
 \*****************************************************************/


// TODO: user defined number of variables in file
//       with user defined names
//       Timesteps

#include "WriteNetCDF.h"
#include <iostream>
#include <do/coDoStructuredGrid.h>
#include <do/coDoData.h>
#include <netcdfcpp.h>


WriteNetCDF::WriteNetCDF(int argc, char *argv[])
    :coSimpleModule(argc, argv, " Write data to NetCDF file")
{

    p_gridIn = addInputPort("GridIn","StructuredGrid","structured grid");
    // p_gridIn->setRequired(0); //only for testing

    p_fileName = addFileBrowserParam("FileName","name for the generated file");
    p_fileName->setValue(".","*.nc/*");

    char buf[20];
    for (int i = 0; i < numVars; ++i) {
        sprintf(buf,"nameVar%d",i);
        p_varName[i] = addStringParam(buf, "Name for variable");
        sprintf(buf,"var%d",i);
        p_varName[i]->setValue(buf);

        sprintf(buf,"dataIn%d",i);
        p_dataIn[i] = addInputPort(buf,"Float","data values matching the grid");
        p_dataIn[i]->setRequired(0);

    }


    ncOutFile = NULL;
}


WriteNetCDF::~WriteNetCDF()
{

}

int WriteNetCDF::compute(const char *)
{
    int nx, ny, nz;
    float *val[numVars];
    char buf[32];
    int numVals[numVars];

    string filePath = p_fileName->getValue();
    if (filePath.empty())
    {
        sendError("no file given");
        return STOP_PIPELINE;
    }else
    {
       ncOutFile = new NcFile(filePath.c_str(), NcFile::New, NULL, 0, NcFile::Offset64Bits);
        if (!ncOutFile->is_valid())
        {
            sendError("failed to create new file");
        }
    }

    const coDistributedObject *gridIn = p_gridIn->getCurrentObject();
    if (!gridIn)
    {
        sendError("Cannot read input grid");
        return STOP_PIPELINE;
    }
    const coDoStructuredGrid *grid = dynamic_cast<const coDoStructuredGrid *>(gridIn);
    grid->getAddresses(&x_c, &y_c, &z_c);
    grid->getGridSize(&nx, &ny, &nz);

    int varUsed[numVars];
    for (int i = 0; i < numVars; ++i) {
        const coDistributedObject *dataIn = p_dataIn[i]->getCurrentObject();
        if (dataIn)
        {
            varUsed[i] = 1;
            const coDoFloat *data = dynamic_cast<const coDoFloat *>(dataIn);
            data->getAddress(&val[i]);

            numVals[i] = data->getNumPoints();
        }else {
            varUsed[i] = 0;
        }
        //delete [] data;
    }


    NcDim *dimTime = ncOutFile->add_dim("Time",1); //TODO: if more than 1 -> handle coDoSet
    NcDim *dimSN = ncOutFile->add_dim("south_north", nz);
    NcDim *dimEW = ncOutFile->add_dim("east_west", ny);
    NcDim *dimBT = ncOutFile->add_dim("bottom_top", nx);

    NcVar *var[numVars];
    for (int i = 0; i < numVars; ++i) {
        if (varUsed[i] > 0)
        {
            sprintf(buf, "%s", p_varName[i]->getValue());
            var[i] = ncOutFile->add_var(buf, ncFloat, dimTime, dimSN, dimEW);
        }
    }

    //Note: no global attributes added so far

    NcBool putOK[numVars];
    for (int i = 0; i < numVars; ++i) {
        if (varUsed[i] > 0)
        {
            putOK[i] = var[i]->put(val[i], 1, nz, ny);

            if (!putOK[i])
            {
                sendInfo("WriteNetCDF: Failed to write values for variable %d", i);
            }else
            {
                sendInfo("Writing variable %d done", i);
                var[i]->sync();

            }
        }

    }
     ncOutFile->sync();

     return CONTINUE_PIPELINE;
}


MODULE_MAIN(IO, WriteNetCDF)
