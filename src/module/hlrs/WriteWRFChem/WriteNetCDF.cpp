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
        try
        {
       ncOutFile = new NcFile(filePath.c_str(), NcFile::write, NcFile::classic64);
        }
        catch (...)
        {
            sendError("failed to open %s", filePath.c_str());
            return STOP_PIPELINE;
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


    NcDim dimTime = ncOutFile->addDim("Time",1); //TODO: if more than 1 -> handle coDoSet
    NcDim dimSN = ncOutFile->addDim("south_north", nz);
    NcDim dimEW = ncOutFile->addDim("east_west", ny);
    NcDim dimBT = ncOutFile->addDim("bottom_top", nx);

    NcVar var[numVars];
    for (int i = 0; i < numVars; ++i) {
        if (varUsed[i] > 0)
        {
            sprintf(buf, "%s", p_varName[i]->getValue());
            std::vector<NcDim> dims{ dimTime, dimSN, dimEW };
            var[i] = ncOutFile->addVar(buf, ncFloat, dims);
        }
    }

    //Note: no global attributes added so far

    for (int i = 0; i < numVars; ++i) {
        if (varUsed[i] > 0)
        {
            var[i].putVar(val[i]);
        }

    }
     ncOutFile->sync();

     return CONTINUE_PIPELINE;
}


MODULE_MAIN(IO, WriteNetCDF)
