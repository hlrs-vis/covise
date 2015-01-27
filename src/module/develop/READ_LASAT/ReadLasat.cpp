/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   ReadLasat
//
// Description:
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: ReadLasat.cpp,v 1.3 2002/12/17 13:36:05 ralf Exp $
//

#include <api/coModule.h>
#include "ReadLasat.h"
#include <fstream>
using namespace std;
#include "HouseFile.h"
#include "DmnaFiles.h"
ReadLasat::ReadLasat(const char *houseFile, const char *dmnaFile, const char *zFile)
{
    init();
    p_houseFile->setValue(houseFile, "*.*");
    p_dmnaFile->setValue(dmnaFile, "*.*");
    p_zFile->setValue(zFile, "*.*");
}

ReadLasat::ReadLasat()
{
    init();
}

void ReadLasat::init()
{

    /****************************************
     const char *ChoiseVal[] = {"Nodal_Results", "Element_Results",};
     strcpy(init_path, "~/");

     //parameters
     p_gridpath = addFileBrowserParam("grid_path","Neutral File path");
     p_gridpath->setValue(init_path,"*");
     //p_gridpath->setImmediate(1);
     p_displpath = addFileBrowserParam("nodal_displ_force_path","Nodal Displacement File path");
     p_displpath->setValue(init_path,"*");
     //p_displpath->setImmediate(1);
   p_nshpath = addFileBrowserParam("nodal_result_path","Nodal Results File path");
   p_nshpath->setValue(init_path,"*");
   //p_nshpath->setImmediate(1);
   p_elempath = addFileBrowserParam("element_result_path","Element Results File path");
   p_elempath->setValue(init_path,"*");
   p_option = addChoiceParam("Option","open or closed cylinder");
   p_option->setValue(2,ChoiseVal,1);
   p_timesteps = addInt32Param("timesteps","timesteps");
   p_timesteps->setValue(1);
   p_skip = addInt32Param("skipped_files","number of skip files for each timestep");
   p_skip->setValue(0);
   p_columns = addInt32Param("nb_columns","number of column in the result file");
   p_columns->setValue(1);
   *******************************/
    zValues = NULL;
    p_houseFile = addFileBrowserParam("house_path", "House File path");
    p_dmnaFile = addFileBrowserParam("dmna_path", "Dna File path");
    p_zFile = addFileBrowserParam("z_path", "Z coordinates File path");
    //ports

    p_grid = addOutputPort("grid", "coDoRectilinearGrid", "Mesh output");
    p_data = addOutputPort("data", "coDoFloat ", "Scalar Data output");
    p_house = addOutputPort("houses", "coDoPolygons", "Houses");

    //private data
    houseFile = NULL;
}

void ReadLasat::param(const char *paramName)
{
}

bool ReadLasat::readZFile(const char *zFile, char *&errMsg)
{
    char buf[MAXLINE];
    ifstream input(zFile, ios::in);
    if (!input)
    {
        errMsg = new char[1024];
        sprintf(errMsg, "file %s could not be opened", zFile);
        return false;
    }
    zSize = 0;
    float value;
    while (!input.eof())
    {
        input.getline(buf, MAXLINE);
        if (!input.eof())
        {
            value = atof(buf);
#ifdef DEBUG
            cerr << "value:" << zSize << " " << value << endl;
#endif
            zSize++;
        }
    }
    input.close();
    zValues = new float[zSize];
    input.open(zFile, ios::in);
    int i = 0;
    while (!input.eof())
    {
        input.getline(buf, MAXLINE);
        if (!input.eof())
        {
            zValues[i] = atof(buf);
#ifdef DEBUG
            cerr << "zValues[" << i << "]" << zValues[i] << endl;
#endif
            i++;
        }
    }
    return true;
}

int ReadLasat::compute()
{

    //First we read in the z-Coordiantes from a separate file
    delete zValues;
    char *errMsg;
#if 1
    const char *zPath = p_zFile->getValue();
    bool correct = readZFile(zPath, errMsg);
    if (!correct)
    {
        sendError(errMsg);
        return FAIL;
    }

    const char *housePath = p_houseFile->getValue();
#else
    const char *zPath = "/data/Kunden/VW/Umweltplanung/zCoordinates";
    bool correct = readZFile(zPath, errMsg);
    if (!correct)
    {
        sendError(errMsg);
        return FAIL;
    }
    const char *housePath = "/data/Kunden/VW/Umweltplanung//gebdaten.dat.orig";
#endif
//   int fd = ::open(housePath,O_RDONLY);

//Reading the coordinates of the building
#if 1
    houseFile = new HouseFile(housePath, zValues[0], p_house->getObjName());
#else
    houseFile = new HouseFile(housePath, zValues[0], "houses");
#endif
    if ((!houseFile) || (!houseFile->isValid()))
    {
        sendError("Could not read %s as ReadLasat File", "test");
        if (houseFile)
            delete houseFile;
        houseFile = NULL;
        return FAIL;
    }

    //Reading the grid data
    p_house->setCurrentObject(houseFile->getPolygon());
#if 1
    const char *dmnaPath = p_dmnaFile->getValue();
#else
    const char *dmnaPath = "/data/Kunden/VW/Umweltplanung/01.dmna";
#endif
    dmnaFile = new DmnaFiles(dmnaPath, zPath, p_data->getObjName(), errMsg);
    if ((!dmnaFile) || (!dmnaFile->isValid()))
    {
        sendError(errMsg);
        if (dmnaFile)
            delete dmnaFile;
        dmnaFile = NULL;
        return FAIL;
    }
    //constructing the result data
    p_data->setCurrentObject(dmnaFile->getData());
    ySize = dmnaFile->getYDim();
    xSize = dmnaFile->getXDim();
    coDoRectilinearGrid *rgrid = new coDoRectilinearGrid(p_grid->getObjName(), xSize, ySize, zSize);
    float *x, *y, *z;
    rgrid->getAddresses(&x, &y, &z);
    int i;
    float delta = dmnaFile->getDelta();
    float yMin = dmnaFile->getYMin();
    float xMin = dmnaFile->getXMin();
    for (i = 0; i < xSize; i++)
    {
        x[i] = xMin + i * delta;
    }
    for (i = 0; i < ySize; i++)
    {
        y[i] = yMin + i * delta;
    }
    for (i = 0; i < zSize; i++)
    {
        z[i] = zValues[i];
    }
    p_grid->setCurrentObject(rgrid);

    return CONTINUE_PIPELINE;
}

ReadLasat::~ReadLasat()
{
    delete houseFile;
    delete dmnaFile;
    delete zValues;
}

int main(int argc, char *argv[])
{
    // create the module
    ReadLasat *application = new ReadLasat;
    application->start(argc, argv);
    return 0;
}

//
// History:
//
// $Log: ReadLasat.cpp,v $
// Revision 1.3  2002/12/17 13:36:05  ralf
// adapted for windows
//
// Revision 1.2  2002/12/12 16:56:54  ralfm_te
// -
//
// Revision 1.1  2002/12/12 11:58:58  cs_te
// initial version
//
//
