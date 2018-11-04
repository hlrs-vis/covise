/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ EAS result files             	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner                                                   **                             **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include "ReadEAS.h"

#ifndef BYTESWAP
#define byteSwap(x) (void)(x)
#endif

ReadEAS::ReadEAS(int argc, char *argv[])
    : coReader(argc, argv, "Read iag EAS mesh and data")
{
    headerState = 1;
    initHeader();
}

ReadEAS::~ReadEAS()
{
}

// =======================================================

int ReadEAS::readHeader(const char *filename)
{
    // open the file
    int nu = 0, nv = 0, nw = 0;
    coDoStructuredGrid *str_grid = NULL;
    coDoFloat *ustr_s3d_out = NULL;
    if (headerState != 1)
    {
        freeHeader();
    }
    fp = fopen(filename, "rb");
    if (fp)
    {
        if (fread(&(ks.kennung), 20, 1, fp) != 1)
        {
            sendError("Could not read Kennung");
            return FAIL;
        }
        if (fread((char *)&(ks.header.type), sizeof(EASHeader), 1, fp) != 1)
        {
            sendError("Could not read Header");
            return FAIL;
        }
        byteSwap(ks.header.attribMode);
        byteSwap(ks.header.datasize);
        byteSwap(ks.header.geomModeDim1);
        byteSwap(ks.header.geomModeDim2);
        byteSwap(ks.header.geomModeDim3);
        byteSwap(ks.header.geomModeParam);
        byteSwap(ks.header.geomModeTime);
        byteSwap(ks.header.ndim1);
        byteSwap(ks.header.ndim2);
        byteSwap(ks.header.ndim3);
        byteSwap(ks.header.npar);
        byteSwap(ks.header.nzs);
        byteSwap(ks.header.sizeDim1);
        byteSwap(ks.header.sizeDim2);
        byteSwap(ks.header.sizeDim3);
        byteSwap(ks.header.sizeParam);
        byteSwap(ks.header.sizeTime);
        byteSwap(ks.header.type);
        byteSwap(ks.header.userData);
        byteSwap(ks.header.UserDataCharSize);
        byteSwap(ks.header.UserDataIntSize);
        byteSwap(ks.header.UserDataRealSize);
        ks.timestepData = new int64_t[ks.header.nzs];
        if (fread((char *)ks.timestepData, sizeof(int64_t), ks.header.nzs, fp) != ks.header.nzs)
        {
            sendError("Could not read timestepData");
            return FAIL;
        }
        for (int i = 0; i < ks.header.nzs; i++)
        {
            byteSwap(ks.timestepData[i]);
        }
        ks.timestepAttrib = NULL;
        ks.paramAttrib = NULL;
        ks.dimAttrib = NULL;
        if (ks.header.attribMode == EAS3_ALL_ATTR)
        {

            ks.timestepAttrib = new char *[ks.header.nzs];
            ks.paramAttrib = new char *[ks.header.npar];
            ks.dimAttrib = new char *[3];
            for (int i = 0; i < ks.header.nzs; i++)
            {
                ks.timestepAttrib[i] = new char[ATTRLEN + 1];
                ks.timestepAttrib[i][ATTRLEN] = '\0';
                if (fread(ks.timestepAttrib[i], ATTRLEN, 1, fp) != 1)
                {
                    sendError("Could not read timestepAttrib");
                    return FAIL;
                }
            }
            for (int i = 0; i < ks.header.npar; i++)
            {
                ks.paramAttrib[i] = new char[ATTRLEN + 1];
                ks.paramAttrib[i][ATTRLEN] = '\0';
                if (fread(ks.paramAttrib[i], ATTRLEN, 1, fp) != 1)
                {
                    sendError("Could not read paramAttrib");
                    return FAIL;
                }
            }
            for (int i = 0; i < 3; i++)
            {
                ks.dimAttrib[i] = new char[ATTRLEN + 1];
                ks.dimAttrib[i][ATTRLEN] = '\0';
                if (fread(ks.dimAttrib[i], ATTRLEN, 1, fp) != 1)
                {
                    sendError("Could not read dim1Attrib");
                    return FAIL;
                }
            }
        }

        ks.timeData = NULL;
        ks.paramData = NULL;
        ks.dim1Data = NULL;
        ks.dim2Data = NULL;
        ks.dim3Data = NULL;
        if (ks.header.geomModeTime > EAS3_NO_G)
        {
            ks.timeData = new double[ks.header.sizeTime];
            if (fread((char *)ks.timeData, sizeof(double), ks.header.sizeTime, fp) != ks.header.sizeTime)
            {
                sendError("Could not read timeData");
                return FAIL;
            }
            for (int i = 0; i < ks.header.sizeTime; i++)
            {
                byteSwap(ks.timeData[i]);
            }
        }
        if (ks.header.geomModeParam > EAS3_NO_G)
        {
            ks.paramData = new double[ks.header.sizeParam];
            if (fread((char *)ks.paramData, sizeof(double), ks.header.sizeParam, fp) != ks.header.sizeParam)
            {
                sendError("Could not read paramData");
                return FAIL;
            }
            for (int i = 0; i < ks.header.sizeParam; i++)
            {
                byteSwap(ks.paramData[i]);
            }
        }
        if (ks.header.geomModeDim1 > EAS3_NO_G)
        {
            ks.dim1Data = new double[ks.header.sizeDim1];
            if (fread((char *)ks.dim1Data, sizeof(double), ks.header.sizeDim1, fp) != ks.header.sizeDim1)
            {
                sendError("Could not read dim1Data");
                return FAIL;
            }
            for (int i = 0; i < ks.header.sizeDim1; i++)
            {
                byteSwap(ks.dim1Data[i]);
            }
        }
        if (ks.header.geomModeDim2 > EAS3_NO_G)
        {
            ks.dim2Data = new double[ks.header.sizeDim2];
            if (fread((char *)ks.dim2Data, sizeof(double), ks.header.sizeDim2, fp) != ks.header.sizeDim2)
            {
                sendError("Could not read dim2Data");
                return FAIL;
            }
            for (int i = 0; i < ks.header.sizeDim2; i++)
            {
                byteSwap(ks.dim2Data[i]);
            }
        }
        if (ks.header.geomModeDim3 > EAS3_NO_G)
        {
            ks.dim3Data = new double[ks.header.sizeDim3];
            if (fread((char *)ks.dim3Data, sizeof(double), ks.header.sizeDim3, fp) != ks.header.sizeDim3)
            {
                sendError("Could not read dim3Data");
                return FAIL;
            }
            for (int i = 0; i < ks.header.sizeDim3; i++)
            {
                byteSwap(ks.dim3Data[i]);
            }
        }
        ks.userDataChar = NULL;
        ks.userDataInt = NULL;
        ks.userDataReal = NULL;
        if (ks.header.userData > EAS3_NO_UDEF)
        {
            ks.userDataChar = new char *[ks.header.UserDataCharSize];
            for (int i = 0; i < ks.header.UserDataCharSize; i++)
            {
                ks.userDataChar[i] = new char[UDEFLEN + 1];
                ks.userDataChar[i][UDEFLEN] = '\0';
                if (fread(ks.userDataChar[i], UDEFLEN, 1, fp) != 1)
                {
                    sendError("Could not read userDataChar");
                    return FAIL;
                }
            }
            ks.userDataInt = new int64_t[ks.header.UserDataIntSize];
            if (fread((char *)ks.userDataInt, sizeof(int64_t), ks.header.UserDataIntSize, fp) != ks.header.UserDataIntSize)
            {
                sendError("Could not read userDataInt");
                return FAIL;
            }
            for (int i = 0; i < ks.header.UserDataIntSize; i++)
            {
                byteSwap(ks.userDataInt[i]);
            }
            ks.userDataReal = new double[ks.header.UserDataRealSize];
            if (fread((char *)ks.userDataReal, sizeof(int64_t), ks.header.UserDataRealSize, fp) != ks.header.UserDataRealSize)
            {
                sendError("Could not read userDataReal");
                return FAIL;
            }
            for (int i = 0; i < ks.header.UserDataRealSize; i++)
            {
                byteSwap(ks.userDataReal[i]);
            }
        }
    }
    else
    {
        sendError("could not open file: %s", filename);
        return FAIL;
    }
    dataPos = ftello(fp);
    return SUCCESS;
}

void ReadEAS::initHeader()
{
    ks.header.UserDataCharSize = 0;
    ks.userDataChar = NULL;
    ks.userDataInt = NULL;
    ks.userDataReal = NULL;

    ks.timeData = NULL;
    ks.paramData = NULL;
    ks.dim1Data = NULL;
    ks.dim2Data = NULL;
    ks.dim3Data = NULL;
    ks.header.nzs = 0;
    ks.header.npar = 0;

    ks.dimAttrib = NULL;
    ks.timestepAttrib = NULL;
    ks.paramAttrib = NULL;
    ks.dimAttrib = NULL;
    ks.timestepData = NULL;
}

void ReadEAS::freeHeader()
{
    headerState = 1;

    for (int i = 0; i < ks.header.UserDataCharSize; i++)
    {
        delete[] ks.userDataChar[i];
    }
    delete[] ks.userDataChar;
    delete[] ks.userDataInt;
    delete[] ks.userDataReal;

    delete[] ks.timeData;
    delete[] ks.paramData;
    delete[] ks.dim1Data;
    delete[] ks.dim2Data;
    delete[] ks.dim3Data;

    for (int i = 0; i < ks.header.nzs; i++)
    {
        delete[] ks.timestepAttrib[i];
    }
    for (int i = 0; i < ks.header.npar; i++)
    {
        delete[] ks.paramAttrib[i];
    }
    if (ks.dimAttrib)
    {
        for (int i = 0; i < 3; i++)
        {
            delete[] ks.dimAttrib[i];
        }
    }
    delete[] ks.timestepAttrib;
    delete[] ks.paramAttrib;
    delete[] ks.dimAttrib;
    delete[] ks.timestepData;
    initHeader();
}

int ReadEAS::compute(const char *)
{
    if (headerState == 1) // not read)
    {
        FileItem *fi(READER_CONTROL->getFileItem(string("data_file_path")));
        if (fi)
        {
            coFileBrowserParam *bP = fi->getBrowserPtr();

            if (bP)
            {
                headerState = readHeader(bP->getValue());
            }
        }
    }
    if (headerState == FAIL)
    {
        return FAIL;
    }
    fseeko(fp, dataPos, SEEK_SET);
    int nb = 2;
    if (ks.header.datasize == IEEES)
    {
        nb = 4;
    }
    else if (ks.header.datasize == IEEED)
    {
        nb = 8;
    }
    else
    {
        nb = 16;
    }

    size_t dataSize = ks.header.ndim1 * ks.header.ndim2 * ks.header.ndim3 * nb;

    std::string gridNameBase = READER_CONTROL->getAssocObjName(MESHPORT);
    // we have a vaild header, thus read the grid and data
    if (ks.header.nzs < 2)
    {
        coDistributedObject *grid = makegrid(gridNameBase.c_str());
        //p_mesh->setCurrentObject(grid);

        int *portNumbers = new int[ks.header.npar]; // port # to store dataset (-1 if not read)
        for (int i = 0; i < ks.header.npar; i++)
        {
            portNumbers[i] = -1;
        }
        for (int n = 0; n < NUM_DATA_PORTS; n++)
        {
            int pos = READER_CONTROL->getPortChoice(DPORT1 + n);
            if (pos > 0)
            {
                portNumbers[pos - 1] = n;
            }
        }
        for (int n = 0; n < ks.header.npar; n++)
        {
            if (portNumbers[n] < 0) // skip this data
            {
                fseeko(fp, dataSize, SEEK_CUR);
            }
            else
            {
                std::string objNameBase = READER_CONTROL->getAssocObjName(DPORT1 + n);
                coDistributedObject *dobj = makeDataObject(objNameBase.c_str(), n);
            }
        }
    }
    else
    {
        coDistributedObject **grids = new coDistributedObject *[ks.header.nzs + 1];
        coDistributedObject ***dataObjs = new coDistributedObject **[NUM_DATA_PORTS];
        for (int i = 0; i < NUM_DATA_PORTS; i++)
        {
            dataObjs[i] = new coDistributedObject *[ks.header.nzs + 1];
        }

        char *gridName = new char[gridNameBase.length() + 100];
        sprintf(gridName, "%s_firstGrid", gridNameBase.c_str());
        coDistributedObject *grid = makegrid(gridName);
        int *portNumbers = new int[ks.header.npar]; // port # to store dataset (-1 if not read)
        for (int i = 0; i < ks.header.npar; i++)
        {
            portNumbers[i] = -1;
        }
        for (int n = 0; n < NUM_DATA_PORTS; n++)
        {
            int pos = READER_CONTROL->getPortChoice(DPORT1 + n);
            if (pos > 0)
            {
                portNumbers[pos - 1] = n;
            }
        }
        for (int t = 0; t < ks.header.nzs; t++)
        {
            grids[t] = grid;
            if (t > 0)
                grid->incRefCount();
            grids[t + 1] = NULL;
            for (int n = 0; n < ks.header.npar; n++)
            {
                if (portNumbers[n] < 0) // skip this data
                {
                    fseeko(fp, dataSize, SEEK_CUR);
                }
                else
                {
                    std::string objNameBase = READER_CONTROL->getAssocObjName(DPORT1 + n);
                    char *objName = new char[objNameBase.length() + 100];
                    sprintf(objName, "%s_%d", objNameBase.c_str(), t);
                    dataObjs[portNumbers[n]][t] = makeDataObject(objName, n);
                    dataObjs[portNumbers[n]][t + 1] = NULL;
                }
            }
        }
        coDoSet *setOut = new coDoSet(gridNameBase.c_str(), grids);
        setOut->addAttribute("TIMESTEP", "0 1");
        delete[] grids;

        for (int i = 0; i < NUM_DATA_PORTS; i++)
        {
            int pos = READER_CONTROL->getPortChoice(DPORT1 + i);
            if (pos > 0 && dataObjs[i][0])
            {
                std::string objNameBase = READER_CONTROL->getAssocObjName(DPORT1 + i);
                coDoSet *setOut = new coDoSet(objNameBase.c_str(), dataObjs[i]);
                setOut->addAttribute("TIMESTEP", "0 1");
            }
            delete[] dataObjs[i];
        }
        delete[] dataObjs;
        delete[] portNumbers;
    }
    return SUCCESS;
}

// param callback update data choice
void
ReadEAS::param(const char *paramName, bool inMapLoading)
{

    FileItem *fii = READER_CONTROL->getFileItem(EAS_BROWSER);

    string txtBrowserName;
    if (fii)
    {
        txtBrowserName = fii->getName();
    }

    /////////////////  CALLED BY FILE BROWSER  //////////////////
    if (txtBrowserName == string(paramName))
    {
        FileItem *fi(READER_CONTROL->getFileItem(string(paramName)));
        if (fi)
        {
            coFileBrowserParam *bP = fi->getBrowserPtr();

            if (bP)
            {
                string dataNm(bP->getValue());
                if (dataNm.empty())
                {
                    cerr << "ReadEAS::param(..) no data file found " << endl;
                }
                else
                {
                    if (fileName != dataNm)
                    {
                        headerState = readHeader(dataNm.c_str());
                        if (headerState == SUCCESS)
                        {

                            coModule::sendInfo("Found Dataset with %ld timesteps", (long)ks.header.nzs);

                            // lists for Choice Labels
                            vector<string> dataChoices;

                            // fill in NONE to READ no data
                            string noneStr("NONE");
                            dataChoices.push_back(noneStr);

                            // fill in all species for the appropriate Ports/Choices
                            for (int i = 0; i < ks.header.npar; i++)
                            {
                                dataChoices.push_back(ks.paramAttrib[i]);
                            }
                            if (inMapLoading)
                                return;
                            READER_CONTROL->updatePortChoice(DPORT1, dataChoices);
                            READER_CONTROL->updatePortChoice(DPORT2, dataChoices);
                            READER_CONTROL->updatePortChoice(DPORT3, dataChoices);
                            READER_CONTROL->updatePortChoice(DPORT4, dataChoices);
                            READER_CONTROL->updatePortChoice(DPORT5, dataChoices);
                        }
                    }
                }
                return;
            }

            else
            {
                cerr << "ReadEAS::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

coDistributedObject *ReadEAS::makeDataObject(const char *objName, int paramNumber)
{
    size_t nElem = ks.header.ndim1 * ks.header.ndim2 * ks.header.ndim3;
    coDoFloat *dobj = new coDoFloat(objName, (int)nElem);
    float *scalar;
    dobj->getAddress(&scalar);
    if (ks.header.datasize == IEEES)
    {
        float *tmpData = new float[nElem];
        if (fread(tmpData, 4, nElem, fp) < nElem)
        {
            return NULL;
        }
        int p = 0;
        size_t n12 = ks.header.ndim1 * ks.header.ndim2;
        for (int i = 0; i < ks.header.ndim1; i++)
            for (int j = 0; j < ks.header.ndim2; j++)
                for (int k = 0; k < ks.header.ndim3; k++)
                {
                    float f = tmpData[k * n12 + j * ks.header.ndim1 + i];
                    byteSwap(f);
                    scalar[p] = f;
                    p++;
                }
    }
    else if (ks.header.datasize == IEEED)
    {
        double *tmpData = new double[nElem];
        if (fread(tmpData, 8, nElem, fp) < nElem)
        {
            return NULL;
        }
        int p = 0;
        size_t n12 = ks.header.ndim1 * ks.header.ndim2;
        for (int i = 0; i < ks.header.ndim1; i++)
            for (int j = 0; j < ks.header.ndim2; j++)
                for (int k = 0; k < ks.header.ndim3; k++)
                {
                    double d = tmpData[k * n12 + j * ks.header.ndim1 + i];
                    byteSwap(d);
                    scalar[p] = float(d);
                    p++;
                }
    }
    else
    {
        long double *tmpData = new long double[nElem];
        if (fread(tmpData, 16, nElem, fp) < nElem)
        {
            return NULL;
        }
        int p = 0;
        size_t n12 = ks.header.ndim1 * ks.header.ndim2;
        for (int i = 0; i < ks.header.ndim1; i++)
            for (int j = 0; j < ks.header.ndim2; j++)
                for (int k = 0; k < ks.header.ndim3; k++)
                {
                    long double ld = tmpData[k * n12 + j * ks.header.ndim1 + i];
                    //byteSwap(ld);
                    scalar[p] = float(ld);
                    p++;
                }
    }
    return dobj;
}

coDistributedObject *ReadEAS::makegrid(const char *objName)
{
    if (ks.header.geomModeDim1 == EAS3_FULL_G)
    {
        coDoStructuredGrid *strGrd = new coDoStructuredGrid(objName, (int)ks.header.ndim1, (int)ks.header.ndim2, (int)ks.header.ndim3);
        float *xCoord, *yCoord, *zCoord;
        strGrd->getAddresses(&xCoord, &yCoord, &zCoord);

        int p = 0;
        size_t n12 = ks.header.ndim1 * ks.header.ndim2;
        for (int i = 0; i < ks.header.ndim1; i++)
            for (int j = 0; j < ks.header.ndim2; j++)
                for (int k = 0; k < ks.header.ndim3; k++)
                {
                    size_t index = k * n12 + j * ks.header.ndim1 + i;
                    xCoord[p] = (float)ks.dim2Data[index];
                    yCoord[p] = (float)ks.dim1Data[index];
                    zCoord[p] = (float)ks.dim3Data[index];
                    p++;
                }
        return (strGrd);
    }
    else
    {
        coDoRectilinearGrid *rectGrd = new coDoRectilinearGrid(objName, (int)ks.header.ndim1, (int)ks.header.ndim2, (int)ks.header.ndim3);
        float *xCoord, *yCoord, *zCoord;
        rectGrd->getAddresses(&xCoord, &yCoord, &zCoord);
        if (ks.header.geomModeDim1 == EAS3_X0DX_G)
        {
            float x0, dx;
            x0 = (float)ks.dim1Data[0];
            dx = (float)ks.dim1Data[1];
            for (int i = 0; i < ks.header.ndim1; i++)
            {
                xCoord[i] = (float)x0 + i * dx;
            }
        }
        else if (ks.header.geomModeDim1 == EAS3_ALL_G)
        {
            for (int i = 0; i < ks.header.ndim1; i++)
            {
                xCoord[i] = (float)ks.dim2Data[i];
            }
        }
        if (ks.header.geomModeDim2 == EAS3_X0DX_G)
        {
            float x0, dx;
            x0 = (float)ks.dim2Data[0];
            dx = (float)ks.dim2Data[1];
            for (int i = 0; i < ks.header.ndim2; i++)
            {
                yCoord[i] = (float)x0 + i * dx;
            }
        }
        else if (ks.header.geomModeDim2 == EAS3_ALL_G)
        {
            for (int i = 0; i < ks.header.ndim2; i++)
            {
                yCoord[i] = (float)ks.dim1Data[i];
            }
        }
        if (ks.header.geomModeDim3 == EAS3_X0DX_G)
        {
            float x0, dx;
            x0 = (float)ks.dim3Data[0];
            dx = (float)ks.dim3Data[1];
            for (int i = 0; i < ks.header.ndim3; i++)
            {
                zCoord[i] = x0 + i * dx;
            }
        }
        else if (ks.header.geomModeDim3 == EAS3_ALL_G)
        {
            for (int i = 0; i < ks.header.ndim3; i++)
            {
                zCoord[i] = (float)ks.dim3Data[i];
            }
        }
        return (rectGrd);
    }
    return NULL;
}

int main(int argc, char *argv[])
{

    // define outline of reader
    READER_CONTROL->addFile(ReadEAS::EAS_BROWSER, "data_file_path", "Data file path", "/data/iag/fou", "*.eas;*.EAS");

    READER_CONTROL->addOutputPort(ReadEAS::MESHPORT, "geoOut", "StructuredGrid|RectilinearGrid", "Geometry", false);

    READER_CONTROL->addOutputPort(ReadEAS::DPORT1, "data1", "Float|Vec3", "data1");
    READER_CONTROL->addOutputPort(ReadEAS::DPORT2, "data2", "Float|Vec3", "data2");
    READER_CONTROL->addOutputPort(ReadEAS::DPORT3, "data3", "Float|Vec3", "data3");
    READER_CONTROL->addOutputPort(ReadEAS::DPORT4, "data4", "Float|Vec3", "data4");
    READER_CONTROL->addOutputPort(ReadEAS::DPORT5, "data5", "Float|Vec3", "data5");

    // create the module
    coReader *application = new ReadEAS(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
