/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ Laerm result files             	                  **
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
#include "ReadLaerm.h"
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#ifndef BYTESWAP
#define byteSwap(x) (void)(x)
#endif

ReadLaerm::ReadLaerm(int argc, char *argv[])
    : coReader(argc, argv, "Read iag Laerm mesh and data")
{
    headerState = 1;
    initHeader();

	xc = NULL;
	yc = NULL;
	zc = NULL;
	gc = NULL;
	LD = NULL;
	Nacht = NULL;
	l3 = NULL;
	l4 = NULL;
}

ReadLaerm::~ReadLaerm()
{
}

// =======================================================

int ReadLaerm::readHeader(const char *filename)
{

	//GRID	3452400.00	5393300.00	25.00	25.00	21	17
    // open the file
    int nu = 0, nv = 0, nw = 0;
    coDoStructuredGrid *str_grid = NULL;
    coDoFloat *ustr_s3d_out = NULL;
    if (headerState != 1)
    {
        freeHeader();
    }
    fp = fopen(filename, "r");
    if (fp)
    {
		char buf[1000];
		headerState = 0;
		if(fgets(buf, 1000, fp)==NULL)
		{
			headerState = -1;
		}
		sscanf(buf, "GRID %f %f %f %f %d %d", &header.ox, &header.oy, &header.gridSizeX, &header.gridSizeY, &header.ndimx, &header.ndimy);
		if (fgets(buf, 1000, fp) == NULL)// x	y	z	g	Ld	Nacht	l3	l4
		{
			headerState = -1;
		}
		std::string variables(buf);
		istringstream iss(variables);
		vector<string> tokens;
		copy(std::istream_iterator<std::string>(iss),
			std::istream_iterator<std::string>(),
			back_inserter(tokens));
		for (int i = 4; i < tokens.size(); i++)
			header.variables.push_back(tokens[i]);
		header.ndimz = 2;
    }
    else
    {
        sendError("could not open file: %s", filename);
        return FAIL;
    }
    dataPos = ftello(fp);
    return SUCCESS;
}

void ReadLaerm::initHeader()
{
	header.ndimx = 0;
	header.ndimy = 0;
	header.ndimz = 0;
}

void ReadLaerm::freeHeader()
{
    headerState = 1;
	header.variables.clear();
    initHeader();
}

int ReadLaerm::compute(const char *)
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
    std::string gridNameBase = READER_CONTROL->getAssocObjName(MESHPORT);
	uint64_t dataSize = header.ndimx*header.ndimy;
	xc = new float[dataSize];
	yc = new float[dataSize];
	zc = new float[dataSize];
	gc = new float[dataSize];
	LD = new float[dataSize];
	Nacht = new float[dataSize];
	l3 = new float[dataSize];
	l4 = new float[dataSize];

	for (int i=0; i < dataSize; i++)
	{
		char buf[1000];
		if (fgets(buf, 1000, fp) == NULL)
		{
			return FAIL;
		}
		sscanf(buf, "%f %f %f %f %f %f %f %f", &xc[i], &yc[i], &zc[i], &gc[i], &LD[i], &Nacht[i], &l3[i], &l4[i]);
	}

	coDistributedObject *grid = makegrid(gridNameBase.c_str());
	for (int n = 0; n < header.variables.size(); n++)
	{
		std::string objNameBase = READER_CONTROL->getAssocObjName(DPORT1 + n);
		coDistributedObject *dobj = makeDataObject(objNameBase.c_str(), n);
	}

	delete[] xc;
	delete[] yc;
	delete[] zc;
	delete[] gc;
	delete[] LD;
	delete[] Nacht;
	delete[] l3;
	delete[] l4;
	xc = NULL;
	yc = NULL;
	zc = NULL;
	gc = NULL;
	LD = NULL;
	Nacht = NULL;
	l3 = NULL;
	l4 = NULL;
    return SUCCESS;
}

// param callback update data choice
void
ReadLaerm::param(const char *paramName, bool inMapLoading)
{

    FileItem *fii = READER_CONTROL->getFileItem(Laerm_BROWSER);

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
                    cerr << "ReadLaerm::param(..) no data file found " << endl;
                }
                else
                {
                    if (fileName != dataNm)
                    {
                        headerState = readHeader(dataNm.c_str());
                        if (headerState == SUCCESS)
                        {

                            coModule::sendInfo("Found Dataset with size %dx%d", header.ndimx, header.ndimy);

                            // lists for Choice Labels
                            vector<string> dataChoices;

                            // fill in NONE to READ no data
                            string noneStr("NONE");
                            dataChoices.push_back(noneStr);

                            // fill in all species for the appropriate Ports/Choices
                            for (int i = 0; i < header.variables.size(); i++)
                            {
                                dataChoices.push_back(header.variables[i]);
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
                cerr << "ReadLaerm::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

coDistributedObject *ReadLaerm::makeDataObject(const char *objName, int paramNumber)
{

	uint64_t dataSize = header.ndimx*header.ndimy;
	uint64_t gridSize = dataSize*2;
	float *dataField=NULL;
	switch (READER_CONTROL->getPortChoice(paramNumber + DPORT1))
	{
	case 1:
		dataField = LD;
		break;
	case 2:
		dataField = Nacht;
		break;
	case 3:
		dataField = l3;
		break;
	case 4:
		dataField = l4;
		break;
	}
	if (dataField)
	{
		coDoFloat *dobj = new coDoFloat(objName, (int)gridSize);
		float *scalar;
		dobj->getAddress(&scalar);
		int p = 0;
		for (int i = 0; i < header.ndimx; i++)
			for (int j = 0; j < header.ndimy; j++)
				for (int k = 0; k < header.ndimz; k++)
				{
					float f = dataField[i * header.ndimy + j];
					scalar[p] = f;
					p++;
				}
		return dobj;
	}
	return NULL;
}

coDistributedObject *ReadLaerm::makegrid(const char *objName)
{
        coDoStructuredGrid *strGrd = new coDoStructuredGrid(objName, (int)header.ndimx, (int)header.ndimy, (int)header.ndimz);
        float *xCoord, *yCoord, *zCoord;
        strGrd->getAddresses(&xCoord, &yCoord, &zCoord);

        int p = 0;
		for (int i = 0; i < header.ndimx; i++)
			for (int j = 0; j < header.ndimy; j++)
			{
				size_t index =i * header.ndimy + j;
				xCoord[p] = (float)xc[index];
				yCoord[p] = (float)yc[index];
				zCoord[p] = (float)gc[index];
				p++;
				xCoord[p] = (float)xc[index];
				yCoord[p] = (float)yc[index];
				zCoord[p] = (float)zc[index];
				p++;
			}
        return (strGrd);
    return NULL;
}

int main(int argc, char *argv[])
{

    // define outline of reader
    READER_CONTROL->addFile(ReadLaerm::Laerm_BROWSER, "data_file_path", "Data file path", "/data/enbw/2015/laerm", "*.RST;*.rst");

    READER_CONTROL->addOutputPort(ReadLaerm::MESHPORT, "geoOut", "StructuredGrid", "Geometry", false);

    READER_CONTROL->addOutputPort(ReadLaerm::DPORT1, "data1", "Float|Vec3", "data1");
    READER_CONTROL->addOutputPort(ReadLaerm::DPORT2, "data2", "Float|Vec3", "data2");
    READER_CONTROL->addOutputPort(ReadLaerm::DPORT3, "data3", "Float|Vec3", "data3");
    READER_CONTROL->addOutputPort(ReadLaerm::DPORT4, "data4", "Float|Vec3", "data4");
    READER_CONTROL->addOutputPort(ReadLaerm::DPORT5, "data5", "Float|Vec3", "data5");

    // create the module
    coReader *application = new ReadLaerm(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
