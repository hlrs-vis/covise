/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadDepthmapXMIF
//
// This module Reads CSV files
//

#include "ReadDepthmapXMIF.h"

#ifndef _WIN32
#include <inttypes.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>

#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPoints.h>
#include <util/unixcompat.h>

// remove  path from filename
inline const char *coBasename(const char *str)
{
    const char *lastslash = strrchr(str, '/');
    const char *lastslash2 = strrchr(str, '\\');
    if (lastslash2 > lastslash)
        lastslash = lastslash2;
    if (lastslash)
        return lastslash + 1;
    else
    {
        return str;
    }
}

// Module set-up in Constructor
ReadDepthmapXMIF::ReadDepthmapXMIF(int argc, char *argv[])
    : coReader(argc, argv, "Read MIF or CSV data")
{
    z_col = addChoiceParam("z_col", "Select column for z-coordinates");
    d_linesFile = NULL;
}

ReadDepthmapXMIF::~ReadDepthmapXMIF()
{
}

// param callback read header again after all changes
void
ReadDepthmapXMIF::param(const char *paramName, bool inMapLoading)
{

    FileItem *fii = READER_CONTROL->getFileItem(TXT_BROWSER);

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
                    cerr << "ReadDepthmapXMIF::param(..) no data file found " << endl;
                }
                else
                {
                    if (d_linesFile != NULL)
                        fclose(d_linesFile);
                    linesFileName = dataNm;
                    d_linesFile = fopen(linesFileName.c_str(), "r");
                    int result = STOP_PIPELINE;

                    if (d_linesFile != NULL)
                    {
                        result = readHeader();
                        fseek(d_linesFile, 0, SEEK_SET);
                    }

					fileName = dataNm.substr(0,dataNm.length()-3)+"mid";
					d_dataFile = fopen(fileName.c_str(), "r");

					if (d_dataFile == NULL)
                    {
						result = STOP_PIPELINE;
                        cerr << "ReadDepthmapXMIF::param(..) could not read file: " << fileName << endl;
                    }
                    else
                    {

                        // lists for Choice Labels
                        vector<string> dataChoices;

                        // fill in NONE to READ no data
                        string noneStr("NONE");
                        dataChoices.push_back(noneStr);

                        // fill in all species for the appropriate Ports/Choices
                        for (int i = 0; i < varInfos.size(); i++)
                        {
                            dataChoices.push_back(varInfos[i].name);
                        }
                        if (inMapLoading)
                            return;
                        READER_CONTROL->updatePortChoice(DPORT1_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_3D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_3D, dataChoices);
                        if (varInfos.size() > 0)
                            z_col->setValue((int)varInfos.size() + 1, dataChoices, 1);
                    }
                }
                return;
            }

            else
            {
                cerr << "ReadDepthmapXMIF::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

void ReadDepthmapXMIF::nextLine(FILE *fp)
{
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        cerr << "ReadDepthmapXMIF::nextLine: fgets failed" << endl;
    }
}

int ReadDepthmapXMIF::readHeader()
{

    int ii_choiseVals = 0;

	do {
		nextLine(d_linesFile);
		sendInfo("Found header line : %s", buf);
		if (feof(d_linesFile))
		{
			return STOP_PIPELINE;
		}
	} while (strncmp(buf, "Columns", 7) != 0);
	int numColumns = 0;
	sscanf(buf + 7, "%d", &numColumns);

    buf[strlen(buf) - 1] = '\0';
    varInfos.clear();
	for (int i = 0; i<numColumns; i++)
	{
		nextLine(d_linesFile);
		VarInfo vi;
		vi.col = i;
		const char *name = strtok(buf, " ");
		const char *type = strtok(NULL, " ");
		vi.name = name;
		vi.type = type;
		varInfos.push_back(vi);
	}
	do {
		nextLine(d_linesFile);
		if (feof(d_linesFile))
		{
			return STOP_PIPELINE;
		}
	} while (strncmp(buf, "Data", 4) != 0);
	return CONTINUE_PIPELINE;
}

int ReadDepthmapXMIF::readLines()
{
	int res = readHeader();
	numLines = 0;
	numRows = 0;
	std::vector<int> Vlines;
	std::vector<float> VxCoords;
	std::vector<float> VyCoords;
	if(res != STOP_PIPELINE)
	{
		do {
			nextLine(d_linesFile);
			if (feof(d_linesFile))
			{
				return STOP_PIPELINE;
			}
		} while (strncasecmp(buf, "LINE", 4) != 0);

		while (!feof(d_linesFile))
		{
			if (strncasecmp(buf, "LINE", 4) == 0)
			{
				Vlines.push_back(numRows);
				numLines++;
				char *c = buf + 4;
				do{
					float x = 0,y = 0;
					while (*c == ' ' && *c != '\0') // skip whitespace until we have the next pair of numbers
					{
						c++;
					}
					if ((*c >= '0' && *c <= '9')|| *c == '-') // we got a number
					{
						sscanf(c, "%f", &x);
					}
					while (*c != ' ' && *c != '\0') // skip number
					{
						c++;
					}
					while (*c == ' ' && *c != '\0') // skip whitespace until we have the next pair of numbers
					{
						c++;
					}
					if ((*c >= '0' && *c <= '9') || *c == '-') // we got a number
					{
						sscanf(c, "%f", &y);
						VxCoords.push_back(x);
						VyCoords.push_back(y);
						numRows++;
					}
					while (*c != ' ' && *c != '\0') // skip number
					{
						c++;
					}
					if (*c == '\n' || *c == '\r' || *c == '\0')
					{
						nextLine(d_linesFile);
					}

						
				} while (*c == ' ' || *c == '-' || (*c >= '0' && *c <= '9'));
			}
			nextLine(d_linesFile);
		}

		std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);

		numVert = (int)VxCoords.size();
		coDoLines *doLines = new coDoLines(objNameBase.c_str(),(int)VxCoords.size(),numVert,numLines);

		doLines->getAddresses(&xCoords, &yCoords, &zCoords,&v_l,&l_l);
		numRows = (int)VxCoords.size();
		memcpy(xCoords, &VxCoords[0], numRows * sizeof(float));
		memcpy(yCoords, &VyCoords[0], numRows * sizeof(float));
		memcpy(l_l, &Vlines[0], numLines * sizeof(int));
		for (int i = 0; i < numRows; i++)
		{
			v_l[i] = i;
		}

	}
	return res;
}

int ReadDepthmapXMIF::readASCIIData()
{
	char *cbuf;
    int ii, RowCount;
    float *tmpdat;

        int col_for_z = z_col->getValue() - 1;

        printf("%d\n", col_for_z);

        if (col_for_z == -1)
            coModule::sendWarning("No column selected for z-coordinates");

        if (col_for_z >= 0)
        {
           
        }

        for (int n = 0; n < varInfos.size(); n++)
        {
            varInfos[n].dataObjs = new coDistributedObject *[1];
            varInfos[n].dataObjs[0] = NULL;
            varInfos[n].assoc = 0;
        }
        int portID = 0;
        for (int n = 0; n < 5; n++)
        {
            int pos = READER_CONTROL->getPortChoice(DPORT1_3D + n);
            //printf("%d %d\n",pos,n);
            if (pos > 0)
            {
                if (varInfos[pos - 1].assoc == 0)
                {
                    portID = DPORT1_3D + n;
                    coDoFloat *dataObj = new coDoFloat(READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(), numLines);
                    varInfos[pos - 1].dataObjs[0] = dataObj;
                    varInfos[pos - 1].assoc = 1;
                    dataObj->getAddress(&varInfos[pos - 1].x_d);
                }
                else
                {
                    sendWarning("Column %s already associated to port %d", varInfos[pos - 1].name.c_str(), n);
                }
            }
        }

        tmpdat = (float *)malloc(varInfos.size() * sizeof(float));
        RowCount = 0;

        while (fgets(buf, sizeof(buf), d_dataFile) != NULL && (RowCount < numLines))
        {

            for (int i = 0; i < varInfos.size(); i++)
            {
                tmpdat[i] = 0.;
            }

            if ((cbuf = strtok(buf, ",;")) != NULL)
            {
                sscanf(cbuf, "%f", &tmpdat[0]);
            }
            else
            {
                coModule::sendWarning("Error parsing line %d", RowCount + 1);
            }

            ii = 0;
            while ((cbuf = strtok(NULL, ";,")) != NULL)
            {
                ii = ii + 1;
                sscanf(cbuf, "%f", &tmpdat[ii]);
            }

            if (ii < varInfos.size() - 1)
            {
                coModule::sendWarning("Found less values than header elements in data Line %d", RowCount + 1);
            }

            if (col_for_z >= 0)
            {
                //printf("%f %f %f %d\n",tmpdat[col_for_x],tmpdat[col_for_y],tmpdat[col_for_z],RowCount);
				int n = 0;
				if (RowCount < (numLines-1))
					n = l_l[RowCount + 1];
				else
					n = numVert;
				for(int i=l_l[RowCount];i<n;i++)
				{
					zCoords[i] = tmpdat[col_for_z];
				}
            }

            for (int i = 0; i < varInfos.size(); i++)
            {
                if (varInfos[i].assoc == 1)
                {
                    varInfos[i].x_d[RowCount] = tmpdat[i];
                }
            }

            RowCount = RowCount + 1;
        }

        free(tmpdat);

        for (int n = 0; n < varInfos.size(); n++)
        {
            delete[] varInfos[n].dataObjs;
            varInfos[n].assoc = 0;
        }
    return CONTINUE_PIPELINE;
}
int ReadDepthmapXMIF::compute(const char *)
{
    if (d_dataFile == NULL)
    {
        if (fileName.empty())
        {
            cerr << "ReadDepthmapXMIF::param(..) no data file found " << endl;
        }
        else
        {

            d_dataFile = fopen(fileName.c_str(), "r");
        }
    }
    int result = STOP_PIPELINE;
	if (d_linesFile == NULL)
	{
		if (linesFileName.empty())
		{
			cerr << "ReadDepthmapXMIF::param(..) no data file found " << endl;
		}
		else
		{

			d_linesFile = fopen(linesFileName.c_str(), "r");
		}
	}
	readLines();

    if (d_dataFile != NULL)
    {
        result = readASCIIData();
        fclose(d_dataFile);
        d_dataFile = NULL;
    }
    return result;
}

int main(int argc, char *argv[])
{

    // define outline of reader
    READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.mif;*.MIF");

    READER_CONTROL->addOutputPort(MESHPORT3D, "geoOut_3D", "Lines", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_3D, "data1_3D", "Float|Vec3", "data1-3d");
    READER_CONTROL->addOutputPort(DPORT2_3D, "data2_3D", "Float|Vec3", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT3_3D, "data3_3D", "Float|Vec3", "data3-3d");
    READER_CONTROL->addOutputPort(DPORT4_3D, "data4_3D", "Float|Vec3", "data4-3d");
    READER_CONTROL->addOutputPort(DPORT5_3D, "data5_3D", "Float|Vec3", "data5-3d");

    // create the module
    coReader *application = new ReadDepthmapXMIF(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
