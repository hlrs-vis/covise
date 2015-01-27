/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadCSV
//
// This module Reads CSV files
//

#include "ReadCSV.h"

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
ReadCSV::ReadCSV(int argc, char *argv[])
    : coReader(argc, argv, "Read CSV data")
{
    x_col = addChoiceParam("x_col", "Select column for x-coordinates");
    y_col = addChoiceParam("y_col", "Select column for y-coordinates");
    z_col = addChoiceParam("z_col", "Select column for z-coordinates");
    d_dataFile = NULL;
}

ReadCSV::~ReadCSV()
{
}

// param callback read header again after all changes
void
ReadCSV::param(const char *paramName, bool inMapLoading)
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
                    cerr << "ReadCSV::param(..) no data file found " << endl;
                }
                else
                {
                    if (d_dataFile != NULL)
                        fclose(d_dataFile);
                    fileName = dataNm;
                    d_dataFile = fopen(dataNm.c_str(), "r");
                    int result = STOP_PIPELINE;

                    if (d_dataFile != NULL)
                    {
                        result = readHeader();
                        fseek(d_dataFile, 0, SEEK_SET);
                    }

                    if (result == STOP_PIPELINE)
                    {
                        cerr << "ReadCSV::param(..) could not read file: " << dataNm << endl;
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
                            x_col->setValue(varInfos.size() + 1, dataChoices, 1);
                        if (varInfos.size() > 1)
                            y_col->setValue(varInfos.size() + 1, dataChoices, 2);
                        if (varInfos.size() > 2)
                            z_col->setValue(varInfos.size() + 1, dataChoices, 3);
                    }
                }
                return;
            }

            else
            {
                cerr << "ReadCSV::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

void ReadCSV::nextLine(FILE *fp)
{
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        cerr << "ReadCSV::nextLine: fgets failed" << endl;
    }
}

int ReadCSV::readHeader()
{

    int ii_choiseVals = 0, ii;

    nextLine(d_dataFile);
    sendInfo("Found header line : %s", buf);

    buf[strlen(buf) - 1] = '\0';

    std::vector<std::string> names;
    varInfos.clear();
    names.clear();
    const char *name = strtok(buf, ";,");
    names.push_back(name);

    while ((name = strtok(NULL, ";,")) != NULL)
    {
        names.push_back(name);
    }

    if (names.size() > 0)
    {

        for (int i = 0; i < names.size(); i++)
        {
            VarInfo vi;
            vi.name = names[i];
            varInfos.push_back(vi);
        }

        sendInfo("Found %lu header elements", (unsigned long)names.size());

        numRows = 0;
        while (fgets(buf, sizeof(buf), d_dataFile) != NULL)
        {
            numRows = numRows + 1;
        }
        coModule::sendInfo("Found %d data lines", numRows);

        rewind(d_dataFile);
        nextLine(d_dataFile);

        return CONTINUE_PIPELINE;
    }
    else
    {
        return STOP_PIPELINE;
    }
}

// taken from old ReadCSV module: 2-Pass reading
int ReadCSV::readASCIIData()
{
    char *cbuf;
    int res = readHeader();
    int ii, RowCount;
    float *tmpdat;

    if (res != STOP_PIPELINE)
    {

        int col_for_x = x_col->getValue() - 1;
        int col_for_y = y_col->getValue() - 1;
        int col_for_z = z_col->getValue() - 1;

        printf("%d %d %d\n", col_for_x, col_for_y, col_for_z);

        if (col_for_x == -1)
            coModule::sendWarning("No column selected for x-coordinates");
        if (col_for_y == -1)
            coModule::sendWarning("No column selected for y-coordinates");
        if (col_for_z == -1)
            coModule::sendWarning("No column selected for z-coordinates");

        if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
        {
            std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
            coDoPoints *grid = new coDoPoints(objNameBase.c_str(), numRows);

            grid->getAddresses(&xCoords, &yCoords, &zCoords);
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
                    coDoFloat *dataObj = new coDoFloat(READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(), numRows);
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

        while (fgets(buf, sizeof(buf), d_dataFile) != NULL)
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

            if ((col_for_x >= 0) && (col_for_y >= 0) && (col_for_z >= 0))
            {
                //printf("%f %f %f %d\n",tmpdat[col_for_x],tmpdat[col_for_y],tmpdat[col_for_z],RowCount);
                xCoords[RowCount] = tmpdat[col_for_x];
                yCoords[RowCount] = tmpdat[col_for_y];
                zCoords[RowCount] = tmpdat[col_for_z];
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
    }
    return CONTINUE_PIPELINE;
}
int ReadCSV::compute(const char *)
{
    if (d_dataFile == NULL)
    {
        if (fileName.empty())
        {
            cerr << "ReadCSV::param(..) no data file found " << endl;
        }
        else
        {

            d_dataFile = fopen(fileName.c_str(), "r");
        }
    }
    int result = STOP_PIPELINE;

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
    READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.csv;*.CSV");

    READER_CONTROL->addOutputPort(MESHPORT3D, "geoOut_3D", "Points", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_3D, "data1_3D", "Float|Vec3", "data1-3d");
    READER_CONTROL->addOutputPort(DPORT2_3D, "data2_3D", "Float|Vec3", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT3_3D, "data3_3D", "Float|Vec3", "data3-3d");
    READER_CONTROL->addOutputPort(DPORT4_3D, "data4_3D", "Float|Vec3", "data4-3d");
    READER_CONTROL->addOutputPort(DPORT5_3D, "data5_3D", "Float|Vec3", "data5-3d");

    // create the module
    coReader *application = new ReadCSV(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
