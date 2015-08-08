/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadCOMSOLData
//
// This module Reads COMSOL COMSOLData files
//

#include "ReadCOMSOLData.h"

#ifndef _WIN32
#include <inttypes.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>

#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>

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
ReadCOMSOLData::ReadCOMSOLData(int argc, char *argv[])
    : coReader(argc, argv, "Read COMSOLData mesh and data")
{
    d_dataFile = NULL;
}

ReadCOMSOLData::~ReadCOMSOLData()
{
}

// param callback read header again after all changes
void
ReadCOMSOLData::param(const char *paramName, bool inMapLoading)
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
                    cerr << "ReadCOMSOLData::param(..) no data file found " << endl;
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
                        cerr << "ReadCOMSOLData::param(..) could not read file: " << dataNm << endl;
                    }
                    else
                    {

                        coModule::sendInfo("Found Dataset with %d timesteps", numTimesteps);

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
                        READER_CONTROL->updatePortChoice(DPORT1_2D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_2D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_2D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_2D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_2D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT1_1D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_1D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_1D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_1D, dataChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_1D, dataChoices);
                    }
                }
                return;
            }

            else
            {
                cerr << "ReadCOMSOLData::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

void ReadCOMSOLData::nextLine(FILE *fp)
{
    if (fgets(buf, sizeof(buf), fp) == NULL)
    {
        cerr << "ReadCOMSOLData::nextLine: fgets failed" << endl;
    }
}
std::string trim(const std::string& str,
                 const std::string& whitespace = " \t\r\n\f\v")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

int ReadCOMSOLData::readHeader()
{
    while (!feof(d_dataFile))
    {
        nextLine(d_dataFile);
        if (strncmp(buf, "% Dimension", 11) == 0) // read a dataset
        {
            sscanf(buf + 13, "%d", &dimentions);
            nextLine(d_dataFile); //% Nodes
            sscanf(buf + 9, "%d", &numPoints);
            nextLine(d_dataFile); //% Elements:           2024
            sscanf(buf + 12, "%d", &numElements);
            nextLine(d_dataFile);
            sscanf(buf + 15, "%d", &numExpressions);
            nextLine(d_dataFile); // variable description
            std::vector<std::string> names;
            char *c = buf + 15;
            while (*c == ' ')
                c++;
            varInfos.clear();
            names.clear();
            const char *name = strtok(c, ",");
            names.push_back(trim(name));
            while ((name = strtok(NULL, ",")) != NULL)
                names.push_back(trim(name));

            numScalars = 0;
            for (int i = 0; i < names.size(); i++)
            {
                VarInfo vi;
                if (i < names.size() - 2 && names[i].substr(2) == "coordinate" && names[i + 1].substr(2) == "coordinate" && names[i + 2].substr(2) == "coordinate")
                {
                    std::string baseName = names[i].substr(2);
                    vi.valueIndex = numScalars;
                    vi.components = 3;
                    numScalars += 3;
                    i += 2;
                    vi.name = baseName;
                }
                else if (i < names.size() - 2 && names[i] == "Particle position" && names[i + 1] == "Particle position" && names[i + 2] == "Particle position")
                {
                    vi.valueIndex = numScalars;
                    vi.components = 3;
                    numScalars += 3;
                    i += 2;
                    vi.name = names[i];
                }
                else if (i < names.size() - 1 && names[i + 1].substr(2) == "component")
                {
                    std::string baseName = names[i];
                    vi.components = 0;
                    vi.valueIndex = numScalars;
                    while ((i + 2 * vi.components) < names.size() && names[i + 2 * vi.components] == baseName)
                    {
                        vi.components++;
                        numScalars++;
                    }
                    i += (vi.components * 2) - 1;
                    vi.name = baseName;
                }
                else
                {
                    vi.name = names[i];
                    vi.components = 1;
                    vi.valueIndex = numScalars;
                    numScalars++;
                }
                varInfos.push_back(vi);
            }
            numTimesteps = numExpressions / numScalars;
            do
            {
                nextLine(d_dataFile); // coordinates
            } while (strncmp(buf, "% Coordinates", 13) != 0);
            return CONTINUE_PIPELINE;
        }
    }
    return STOP_PIPELINE;
}

// taken from old ReadCOMSOLData module: 2-Pass reading
int ReadCOMSOLData::readASCIIData()
{
    char *cbuf;
    int res = readHeader();
    if (res != STOP_PIPELINE)
    {
        if (dimentions == 1)
        {
            std::string objNameBase = READER_CONTROL->getAssocObjName(GEOPORT1D);
            /*coDoPoints *points = new coDoPoints(objNameBase.c_str(), numPoints);
            float *x_c;
            float *y_c;
            float *z_c;
            points->getAddresses(&x_c, &y_c, &z_c);*/
            for (int i = 0; i < numPoints; i++)
            {
                nextLine(d_dataFile);
                float tmpf;
                sscanf(buf, "%d", &tmpf);
            }
            nextLine(d_dataFile);
            for (int i = 0; i < numElements; i++)
            {
                nextLine(d_dataFile);
                float tmpf;
                sscanf(buf, "%d", &tmpf);
            }
        }
        else if (dimentions == 2)
        {
            std::string objNameBase = READER_CONTROL->getAssocObjName(GEOPORT2D);
            coDoPolygons *poly = new coDoPolygons(objNameBase.c_str(), numPoints, numElements * 3, numElements);
            float *x_c;
            float *y_c;
            float *z_c;
            int *v_l;
            int *l_l;
            poly->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
            for (int i = 0; i < numPoints; i++)
            {
                nextLine(d_dataFile);
                sscanf(buf, "%f %f", x_c + i, y_c + i);
                z_c[i] = 0;
            }
            nextLine(d_dataFile);
            for (int i = 0; i < numElements; i++)
            {
                nextLine(d_dataFile);
                sscanf(buf, "%d %d %d", v_l + (i * 3), v_l + (i * 3) + 1, v_l + (i * 3) + 2);
                v_l[i * 3]--;
                v_l[i * 3 + 1]--;
                v_l[i * 3 + 2]--;
                l_l[i] = i * 3;
            }
        }
        if (dimentions == 3)
        {
            std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
            coDoUnstructuredGrid *grid = new coDoUnstructuredGrid(objNameBase.c_str(), numElements, numElements * 4, numPoints, 1);
            float *x_c;
            float *y_c;
            float *z_c;
            int *v_l;
            int *l_l;
            int *t_l;
            grid->getAddresses(&l_l, &v_l, &x_c, &y_c, &z_c);
            for (int i = 0; i < numPoints; i++)
            {
                nextLine(d_dataFile);
                sscanf(buf, "%f %f %f", x_c + i, y_c + i, z_c + i);
            }
            nextLine(d_dataFile);
            grid->getTypeList(&t_l);
            for (int i = 0; i < numElements; i++)
            {
                nextLine(d_dataFile);
                sscanf(buf, "%d %d %d %d", v_l + (i * 4), v_l + (i * 4) + 1, v_l + (i * 4) + 2, v_l + (i * 4) + 3);
                v_l[i * 4]--;
                v_l[i * 4 + 1]--;
                v_l[i * 4 + 2]--;
                v_l[i * 4 + 3]--;
                l_l[i] = i * 4;
                t_l[i] = TYPE_TETRAHEDER;
            }
        }

        for (int n = 0; n < varInfos.size(); n++)
        {
            varInfos[n].dataObjs = new coDistributedObject *[numTimesteps + 1];
            varInfos[n].dataObjs[0] = NULL;
        }
        for (int t = 0; t < numTimesteps; t++)
        {
            for (int i = 0; i < numScalars; i++)
            {
                int portID = 0;
                for (int n = 0; n < 5; n++)
                {
                    int pos = READER_CONTROL->getPortChoice(DPORT1_3D + n);
                    if (pos > 0 && (varInfos[pos - 1].valueIndex == i))
                    {
                        portID = DPORT1_3D + n;
                        break;
                    }
                }
                if (portID > 0)
                {
                    VarInfo &vi=varInfos[0];
                    int varIndex = 0;
                    for (int n = 0; n < varInfos.size(); n++)
                    {
                        if (varInfos[n].valueIndex == i)
                        {
                            vi = varInfos[n];
                            varIndex = n;
                        }
                    }
                    string objName;
                    if(vi.name == "Particle position")
                    {
                        vi.objectName = READER_CONTROL->getAssocObjName(GEOPORT1D);
                    }
                    else
                    {
                        vi.objectName = READER_CONTROL->getAssocObjName(portID);
                    }
                    if (numTimesteps < 2)
                    {
                        objName = vi.objectName;
                    }
                    else
                    {
                        char tmpName[500];
                        sprintf(tmpName, "%s_%d", vi.objectName.c_str(), t);
                        objName = tmpName;
                    }
                    if (vi.components == 1)
                    {
                        nextLine(d_dataFile);
                        coDoFloat *dataObj = new coDoFloat(objName.c_str(), numPoints);

                        vi.dataObjs[t] = dataObj;
                        vi.dataObjs[t + 1] = NULL;
                        float *x_d;
                        float *y_d;
                        float *z_d;
                        dataObj->getAddress(&x_d);
                        for (int i = 0; i < numPoints; i++)
                        {
                            nextLine(d_dataFile);
                            sscanf(buf, "%f", x_d + i);
                        }
                    }
                    else
                    {
                        nextLine(d_dataFile);
                        coDistributedObject *distrObj=NULL;
                        float *x_d;
                        float *y_d;
                        float *z_d;
                        if(vi.name == "Particle position")
                        {
                            coDoPoints *dataObj = new coDoPoints(objName.c_str(), numPoints);
                            dataObj->getAddresses(&x_d, &y_d, &z_d);
                            distrObj = dataObj;
                        }
                        else
                        {
                            coDoVec3 *dataObj = new coDoVec3(objName.c_str(), numPoints);
                            dataObj->getAddresses(&x_d, &y_d, &z_d);
                            distrObj = dataObj;
                        }
                        vi.dataObjs[t] = distrObj;
                        vi.dataObjs[t + 1] = NULL;
                        for (int p = 0; p < numPoints; p++)
                        {
                            nextLine(d_dataFile);
                            sscanf(buf, "%f", x_d + p);
                        }
                        i++; // next scalar
                        nextLine(d_dataFile);
                        for (int p = 0; p < numPoints; p++)
                        {
                            nextLine(d_dataFile);
                            sscanf(buf, "%f", y_d + p);
                        }
                        if (vi.components == 3)
                        {
                            i++; // next scalar
                            nextLine(d_dataFile);
                            for (int p = 0; p < numPoints; p++)
                            {
                                nextLine(d_dataFile);
                                sscanf(buf, "%f", z_d + p);
                            }
                        }
                        else
                        {
                            for (int p = 0; p < numPoints; p++)
                            {
                                z_d[p] = 0.0;
                            }
                        }
                    }
                }
                else
                {
                    nextLine(d_dataFile);
                    for (int i = 0; i < numPoints; i++)
                    {
                        nextLine(d_dataFile);
                    }
                }
            }
        }
        if (numTimesteps > 1)
        {
            for (int n = 0; n < varInfos.size(); n++)
            {
                if (varInfos[n].dataObjs[0])
                {
                    coDoSet *myset = new coDoSet(varInfos[n].objectName, varInfos[n].dataObjs);
                    myset->addAttribute("TIMESTEP", "1 x");
                    if(varInfos[n].name == "Particle position")
                    {
                        READER_CONTROL->setAssocPortObj(GEOPORT1D,myset);
                        
                    }
                }
            }
        }
        for (int n = 0; n < varInfos.size(); n++)
        {
            delete[] varInfos[n].dataObjs;
        }
    }
    return CONTINUE_PIPELINE;
}
int ReadCOMSOLData::compute(const char *)
{
    if (d_dataFile == NULL)
    {
        if (fileName.empty())
        {
            cerr << "ReadCOMSOLData::param(..) no data file found " << endl;
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
    READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.txt;*.TXT");

    READER_CONTROL->addOutputPort(MESHPORT3D, "geoOut_3D", "UnstructuredGrid", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_3D, "data1_3D", "Float|Vec3", "data1-3d");
    READER_CONTROL->addOutputPort(DPORT2_3D, "data2_3D", "Float|Vec3", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT3_3D, "data3_3D", "Float|Vec3", "data3-3d");
    READER_CONTROL->addOutputPort(DPORT4_3D, "data4_3D", "Float|Vec3", "data4-3d");
    READER_CONTROL->addOutputPort(DPORT5_3D, "data5_3D", "Float|Vec3", "data5-3d");

    READER_CONTROL->addOutputPort(GEOPORT2D, "geoOut_2D", "Polygons", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_2D, "data1_2D", "Float|Vec3", "data1-2d");
    READER_CONTROL->addOutputPort(DPORT2_2D, "data2_2D", "Float|Vec3", "data2-2d");
    READER_CONTROL->addOutputPort(DPORT3_2D, "data3_2D", "Float|Vec3", "data3-2d");
    READER_CONTROL->addOutputPort(DPORT4_2D, "data4_2D", "Float|Vec3", "data4-2d");
    READER_CONTROL->addOutputPort(DPORT5_2D, "data5_2D", "Float|Vec3", "data5-2d");

    READER_CONTROL->addOutputPort(GEOPORT1D, "geoOut_1D", "Points", "Measured points", false);

    READER_CONTROL->addOutputPort(DPORT1_1D, "data1_1D", "Float|Vec3", "data1-1d");
    READER_CONTROL->addOutputPort(DPORT2_1D, "data2_1D", "Float|Vec3", "data2-1d");
    READER_CONTROL->addOutputPort(DPORT3_1D, "data3_1D", "Float|Vec3", "data3-1d");
    READER_CONTROL->addOutputPort(DPORT4_1D, "data4_1D", "Float|Vec3", "data4-1d");
    READER_CONTROL->addOutputPort(DPORT5_1D, "data5_1D", "Float|Vec3", "data5-1d");

    // create the module
    coReader *application = new ReadCOMSOLData(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
