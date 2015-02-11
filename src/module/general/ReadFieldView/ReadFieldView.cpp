/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadFieldView
//
// This module Reads ASCII FieldView files as exported from Simulation CFD (Autodesk
//

#include "ReadFieldView.h"

#ifndef _WIN32
#include <inttypes.h>
#endif
#include <stdio.h>

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
ReadFieldView::ReadFieldView(int argc, char *argv[])
    : coReader(argc, argv, "Read FieldView mesh and data")
{
    d_dataFile = NULL;
}

ReadFieldView::~ReadFieldView()
{
}

// param callback read header again after all changes
void
ReadFieldView::param(const char *paramName, bool inMapLoading)
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
                    cerr << "ReadFieldView::param(..) no data file found " << endl;
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
                        cerr << "ReadFieldView::param(..) could not read file: " << dataNm << endl;
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
                cerr << "ReadFieldView::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

void ReadFieldView::nextLine()
{
    if (fgets(buf, sizeof(buf), d_dataFile) == NULL)
    {
        cerr << "ReadFieldView::nextLine: fgets failed" << endl;
    }
    currentBufPos = buf;
}

float ReadFieldView::readFloat()
{
    float data = atof(currentBufPos);
    while(*currentBufPos != '\0' && *currentBufPos != ' ')
        currentBufPos++;
    if(*currentBufPos == ' ')
        currentBufPos++;
    if(*currentBufPos == '\0' || *currentBufPos == '\n'|| *currentBufPos == '\r')
    {
        nextLine();
        currentBufPos = buf;
    }
    return data;
}

int ReadFieldView::getMeshSize()
{
    long int pos = ftell(d_dataFile);
    while (!feof(d_dataFile))
    {
        nextLine();
        if (strncmp(buf, "Nodes", 5) == 0) 
        {
            nextLine(); //
            sscanf(buf, "%d", &numPoints);
        }
        else if (strncmp(buf, "Elements", 8) == 0) 
        {
            numElements = 0;
            numVertices=0;
            while((strncmp(buf, "Variables", 9) != 0) )
            {
                nextLine();
                if(buf[0]!='V')
                {
                    int i1=0,i2=0;
                    sscanf(buf,"%d %d",&i1,&i2);
                    if(i1==1) // Tet
                    {
                        numVertices+=4;
                        numElements++;
                    }
                    else if(i1==3)// Prism
                    {
                        numVertices+=6;
                        numElements++;
                    }
                    else if(i1==4) // Pyramid
                    {
                        numVertices+=5;
                        numElements++;
                    }
                    else
                    {
                        fprintf(stderr,"unknown element %d\n",i1);
                    }
                }
            }
            fseek(d_dataFile,pos,SEEK_SET);
            return CONTINUE_PIPELINE;
        }
    }
    return STOP_PIPELINE;
}
int ReadFieldView::readHeader()
{
    while (!feof(d_dataFile))
    {
        nextLine();
        if (strncmp(buf, "Variable Names", 14) == 0) // read a dataset
        {
            nextLine(); //
            sscanf(buf, "%d", &numScalars);
            varInfos.clear();
			int varNum = 0;
            for (int i = 0; i < numScalars; i++)
            {
                nextLine();
				char *c = buf;
                while (*c != '\n' && *c != '\r'&& *c != '\0')
                    c++;
			    if(*c == '\n' || *c == '\r')
				    *c = '\0';
                c = buf;
                while (*c != ';' && *c != '\0')
                    c++;
                VarInfo vi;
				if(*c == ';') // we found a vector
                {
                    vi.valueIndex = varNum;
                    vi.components = 3;
                    varNum += 3;
                    i += 2;
                    vi.name = c+1;
                    nextLine();
                    nextLine();
                }
                else // a scalar
                {
                    vi.name = buf;
                    vi.components = 1;
                    vi.valueIndex = varNum;
                    varNum++;
                }
                varInfos.push_back(vi);
            }
            numTimesteps = 1; // TODO
            return CONTINUE_PIPELINE;
        }
    }
    return STOP_PIPELINE;
}

// taken from old ReadFieldView module: 2-Pass reading
int ReadFieldView::readASCIIData()
{
    char *cbuf;
    int res = readHeader();
    if (res != STOP_PIPELINE)
    {
        getMeshSize();   
        std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
        coDoUnstructuredGrid *grid = new coDoUnstructuredGrid(objNameBase.c_str(), numElements, numVertices, numPoints, 1);
        float *x_c;
        float *y_c;
        float *z_c;
        int *v_l;
        int *l_l;
        int *t_l;
        grid->getAddresses(&l_l, &v_l, &x_c, &y_c, &z_c);
        while (!feof(d_dataFile))
        {
            nextLine();
            if (strncmp(buf, "Nodes", 5) == 0) 
            {
                nextLine(); // numPoints

                for (int i = 0; i < numPoints; i++)
                {
                    nextLine();
                    sscanf(buf, "%f %f %f", x_c + i, y_c + i, z_c + i);
                }
                break;
            }
        }
        grid->getTypeList(&t_l);
        while (!feof(d_dataFile))
        {
            nextLine();
            if (strncmp(buf, "Elements", 8) == 0) 
            {
                numVertices=0;
                for(int i=0;i<numElements;i++)
                {
                    nextLine();
                    int i1,i2;
                    sscanf(buf,"%d %d",&i1,&i2);
                    if(i1==1)
                    {
                        sscanf(buf, "%d %d %d %d %d %d",&i1, &i2, v_l + (numVertices), v_l + (numVertices) + 1, v_l + (numVertices) + 2, v_l + (numVertices) + 3);
                        v_l[numVertices]--;
                        v_l[numVertices + 1]--;
                        v_l[numVertices + 2]--;
                        v_l[numVertices + 3]--;
                        l_l[i] = numVertices;
                        numVertices+=4;
                        t_l[i] = TYPE_TETRAHEDER;
                    }
                    else if(i1==3)
                    {
                        sscanf(buf, "%d %d %d %d %d %d %d %d",&i1, &i2, v_l + (numVertices), v_l + (numVertices) + 3, v_l + (numVertices) + 5, v_l + (numVertices) + 2, v_l + (numVertices) + 4, v_l + (numVertices) + 1);
                        v_l[numVertices]--;
                        v_l[numVertices + 1]--;
                        v_l[numVertices + 2]--;
                        v_l[numVertices + 3]--;
                        v_l[numVertices + 4]--;
                        v_l[numVertices + 5]--;
                        l_l[i] = numVertices;
                        numVertices+=6;
                        t_l[i] = TYPE_PRISM;
                    }
                    else if(i1==4)
                    {
                        sscanf(buf, "%d %d %d %d %d %d %d",&i1, &i2, v_l + (numVertices), v_l + (numVertices) + 1, v_l + (numVertices) + 2, v_l + (numVertices) + 3, v_l + (numVertices) + 4);
                        v_l[numVertices]--;
                        v_l[numVertices + 1]--;
                        v_l[numVertices + 2]--;
                        v_l[numVertices + 3]--;
                        v_l[numVertices + 4]--;
                        l_l[i] = numVertices;
                        numVertices+=5;
                        t_l[i] = TYPE_PYRAMID;
                    }

                }
                break;
            }
        }
        for (int n = 0; n < varInfos.size(); n++)
        {
            varInfos[n].dataObjs = new coDistributedObject *[numTimesteps + 1];
            varInfos[n].dataObjs[0] = NULL;
        }
        while (!feof(d_dataFile))
        {
            if (strncmp(buf, "Variables", 9) == 0) 
            {
                nextLine();
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
                            VarInfo vi;
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
                            varInfos[varIndex].objectName = READER_CONTROL->getAssocObjName(portID);
                            if (numTimesteps < 2)
                            {
                                objName = varInfos[varIndex].objectName;
                            }
                            else
                            {
                                char tmpName[500];
                                sprintf(tmpName, "%s_%d", varInfos[varIndex].objectName.c_str(), t);
                                objName = tmpName;
                            }
                            if (vi.components == 1)
                            {
                                coDoFloat *dataObj = new coDoFloat(objName.c_str(), numPoints);

                                varInfos[varIndex].dataObjs[t] = dataObj;
                                varInfos[varIndex].dataObjs[t + 1] = NULL;
                                float *x_d;
                                float *y_d;
                                float *z_d;
                                dataObj->getAddress(&x_d);
                                for (int i = 0; i < numPoints; i++)
                                {
                                    x_d[i] = readFloat();
                                }
                            }
                            else
                            {
                                coDoVec3 *dataObj = new coDoVec3(objName.c_str(), numPoints);
                                varInfos[varIndex].dataObjs[t] = dataObj;
                                varInfos[varIndex].dataObjs[t + 1] = NULL;
                                float *x_d;
                                float *y_d;
                                float *z_d;
                                dataObj->getAddresses(&x_d, &y_d, &z_d);
                                for (int p = 0; p < numPoints; p++)
                                {
                                    x_d[p] = readFloat();
                                }
                                i++; // next scalar
                                for (int p = 0; p < numPoints; p++)
                                {
                                    y_d[p] = readFloat();
                                }
                                if (vi.components == 3)
                                {
                                    i++; // next scalar
                                    for (int p = 0; p < numPoints; p++)
                                    {
                                        z_d[p] = readFloat();
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
                            for (int i = 0; i < numPoints; i++)
                            {
                                readFloat();
                            }
                        }
                    }
                }
            }
            nextLine();
        }
        if (numTimesteps > 1)
        {
            for (int n = 0; n < varInfos.size(); n++)
            {
                if (varInfos[n].dataObjs[0])
                {
                    coDoSet *myset = new coDoSet(varInfos[n].objectName, varInfos[n].dataObjs);
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
int ReadFieldView::compute(const char *)
{
    if (d_dataFile == NULL)
    {
        if (fileName.empty())
        {
            cerr << "ReadFieldView::param(..) no data file found " << endl;
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
    READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.fld;*.FLD");

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
    coReader *application = new ReadFieldView(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
