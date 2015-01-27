/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>

#include <map>
#include <string>

using namespace std;

enum DataType
{
    CONNECT = 0,
    VERTEXLIST,
    FL_SCALAR,
    FL_PAIR,
    FL_VECTOR,
    FL_TENSOR6,
    FL_TENSOR9,
    INT_SCALAR,
    INT_VECT3,
    GEOMETRY
};

const int TYPE_TETRAHEDER = 4;
const int TYPE_HEXAEDER = 7;

const int MAX_LINE = 65535;

/////////////////////////////////////////////////////
FILE *covOpenOutFile(const char *fieldname)
{
    char filename[256];
    strcpy(filename, fieldname);

    /// remove non-alpha chars
    char *chPtr = filename;
    while (*chPtr)
    {
        if (!isalpha(*chPtr) && !isdigit(*chPtr))
        {
            *chPtr++ = '@';
        }
        else
            chPtr++;
    }
    strcat(filename, ".covise");

    FILE *fi = fopen(filename, "w");
    if (fi == NULL)
    {
        perror(filename);
        return fi;
    }
#ifdef BYTESWAP
    fwrite("COV_BE", 6, 1, fi);
#else
    fwrite("COV_LE", 6, 1, fi);
#endif
    return fi;
}

/////////////////////////////////////////////////////
int covWriteSetBegin(FILE *fi, int numSteps)
{
    fwrite("SETELE", 6, 1, fi);

    return fwrite(&numSteps, sizeof(int), 1, fi);
}

/////////////////////////////////////////////////////
int covWriteAttrib(FILE *fi, int numAttrib,
                   const char *const *attrName,
                   const char *const *attrValue)
{
    int size = sizeof(int);
    int i;
    for (i = 0; i < numAttrib; i++)
        size += strlen(attrName[i]) + strlen(attrValue[i]) + 2;
    fwrite(&size, sizeof(int), 1, fi);
    fwrite(&numAttrib, sizeof(int), 1, fi);
    for (i = 0; i < numAttrib; i++)
    {
        fwrite(attrName[i], strlen(attrName[i]) + 1, 1, fi);
        fwrite(attrValue[i], strlen(attrValue[i]) + 1, 1, fi);
    }
    return 0;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
FILE *getfile(map<string, FILE *> &dataFileMap,
              const char *name,
              int numFiles)
{
    map<string, FILE *>::iterator iter = dataFileMap.find(name);

    // not found - create new file
    if (iter == dataFileMap.end())
    {
        FILE *fi = covOpenOutFile(name);
        if (numFiles > 1)
            covWriteSetBegin(fi, numFiles);
        dataFileMap[name] = fi;
        return fi;
    }
    else
        return iter->second;
}

int main(int argc, char *argv[])
{
    // map of open files, keyed by type
    map<string, FILE *> dataFileMap;
    map<string, FILE *>::iterator dataFileIter;

    // map of data types, keyed by type
    map<string, DataType> typeMap;
    map<string, DataType>::iterator typeMapIter;

    typeMap["ELMCON"] = CONNECT;
    typeMap["RZ"] = VERTEXLIST;

    typeMap["ACVCOF"] = FL_PAIR;
    typeMap["BCCANG"] = FL_TENSOR9;
    typeMap["BCCDEF"] = INT_VECT3;
    typeMap["BCCDEN"] = INT_VECT3;
    typeMap["BCCFNC"] = FL_PAIR;
    typeMap["BCCTFN"] = INT_SCALAR;
    typeMap["BCCTMP"] = INT_SCALAR;
    typeMap["DAMAGE"] = FL_SCALAR;
    typeMap["DENSTY"] = FL_SCALAR;
    typeMap["DRZ"] = FL_VECTOR;
    typeMap["EMSVTY"] = FL_PAIR;
    typeMap["ENVVTY"] = FL_PAIR;
    typeMap["FRZ"] = FL_VECTOR;
    typeMap["HDNTIM"] = FL_PAIR;
    typeMap["MTLGRP"] = INT_SCALAR;
    typeMap["NDFLUX"] = FL_SCALAR;
    typeMap["NDHEAT"] = FL_SCALAR;
    typeMap["NDTMP"] = FL_SCALAR;
    typeMap["PRZ"] = FL_VECTOR;
    typeMap["SPDLMT"] = FL_VECTOR;
    typeMap["STRAIN"] = FL_SCALAR;
    typeMap["URZ"] = FL_VECTOR;
    typeMap["USRELM"] = FL_VECTOR;
    typeMap["USRNOD"] = FL_VECTOR;

    typeMap["*STRESS"] = FL_SCALAR;
    typeMap["*STRESS_CMP"] = FL_TENSOR6;
    typeMap["*FRZCAL"] = FL_VECTOR;
    typeMap["*PRZCAL"] = FL_VECTOR;
    typeMap["*NDHEATC"] = FL_SCALAR;
    typeMap["*STRATE"] = FL_SCALAR;
    typeMap["*STRAIN_CMP"] = FL_TENSOR6;
    typeMap["*STRATE_CMP"] = FL_TENSOR6;

    int numFiles = argc - 1;

    // we MUST have a mesh
    FILE *meshFile = covOpenOutFile("Mesh");
    if (numFiles > 1)
        covWriteSetBegin(meshFile, numFiles);

    if (argc == 1)
    {
        cerr << "Call: " << argv[0] << " <file> <file> ..." << endl;
        exit(1);
    }

    // loop over files, create data files on-the-fly
    int fileNo;
    for (fileNo = 0; fileNo < numFiles; fileNo++)
    {

        // open input file
        FILE *fi = fopen(argv[fileNo + 1], "r");
        if (!fi)
        {
            perror(argv[fileNo + 1]);
            exit(0);
        }
        printf("====================================\nFile: %s\n\n", argv[fileNo + 1]);

        // buffer for reading
        char line[MAX_LINE + 1];
        line[MAX_LINE] = '\0'; //alway terminate line

        // read connectivity into this fields
        int numElem, numConn, numCoord;
        int *connList = NULL;
        float *xCoord = NULL;
        float *yCoord = NULL;
        float *zCoord = NULL;
        int *elemList;
        int *typeList;

        // not found yet
        numElem = 0;
        numConn = 0;
        numCoord = 0;

        // loop over fields
        do
        {

            // get next line
            fgets(line, MAX_LINE, fi);

            char name[4096];
            name[0] = '\0'; // clean up term in name field
            int numRead = sscanf(line, "%s", name);

            // try to match type from ident data
            typeMapIter = typeMap.find(name);

            if (numRead && typeMapIter != typeMap.end())
            {
                DataType dType = typeMapIter->second;

                switch (dType)
                {
                /////////////////////////////////////////////////////////////////////
                /// read connectivity into field
                case CONNECT:
                {
                    int objNo;
                    sscanf(line, "%s %d %d", name, &objNo, &numElem);
                    printf("%-12s Obj#%-4d  %8d Elements\n", name, objNo, numElem);
                    int elemNo;
                    // skip rubbish
                    char ch;
                    do
                    {
                        fread(&ch, 1, 1, fi);
                    } while (ch != ' ');

                    connList = new int[8 * numElem];
                    typeList = new int[numElem];
                    elemList = new int[numElem];
                    numConn = 0;

                    int i;
                    for (i = 0; i < numElem; i++)
                    {
                        fgets(line, MAX_LINE, fi);
                        int numRead = sscanf(line, "%d %d %d %d %d %d %d %d %d", &elemNo,
                                             &connList[numConn + 0], &connList[numConn + 1],
                                             &connList[numConn + 2], &connList[numConn + 3],
                                             &connList[numConn + 4], &connList[numConn + 5],
                                             &connList[numConn + 6], &connList[numConn + 7]);
                        if (elemNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected" << endl;
                            exit(1);
                        }

                        elemList[i] = numConn; // old value of numConn is stat

                        if (numRead == 5)
                        {
                            // TET element
                            typeList[i] = TYPE_TETRAHEDER;
                            numConn += 4;
                        }
                        else if (numRead == 9)
                        {
                            typeList[i] = TYPE_HEXAEDER;
                            elemList[i] = numConn; // old value of numConn is stat
                            numConn += 8;
                        }
                        else
                        {
                            cerr << "Found Illegal element in connectivity list " << endl;
                            exit(1);
                        }
                    }

                    // Covise starts counting at 0 !!!
                    for (i = 0; i < numConn; i++)
                        connList[i] -= 1;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                /// read vertices
                case VERTEXLIST:
                {
                    ////// Read Vertex coordinates
                    int objNo;
                    sscanf(line, "%s %d %d", name, &objNo, &numCoord);
                    printf("%-12s Obj#%-4d  %8d Coordinates\n", name, objNo, numCoord);

                    xCoord = new float[numCoord];
                    yCoord = new float[numCoord];
                    zCoord = new float[numCoord];
                    int i;
                    for (i = 0; i < numCoord; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %f %f %f", &vertNo, &xCoord[i], &yCoord[i], &zCoord[i]);
                        if (vertNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected" << endl;
                            exit(1);
                        }
                    }
                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case FL_VECTOR:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: Float Vector\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    float *xVal = new float[numVal];
                    float *yVal = new float[numVal];
                    float *zVal = new float[numVal];

                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %f %f %f\n", &vertNo, &xVal[i], &yVal[i], &zVal[i]);
                        if (vertNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected at " << i
                                 << " vertNo=" << vertNo << endl;
                            exit(1);
                        }
                    }
                    fwrite("USTVDT", 6, 1, outFile);
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(xVal, sizeof(float), numVal, outFile);
                    fwrite(yVal, sizeof(float), numVal, outFile);
                    fwrite(zVal, sizeof(float), numVal, outFile);

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);
                    delete[] xVal;
                    delete[] yVal;
                    delete[] zVal;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case FL_PAIR: // pack it into vector type
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: Float Vector\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    float *xVal = new float[numVal];
                    float *yVal = new float[numVal];
                    float *zVal = new float[numVal];

                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%f %f\n", &xVal[i], &yVal[i]);
                        zVal[i] = 0.0;
                    }
                    fwrite("USTVDT", 6, 1, outFile);
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(xVal, sizeof(float), numVal, outFile);
                    fwrite(yVal, sizeof(float), numVal, outFile);
                    fwrite(zVal, sizeof(float), numVal, outFile);

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);
                    delete[] xVal;
                    delete[] yVal;
                    delete[] zVal;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case FL_SCALAR:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: Float Scalar\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    float *xVal = new float[numVal];
                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %f\n", &vertNo, &xVal[i]);
                        if (vertNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected at " << i
                                 << " vertNo=" << vertNo << endl;
                            exit(1);
                        }
                    }
                    fwrite("USTSDT", 6, 1, outFile);
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(xVal, sizeof(float), numVal, outFile);

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);
                    delete[] xVal;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case FL_TENSOR6:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal;
                    int numTensVal = 6;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: 6D Tensor\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    fwrite("USTTDT", 6, 1, outFile);
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(&numTensVal, sizeof(int), 1, outFile);

                    float tens[6];
                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %f %f %f %f %f %f\n", &vertNo,
                               tens, tens + 1, tens + 2, tens + 3, tens + 4, tens + 5);
                        if (vertNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected at " << i
                                 << " vertNo=" << vertNo << endl;
                            exit(1);
                        }
                        fwrite(tens, sizeof(float), 6, outFile);
                    }

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case FL_TENSOR9:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal;
                    int numTensVal = 9;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: 9D Tensor\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    fwrite("USTTDT", 6, 1, outFile);
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(&numTensVal, sizeof(int), 1, outFile);

                    float tens[6];
                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %f %f %f %f %f %f %f %f %f\n", &vertNo,
                               tens, tens + 1, tens + 2, tens + 3, tens + 4, tens + 5, tens + 6, tens + 7, tens + 8);
                        if (vertNo != i + 1)
                        {
                            cerr << "non-continuous numbering detected at " << i
                                 << " vertNo=" << vertNo << endl;
                            cerr << line << endl;
                            exit(1);
                        }
                        fwrite(tens, sizeof(float), numTensVal, outFile);
                    }

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case INT_SCALAR:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal, dims = 1;
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: Int   Scalar\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    int *ival = new int[numVal];
                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %d\n", &vertNo, &ival[i]);
                    }
                    fwrite("INTARR", 6, 1, outFile);
                    // # of dims
                    fwrite(&dims, sizeof(int), 1, outFile);
                    // dimension of field
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    // sizes per dimension
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    fwrite(ival, sizeof(int), numElem, outFile);

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);
                    delete[] ival;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                case INT_VECT3:
                {
                    ////// read minimal and maximal index and allocate buffer
                    int objNo, numVal, dims = 2, dimArr[2] = { 0, 3 };
                    sscanf(line, "%s %d %d", name, &objNo, &numVal);
                    printf("%-12s Obj#%-4d  %8d Values: Int   Vector\n", name, objNo, numVal);

                    // get the file
                    FILE *outFile = getfile(dataFileMap, name, numFiles);

                    int *ival = new int[3 * numVal];
                    int i;
                    for (i = 0; i < numVal; i++)
                    {
                        int vertNo;
                        fgets(line, MAX_LINE, fi);
                        sscanf(line, "%d %d %d %d\n", &vertNo, &ival[3 * i], &ival[3 * i + 1], &ival[3 * i + 2]);
                    }

                    dimArr[0] = numVal;
                    numVal *= 3; // total number

                    fwrite("INTARR", 6, 1, outFile);
                    // # of dims
                    fwrite(&dims, sizeof(int), 1, outFile);
                    // dimension of field
                    fwrite(&numVal, sizeof(int), 1, outFile);
                    // sizes per dimension
                    fwrite(dimArr, sizeof(int), 1, outFile);
                    fwrite(ival, sizeof(int), numElem, outFile);

                    const char *attrName[] = { "SPECIES" };
                    const char *attrVal[] = { name };

                    covWriteAttrib(outFile, 1, attrName, attrVal);
                    delete[] ival;

                    break;
                }

                /////////////////////////////////////////////////////////////////////
                default:
                    cerr << "Identified type of " << typeMapIter->first
                         << " but not handled, type = " << typeMapIter->second << endl;
                    break;
                }
            }

        } while (!feof(fi));

        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        /// now create the grid here
        if (!connList)
        {
            cerr << "Did not find ELMCON section" << endl;
            exit(0);
        }

        ////// - write mesh

        fwrite("UNSGRD", 6, 1, meshFile);
        fwrite(&numElem, sizeof(int), 1, meshFile);
        fwrite(&numConn, sizeof(int), 1, meshFile);
        fwrite(&numCoord, sizeof(int), 1, meshFile);
        fwrite(elemList, sizeof(int), numElem, meshFile);
        fwrite(typeList, sizeof(int), numElem, meshFile);
        fwrite(connList, sizeof(int), numElem * 4, meshFile);
        fwrite(xCoord, sizeof(float), numCoord, meshFile);
        fwrite(yCoord, sizeof(float), numCoord, meshFile);
        fwrite(zCoord, sizeof(float), numCoord, meshFile);
        covWriteAttrib(meshFile, 0, NULL, NULL);

        delete[] elemList;
        delete[] typeList;
        delete[] connList;
        delete[] xCoord;
        delete[] yCoord;
        delete[] zCoord;

        cout << endl;

        // close input file
        fclose(fi);
    }

    // finish sets by writing their Attributes
    char buffer[64];
    sprintf(buffer, "0 %d", numFiles);
    const char *attrName[] = { "TIMESTEP", "SPECIES" };
    const char *attrVal[] = { buffer, "Mesh" };

    if (numFiles > 1)
        covWriteAttrib(meshFile, 1, attrName, attrVal);
    fclose(meshFile);

    for (dataFileIter = dataFileMap.begin();
         dataFileIter != dataFileMap.end();
         dataFileIter++)
    {
        attrVal[1] = dataFileIter->first.c_str();
        if (numFiles > 1)
            covWriteAttrib(dataFileIter->second, 2, attrName, attrVal);
        fclose(dataFileIter->second);
    }

    return 0;
}
