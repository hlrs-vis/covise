/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <fstream.h>
#include <iostream.h>
#include <stdio.h>
#include <strings.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

/////////////////////////////////////////////////////
FILE *covOpenOutFile(char *filename)
{
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
char *quote(char *str)
{
    char *start = strchr(str, '"') + 1;
    char *end = strchr(start, '"');
    *end = '\0';
    return start;
}

/////////////////////////////////////////////////////
int getInt(char *&cPtr)
{
    while (!isdigit(*cPtr) && !(*cPtr == '-') && *cPtr)
        cPtr++;
    int val = atoi(cPtr);

    while ((isdigit(*cPtr) || (*cPtr == '-')) && *cPtr)
        cPtr++;
    return val;
}

/////////////////////////////////////////////////////
void getField(ifstream &str, float *data, int numNodes)
{
    int i;
    for (i = 0; i < numNodes; i++)
        str >> data[i];
}

/////////////////////////////////////////////////////
void main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Call: " << argv[0] << " <filename.dat>" << endl;
        exit(0);
    }

    char species[256][64];

    ifstream infile(argv[1]);
    if (infile.bad())
    {
        cerr << "could not open " << argv[1] << " for input" << endl;
        exit(0);
    }
    char buffer[8192];
    infile.getline(buffer, 8192);
    // skip all before "VARIABLES"
    while (NULL == strstr(buffer, "VARIABLES"))
        infile.getline(buffer, 8192);

    int numSpecies = 0;
    while (NULL == strstr(buffer, "ZONE"))
    {
        strcpy(species[numSpecies], quote(buffer));
        cout << "species[" << numSpecies << "] = \"" << species[numSpecies] << "\"" << endl;

        numSpecies++;
        infile.getline(buffer, 8192);
    }

    infile.getline(buffer, 8192);

    char *cPtr = buffer;
    int numNodes = getInt(cPtr);
    int numElem = getInt(cPtr);

    cout << "-->  " << numNodes << " Nodes, " << numElem << " Elements" << endl;

    infile.getline(buffer, 8192); // skip one line

    ///////////////// read coordinates

    float *xCoord = new float[numNodes];
    getField(infile, xCoord, numNodes);

    float *yCoord = new float[numNodes];
    getField(infile, yCoord, numNodes);

    float *zCoord = new float[numNodes];
    getField(infile, zCoord, numNodes);

    ///////////////// read data fields

    const char *attrName[] = { "CREATOR", "FILENAME", "SPECIES" };
    const char *attrVal[] = { argv[0], argv[1], NULL };

    float *data = new float[numNodes];
    int spec;
    for (spec = 3; spec < numSpecies; spec++)
    {
        cout << "reading " << species[spec] << " with " << numNodes << " Elements" << endl;
        getField(infile, data, numNodes);

        char filename[256];
        char *fn = strrchr(argv[1], '/');
        if (!fn)
            fn = argv[1];
        else
            fn++;

        char *suff = strstr(fn, ".dat");
        if (suff)
            *suff = '\0';

        strcpy(filename, fn);
        fn = strrchr(argv[1], '/');
        strcat(filename, "::");
        strcat(filename, species[spec]);

        /// remove non-alpha chars in filenames
        char *chPtr = filename;
        while (*chPtr)
        {
            if (*chPtr == '(')
            {
                *chPtr = '\0';
                break;
            }

            if (!isalpha(*chPtr) && !isdigit(*chPtr) && ((*chPtr) != '_') && ((*chPtr) != ':'))
            {
                char rest[256];
                strcpy(rest, chPtr + 1); // do not use strcpy overlapping
                strcpy(chPtr, rest);
            }
            else
                chPtr++;
        }
        strcat(filename, ".covise");

        FILE *dataFile = covOpenOutFile(filename);

        fwrite("USTSDT", 6, 1, dataFile);
        fwrite(&numNodes, sizeof(int), 1, dataFile);
        fwrite(data, sizeof(float), numNodes, dataFile);

        attrVal[2] = species[spec];
        covWriteAttrib(dataFile, 3, attrName, attrVal);

        fclose(dataFile);
    }
    delete[] data;
    infile.getline(buffer, 8192); // skip to end of line
    cout << buffer << endl;

    ///////////////// read connectivity

    int *connList = new int[4 * numElem];
    int i;
    for (i = 0; i < numElem * 4; i++)
    {
        infile >> connList[i];
        connList[i] -= 1; //fortran
    }

    cout << connList[0] << " " << connList[1] << " "
         << connList[2] << " " << connList[3]
         << endl;

    cout << connList[4 * numElem - 4] << " " << connList[4 * numElem - 3] << " "
         << connList[4 * numElem - 2] << " " << connList[4 * numElem - 1]
         << endl;

    const int TYPE_TETRAHEDER = 4;

    int *elemList = new int[numElem];
    int *typeList = new int[numElem];
    for (i = 0; i < numElem; i++)
    {
        elemList[i] = i * 4;
        typeList[i] = TYPE_TETRAHEDER;
    }

    char filename[256];
    char *fn = strrchr(argv[1], '/');
    if (!fn)
        fn = argv[1];
    else
        fn++;
    strcpy(filename, fn);
    strcat(filename, "::Mesh.covise");
    int numConn = 4 * numElem;
    FILE *meshFile = covOpenOutFile(filename);
    fwrite("UNSGRD", 6, 1, meshFile);
    fwrite(&numElem, sizeof(int), 1, meshFile);
    fwrite(&numConn, sizeof(int), 1, meshFile);
    fwrite(&numNodes, sizeof(int), 1, meshFile);
    fwrite(elemList, sizeof(int), numElem, meshFile);
    fwrite(typeList, sizeof(int), numElem, meshFile);
    fwrite(connList, sizeof(int), numElem * 4, meshFile);
    fwrite(xCoord, sizeof(float), numNodes, meshFile);
    fwrite(yCoord, sizeof(float), numNodes, meshFile);
    fwrite(zCoord, sizeof(float), numNodes, meshFile);
    covWriteAttrib(meshFile, 2, attrName, attrVal);
}
