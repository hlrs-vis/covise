/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Simple Reader for Wavefront OBJ Format	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** History:                                                               **
 ** April 99         v1                                                    **
 ** September 99     new covise api                                        **                               **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadAbaqus.h"

void main(int argc, char *argv[])
{
    ReadAbaqus *application = new ReadAbaqus();
    application->start(argc, argv);
}

ReadAbaqus::ReadAbaqus()
    : coModule("Abaqus Reader")
{

    // the output ports
    p_mesh = addOutputPort("mesh", "coDoUnstructuredGrid", "mesh");
    p_data = addOutputPort("data", "coDoFloat|coDoVec3",
                           "data");

    // select the OBJ file name with a file browser
    p_fileParam = addFileBrowserParam("filename", "abaqus file");
    p_fileParam->setValue("data/ipt/filename", "*");
}

ReadAbaqus::~ReadAbaqus()
{
}

// =======================================================

int ReadAbaqus::compute()
{
    // open the file
    ReadAbaqus_Stream *abaqIn = new ReadAbaqus_Stream(p_fileParam->getValue());
    if (!abaqIn || !abaqIn->isGood())
        return FAIL;

    readFile(abaqIn);

    return SUCCESS;
}

// =======================================================
// =======================================================
// =======================================================

void ReadAbaqus::readFile(ReadAbaqus_Stream *abaqIn)
{
    char buffer[65536];

    // count connectivities and vertices;
    int numElem = 0, numConn = 0, numVert = 0, timesteps = 0;

    while (abaqIn->readRec(buffer))
    {
        int &type = *(int *)buffer;
        switch (type)
        {
        case 1900:
            numConn += 8;
            numElem += 1;
            break;

        case 1901:
            numVert += 1;
            break;

        case 2000:
            timesteps += 1;
            break;
        }
    }

    /// rewind to really read
    abaqIn->rewind();

    cerr << "numElem   = " << numElem << endl;
    cerr << "numVert   = " << numVert << endl;
    cerr << "timesteps = " << timesteps << "\n" << endl;

    if (timesteps == 0)
        timesteps = 1;

    // we need to map elements numbers to continuous
    int *vIdx = new int[numVert];

    // create grid obj

    char namebuf[512];
    sprintf(namebuf, "%s_0", p_mesh->getObjName());
    coDoUnstructuredGrid *grd
        = new coDoUnstructuredGrid(namebuf, numElem, numConn, numVert, 1);

    int *elem, *conn, *type, *iData;
    float *x, *y, *z;
    double *dData;

    grd->getAddresses(&elem, &conn, &x, &y, &z);
    grd->getTypeList(&type);

    int readConn = 0, readElem = 0, readVert = 0, readSteps = 0;
    iData = (int *)buffer;
    dData = (double *)buffer;

    ///////////////////////////////////////////////////////////////
    ///  2nd read pass - really read trhe mesh

    /// read elements with wrong numbers in shm -> translate later

    // read only grid definition
    while (abaqIn->readRec(buffer)
           && (readVert < numVert || readElem < numElem))
    {
        int &recType = *(int *)buffer;
        switch (recType)
        {
        case 1900:
            *elem = readConn;
            elem++;
            *type = TYPE_HEXAGON;
            type++;

            *conn = iData[6];
            conn++;
            *conn = iData[8];
            conn++;
            *conn = iData[10];
            conn++;
            *conn = iData[12];
            conn++;
            *conn = iData[14];
            conn++;
            *conn = iData[16];
            conn++;
            *conn = iData[18];
            conn++;
            *conn = iData[20];
            conn++;

            //eIdx[numElem] = iData[ 2];

            readConn += 8;
            readElem += 1;
            break;

        case 1901:

            *x = dData[2];
            x++;
            *y = dData[3];
            y++;
            *z = dData[4];
            z++;

            vIdx[readVert] = iData[2];

            readVert += 1;
            break;

        case 2000:
            readSteps += 1;
            break;
        }
    }

    /////// prepare translations
    int i;

    // find min and max Vertex number
    int minVert = vIdx[0], maxVert = vIdx[0];
    for (i = 1; i < numVert; i++)
    {
        if (vIdx[i] < minVert)
            minVert = vIdx[i];
        else if (vIdx[i] > maxVert)
            maxVert = vIdx[i];
    }
    cerr << "Min/Max Vert = " << minVert << " / " << maxVert << endl;

    // fast, but memory-consuming: inverse maps
    int *trVert = new int[maxVert - minVert + 1];
    for (i = 0; i < numVert; i++)
        trVert[vIdx[i] - minVert] = i;

    /// now: map it!
    grd->getAddresses(&elem, &conn, &x, &y, &z);

    for (i = 0; i < numConn; i++)
    {
        *conn = trVert[*conn - minVert];
        conn++;
    }

    ////////////// create grid set object
    coDistributedObject **objArr = new coDistributedObject *[timesteps + 1];
    objArr[timesteps] = NULL;
    objArr[0] = grd;
    for (i = 1; i < timesteps; i++)
    {
        objArr[i] = grd;
        grd->incRefCount();
    }
    coDoSet *set = new coDoSet(p_mesh->getObjName(), objArr);
    sprintf(namebuf, "1 %d", timesteps);
    set->addAttribute("TIMESTEP", namebuf);
    p_mesh->setCurrentObject(set);

    //////////////////////////////////////////////////////////////////////////
    ////////////// reading Grid is complete !!
    //////////////////////////////////////////////////////////////////////////

    // if we already 'overran' our data, start a new sweep
    if (readSteps)
        abaqIn->rewind();

    //////  @@@@@@@ this is bad: fixed to temperature
    //////                       only nodal data (no cell data) -> needs mapping too
    coDoFloat **dataArr = new coDoFloat *[timesteps + 1];
    dataArr[timesteps] = NULL;

    const char *dataName = p_data->getObjName();
    for (i = 0; i < timesteps; i++)
    {
        sprintf(namebuf, "%s_%d", dataName, i);
        objArr[i] = dataArr[i] = new coDoFloat(namebuf, numVert);
    }

    int actTimeStep = 0;
    float *value;

    iData = (int *)buffer;
    dData = (double *)buffer;
    while (abaqIn->readRec(buffer))
    {
        int &recType = *(int *)buffer;
        switch (recType)
        {
        /// Increment Start Record
        case 2000:
            dataArr[actTimeStep]->getAddress(&value);
            actTimeStep++;
            break;

        /// Temperature Record
        case 201:
            int &vertNo = iData[2];
            value[trVert[vertNo - minVert]] = dData[2];
            break;
        }
    }

    set = new coDoSet(p_data->getObjName(), objArr);
    sprintf(namebuf, "1 %d", timesteps);
    set->addAttribute("TIMESTEP", namebuf);
    p_data->setCurrentObject(set);

    delete[] vIdx;
}

static char *cvFilename(const char *filename)
{
    static char buffer[512];
    Covise::getname(buffer, filename);
    return buffer;
}

// =======================================================
// =======================================================
// =======================================================

ReadAbaqus_Stream::ReadAbaqus_Stream(const char *filename)
{
    char infobuf[300];

    // check whether we have a file
    d_filename = new char[512];
    Covise::getname(d_filename, filename);
    d_filePtr = new ifstream(d_filename);

    if (!d_filePtr || !*(d_filePtr))
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, filename);
        Covise::sendError(infobuf);
        return;
    }

    char start[4];
    d_filePtr->read(start, 2);

    if (!*d_filePtr)
        d_isBinary = -1; /// error
    else if (start[0] == '*' && start[1] == 'I')
    {
        d_filePtr->rdbuf()->seekpos(1, ios::beg); // skip 1st '*'
        d_isBinary = 0;
    }
    else
    {
        d_filePtr->rdbuf()->seekpos(0, ios::beg);
        d_isBinary = 1;
        d_filePtr->read((char *)&d_bytesInBlock, 4); // read 1st blocklen
    }
}

// read n bytes across FTN blocks
int ReadAbaqus_Stream::readBytesBin(void *buffer, int len)
{
    char *cBuf = (char *)buffer;
    while (len)
    {

        if (len > d_bytesInBlock)
        {
            d_filePtr->read(cBuf, d_bytesInBlock);
            cBuf += d_bytesInBlock;
            len -= d_bytesInBlock;
            // skip old
            d_filePtr->read((char *)&d_bytesInBlock, 4);
            // read new
            d_filePtr->read((char *)&d_bytesInBlock, 4);
        }
        else
        {
            d_filePtr->read(cBuf, len);
            d_bytesInBlock -= len;
            len = 0;
        }
    }

    return !d_filePtr->good();
}

// extract size from asc Buffer and advance Pointer
inline int readSizeASC(char *&bufPtr)
{
    int size;
    if (isdigit(bufPtr[0]))
        size = 10 * (bufPtr[0] - '0') + (bufPtr[1] - '0');
    else
        size = (bufPtr[1] - '0');
    bufPtr += 2;
    return size;
}

// Read Int element with given size
inline int readIntASC(char *&bufPtr)
{
    int size, res;

    size = readSizeASC(bufPtr);
    char saveChar = bufPtr[size]; // save char behind my field
    bufPtr[size] = '\0'; // and terminate string there

    res = atoi(bufPtr);

    bufPtr += size; // advance pointer ane
    *bufPtr = saveChar; // restore saved char

    return res;
}

// Read Float element with given size
inline float readFloatASC(char *&bufPtr)
{
    int size;
    float res;

    size = readSizeASC(bufPtr);
    char saveChar = bufPtr[size]; // save char behind my field
    bufPtr[size] = '\0'; // and terminate string there

    sscanf(bufPtr, "%f", &res);

    bufPtr += size; // advance pointer ane
    *bufPtr = saveChar; // restore saved char

    return res;
}

// Read Double element with given size
inline double readDoubleASC(char *&bufPtr)
{
    double res;
    char saveChar = bufPtr[22]; // save char behind my field
    bufPtr[22] = '\0'; // and terminate string there

    char *dPtr;
    if ((dPtr = strchr(bufPtr, 'D')) != NULL)
        *dPtr = 'e';

    sscanf(bufPtr, "%lf", &res);

    bufPtr += 22; // advance pointer ane
    *bufPtr = saveChar; // restore saved char

    return res;
}

// Read A8 element with given size : always size=8
inline const char *readStringASC(char *&bufPtr)
{
    static char res[9];
    char saveChar = bufPtr[8];
    bufPtr[8] = '\0'; // terminate string there

    strcpy(res, bufPtr);

    bufPtr += 8; // advance pointer ane
    *bufPtr = saveChar; // restore saved char

    return res;
}

// read one record, bin or ascii : return whether stream is ok
int ReadAbaqus_Stream::readRec(char *buffer)
{
    int length, dummy;

    if (d_isBinary)
    {
        readBytesBin(&length, 4);
        if (*d_filePtr)
        {
            readBytesBin(&dummy, 4);
            readBytesBin(buffer, 8 * (length - 1));
        }
    }
    else
    {
        char lineBuf[1024];

        // read till next '*'
        d_filePtr->getline(lineBuf, 1023, '*');
        if (*lineBuf == '\0')
            return 0;

        lineBuf[1023] = '\0'; // ensure termination

        // eliminate all non-printables (e.g. '\n')
        char *bufPtr = lineBuf;
        char *cpyPtr = lineBuf;
        while (*bufPtr)
        {
            if (*bufPtr != '\n')
            {
                *cpyPtr = *bufPtr;
                cpyPtr++;
            }
            bufPtr++;
        }

        // 1st is always int: number of fields : not returned
        bufPtr = lineBuf;
        if (*bufPtr != 'I')
        {
            coModule::sendError("Field must start wit '*I': '%s'", bufPtr);
            return 0;
        }
        bufPtr++;
        int numFields = readIntASC(bufPtr);

        // Pack the rest into the buffer

        int i;
        for (i = 1; i < numFields; i++) // field 0 already in (size)
        {
            char type = *bufPtr;
            bufPtr++;
            switch (type)
            {
            case 'I':
            {
                int val = readIntASC(bufPtr);
                *(int *)buffer = val;
                *(int *)(buffer + 4) = 0;
            }
            break;

            case 'F':
            {
                float val = readFloatASC(bufPtr);
                *(float *)buffer = val;
                *(int *)(buffer + 4) = 0;
            }
            break;

            case 'D':
            {
                double val = readDoubleASC(bufPtr);
                *(double *)buffer = val;
            }
            break;

            case 'A':
            {
                const char *val = readStringASC(bufPtr);
                strcpy(buffer, val);
            }
            break;

            default:
            {
                coModule::sendError("Illegal field type found: '%s'", bufPtr);
                return 0;
            }
            break;
            }
            buffer += 8;
        }

    } /// !is_binary

    return isGood();
}

void ReadAbaqus_Stream::rewind()
{
    delete d_filePtr;
    d_filePtr = new ifstream(d_filename);
    if (d_isBinary)
        d_filePtr->read((char *)&d_bytesInBlock, 4); // read 1st blocklen
    else
        d_filePtr->read((char *)&d_bytesInBlock, 1); // skip 1st char
}
