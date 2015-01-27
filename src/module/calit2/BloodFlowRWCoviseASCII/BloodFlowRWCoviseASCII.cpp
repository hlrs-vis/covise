/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2000 RUS  **
 **                                                                        **
 ** Description: Read / write ASCII data from / to files                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Gabor Duroska                               **
 **                Computer CenteRW_ASCII.cppr University of Stuttgart     **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  08.12.00  V0.1                                                  **
\**************************************************************************/

//Modified to accept directory as input and produce timesteps as output by Sasha Koruga (skoruga@ucsd.edu) on 10/5/09

#undef VERBOSE

#include "BloodFlowRWCoviseASCII.h"
#include <ctype.h>
#include <util/coviseCompat.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoGeometry.h>
#include <util/coFileUtil.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// constructor
RW_ASCII::RW_ASCII(int argc, char *argv[])
    : coModule(argc, argv, "Read/write COVISE ASCII files")
{
    // output port
    p_dataOut = addOutputPort("DataOut",
                              "IntArr|Polygons|Lines|Float|Vec3|UniformGrid|RectilinearGrid|TriangleStrips|StructuredGrid|UnstructuredGrid|Points|Vec3|Float|RGBA|USR_DistFenflossBoco",
                              "OutputObjects");

    // input port
    p_dataIn = addInputPort("DataIn",
                            "Geometry|IntArr|Polygons|Lines|Float|Vec3|UniformGrid|RectilinearGrid|TriangleStrips|StructuredGrid|UnstructuredGrid|Points|Vec3|Float|RGBA|USR_DistFenflossBoco",
                            "InputObjects");
    p_dataIn->setRequired(0);

    // select file name from browser
    p_filename = addFileBrowserParam("path", "ASCII file or directory consisting of ASCII files");
    p_filename->setValue(".", "*");

    p_newFile = addBooleanParam("new", "Create new file");
    p_newFile->setValue(1);
}

// get a line
static char *
getLine(istream &str, char *oBuffer, int numChars)
{
    char *buffer;

    do
    {
        buffer = oBuffer;

        // try to get a line
        *buffer = '\0';

        if (!str.getline(buffer, numChars - 1))
            return NULL;

        // make sure it terminated
        buffer[numChars - 1] = '\0';

        // remove leading blanks
        while ((*buffer) && (isspace(*buffer)))
            buffer++;

        // remove comments
        char *cPtr = strchr(buffer, '#');

        if (cPtr)
            *cPtr = '\0';

        // remove trailing blanks
        int len = strlen(buffer);

        if (len)
        {
            cPtr = buffer + len - 1;
            while ((*buffer) && (isspace(*cPtr)))
            {
                *cPtr = '\0';
                cPtr--;
            }
        }
    } while (*buffer == '\0');

    return buffer;
}

// search opening brace
static int
noOpenBrace(istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("ERROR: Object definition lacks opening brace");
        return 1;
    }
    else
        return 0;
}

// search closing brace
static int
noCloseBrace(istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);

    // closing brace
    if (!command || *command != '}')
    {
        Covise::sendError("ERROR: Object definition lacks closing brace");
        return 1;
    }
    else
        return 0;
}

// read attributes from file
static void
readAttrib(coDistributedObject *obj, char *buffer)
{

    // skip over attrib...
    while (*buffer && !isspace(*buffer))
        buffer++;

    // ...and all blanks behind
    while (*buffer && isspace(*buffer))
        buffer++;

    // now find end of name
    char *value = buffer;

    while (*value && !isspace(*value))
        value++;

    // terminate name field
    if (*value)
    {
        *value = '\0';
        value++;
    }

    // skip trailing blanks of value
    while (*value && isspace(*value))
        value++;

    // now set the attrib
    obj->addAttribute(buffer, value);
}

// readPOLYGN
coDistributedObject *
RW_ASCII::readPOLYGN(const char *name, char *command, istream &str)
{

    // get sizes
    int numPart, numConn, numVert;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPart, &numConn, &numVert) != 3)
    {

        // error in file format
        Covise::sendError("ERROR: POLYGN command needs 3 integer arguments: numLines, numConn, numVert");
        return NULL;
    }

    if (numPart < 1 || numConn < numPart)
    {

        // illegal sizes
        Covise::sendError("ERROR: POLYGN with illegal sizes: numLines: %d numConn: %d numVert: %d",
                          numPart, numConn, numVert);
        return NULL;
    }

    // create an object
    coDoPolygons *polygn = NULL;
    coDistributedObject *obj = NULL;
    int *partList, *connList;
    float *vx, *vy, *vz;

    polygn = new coDoPolygons(name, numVert, numConn, numPart);
    polygn->getAddresses(&vx, &vy, &vz, &connList, &partList);
    obj = polygn;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {

        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete polygn;
        return NULL;
    }

    int readVert = 0;
    int readConn = 0;
    int readPart = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readVert < numVert
                   && command
                   && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f%f%f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX definition: '%s",
                                      buffer);

                    // clean up
                    delete polygn;
                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readConn < numConn && readPart < numPart && command && (isdigit(*command) || (*command == '-')))
            {

                // element starts here
                *partList = readConn;
                readPart++;
                partList++;

                // split one line
                char *end, *nextComm;
                nextComm = command;

                while (*nextComm && (*nextComm != '#') && readConn < numConn)
                {
                    command = end = nextComm;

                    // skip forward to first non-number
                    while (*end && isdigit(*end))
                        end++;

                    if (*end)
                        nextComm = end + 1;
                    else
                        nextComm = end;

                    *end = '\0';

                    // read between
                    *connList = atoi(command);
                    connList++;
                    readConn++;

                    // skip forward to next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }

                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readConn == numConn && readPart == numPart && readVert == numVert)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete polygn;

        return NULL;
    }
}

// readLINES
coDistributedObject *
RW_ASCII::readLINES(const char *name, char *command, istream &str)
{

    // get sizes
    int numPart, numConn, numVert;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPart, &numConn, &numVert) != 3)
    {

        // error in file format
        Covise::sendError("ERROR: LINES command needs 3 integer arguments: numLines, numConn, numVert");
        return NULL;
    }

    if (numPart < 1 || numConn < numPart)
    {
        // illegal sizes
        Covise::sendError("ERROR: LINES with illegal sizes: numLines: %d numConn: %d numVert: %d",
                          numPart, numConn, numVert);
        return NULL;
    }

    // create an object
    coDoLines *lines = NULL;
    coDistributedObject *obj = NULL;
    int *partList, *connList;
    float *vx, *vy, *vz;

    lines = new coDoLines(name, numVert, numConn, numPart);
    lines->getAddresses(&vx, &vy, &vz, &connList, &partList);
    obj = lines;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete lines;
        return NULL;
    }

    int readVert = 0;
    int readConn = 0;
    int readPart = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readVert < numVert && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX definition: '%s",
                                      buffer);

                    // clean up
                    delete lines;
                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readConn < numConn && readPart < numPart && command && (isdigit(*command) || (*command == '-')))
            {

                // element starts here
                *partList = readConn;
                readPart++;
                partList++;

                // split one line
                char *end, *nextComm;
                nextComm = command;

                while (*nextComm && (*nextComm != '#') && readConn < numConn)
                {
                    command = end = nextComm;

                    // skip forward to first non-number
                    while (*end && isdigit(*end))
                        end++;

                    if (*end)
                        nextComm = end + 1;
                    else
                        nextComm = end;

                    *end = '\0';

                    // read between
                    *connList = atoi(command);
                    connList++;
                    readConn++;

                    // skip forward to next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }

                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readConn == numConn && readPart == numPart && readVert == numVert)
        return obj;
    else
    {
        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete lines;
        return NULL;
    }
}

// readTRIANG
coDistributedObject *
RW_ASCII::readTRIANG(const char *name, char *command, istream &str)
{
    // get sizes
    int numPoints, numCorners, numStrips;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPoints, &numCorners, &numStrips) != 3)
    {

        // error in file format
        Covise::sendError("ERROR: TRIANG command needs 3 integer arguments: numPoints, numCorners, numStrips");
        return NULL;
    }

    if (numPoints < 1 || numCorners < numPoints)
    {
        // illegal sizes
        Covise::sendError("ERROR: TRIANG with illegal sizes: numPoints: %d numCorners: %d numStrips: %d",
                          numPoints, numCorners, numStrips);
        return NULL;
    }

    // create an object
    coDoTriangleStrips *triang = NULL;
    coDistributedObject *obj = NULL;
    int *cornerList, *stripList;
    float *vx, *vy, *vz;

    triang = new coDoTriangleStrips(name, numPoints, numCorners, numStrips);
    triang->getAddresses(&vx, &vy, &vz, &cornerList, &stripList);
    obj = triang;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete triang;
        return NULL;
    }

    int readPoints = 0;
    int readCorners = 0;
    int readStrips = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readPoints < numPoints && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {
                    // error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX definition: %s", buffer);

                    // clean up
                    delete triang;
                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readPoints++;
            }
        }
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readCorners < numCorners && readStrips < numStrips && command && (isdigit(*command) || (*command == '-')))
            {
                // element starts here
                *stripList = readCorners;
                readStrips++;
                stripList++;

                // split one line
                char *end, *nextComm;

                nextComm = command;

                while (*nextComm && (*nextComm != '#') && readCorners < numCorners)
                {
                    command = end = nextComm;

                    // skip forward to first non-number
                    while (*end && isdigit(*end))
                        end++;

                    if (*end)
                        nextComm = end + 1;
                    else
                        nextComm = end;

                    *end = '\0';

                    // read between
                    *cornerList = atoi(command);
                    cornerList++;
                    readCorners++;

                    // skip forward to next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }
                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readPoints == numPoints && readCorners == numCorners && readStrips == numStrips)
        return obj;
    else
    {
        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete triang;
        return NULL;
    }
}

// readRGBADT
coDistributedObject *
RW_ASCII::readRGBADT(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        Covise::sendError("ERROR: RGBADT command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        Covise::sendError("ERROR: RGBA with illegal sizes: numElem: %d", numElem);
        return NULL;
    }

    // create an object
    int *cf;

    coDoRGBA *rgbadt = new coDoRGBA(name, numElem);

    rgbadt->getAddress(&cf);
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete rgbadt;
        return NULL;
    }

    int readElem = 0;

    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("DATA", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);
            while (readElem < numElem && command && (isdigit(*command)))
            {
                int d1, d2, d3, d4;
                if (sscanf(command, "%d %d %d %d", &d1, &d2, &d3, &d4) < 4)
                {
                    // an error occured in DATA definition
                    Covise::sendError("ERROR: Illegal read in DATA definition: '%s'", buffer);

                    // clean up
                    delete rgbadt;
                    return NULL;
                }
                *cf = (d1 << 24) | (d2 << 16) | (d3 << 8) | d4;
                cf++;
                command = getLine(str, buffer, 100000);
                readElem++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(rgbadt, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readElem == numElem)
        return rgbadt;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete rgbadt;

        return NULL;
    }
}

// readSETELE
coDistributedObject *
RW_ASCII::readSETELE(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        Covise::sendError("ERROR: SETELE command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        Covise::sendError("ERROR: SETELE with illegal sizes: numElem: %d", numElem);
        return NULL;
    }

    // create an object
    int i;
    char *attribs[100000];
    int numAttr = 0;
    coDistributedObject **objs = new coDistributedObject *[numElem + 1];

    objs[numElem] = NULL;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete[] objs;
        return NULL;
    }

    int readElem = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);
    while (*command && *command != '}')
    {
        if (strncasecmp("ELEM", command, 4) == 0)
        {
            if (noOpenBrace(str))
                return NULL;

            for (i = 0; i < numElem; i++)
            {
                char namebuf[100000];

                sprintf(namebuf, "%s_%d", name, i);
                objs[i] = readObj(namebuf, str);

                if (!objs[i] && i > 0)
                {
                    objs[i] = objs[i - 1];

                    if (objs[i])
                        objs[i]->incRefCount();
                }

                readElem++;
            }

            if (noCloseBrace(str))
                return NULL;

            command = getLine(str, buffer, 100000);
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            attribs[numAttr] = strcpy(new char[strlen(command) + 1], command);

            numAttr++;
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    coDoSet *setele = new coDoSet(name, objs);
    setele->addAttribute("TIMESTEP", "0 1");

    // clean up
    delete[] objs;

    for (i = 0; i < numAttr; i++)
    {
        readAttrib(setele, attribs[i]);

        // clean up
        delete[] attribs[i];
    }

    if (readElem == numElem)
        return setele;
    else
    {
        // clean up - something undefined
        Covise::sendError("ERROR: missing data in file");
        delete setele;
        return NULL;
    }
}

// readUSTVDT
coDistributedObject *
RW_ASCII::readUSTVDT(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        Covise::sendError("ERROR: USTVDT command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        Covise::sendError("ERROR: USTVDT with illegal sizes: numElem %d", numElem);
        return NULL;
    }

    // create an object
    float *vx, *vy, *vz;
    coDoVec3 *ustvdt = NULL;
    coDistributedObject *obj = NULL;

    ustvdt = new coDoVec3(name, numElem);
    ustvdt->getAddresses(&vx, &vy, &vz);
    obj = ustvdt;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete ustvdt;

        return NULL;
    }

    int readVal = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("DATA", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readVal < numElem && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // error occured in DATA definition
                    Covise::sendError("ERROR: Illegal read in DATA definition: '%s'", buffer);

                    // clean up
                    delete ustvdt;

                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVal++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readVal == numElem)
        return obj;
    else
    {

        // clean up - something undefined
        Covise::sendError("ERROR: missing data in file");
        delete ustvdt;

        return NULL;
    }
}

// readSTRSDT
coDistributedObject *
RW_ASCII::readSTRSDT(const char *name, char *command, istream &str)
{
    // get sizes
    char errBuf[600];
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;

    if (sscanf(command, "%d %d %d", &xSize, &ySize, &zSize) != 3)
    {

        // error in file format
        Covise::
            sendError("ERROR: STRSDT command needs 3 integer arguments: xSize, ySize, zSize");
        return NULL;
    }

    // create an object
    coDoFloat *strsdt = NULL;
    coDistributedObject *obj = NULL;
    float *v;
    int i = 0;

    strsdt = new coDoFloat(name, xSize * ySize * zSize);
    strsdt->getAddress(&v);
    obj = strsdt;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete strsdt;

        return NULL;
    }

    int readVal = 0;
    int numVal = xSize * ySize * zSize;

    // read until closing brace has been reached
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("DATA", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (i < numVal && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f", v) != 1))
                {

                    // an error occured in DATA definition
                    sprintf(errBuf,
                            "ERROR: Illegal read in DATA definition '%s'", buffer);

                    // clean up
                    delete strsdt;

                    return NULL;
                }

                i++;
                v++;
                readVal++;
                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (numVal == readVal)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete strsdt;

        return NULL;
    }
}

// readSTRVDT
coDistributedObject *
RW_ASCII::readSTRVDT(const char *name, char *command, istream &str)
{
    // get sizes
    char errBuf[600];
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;

    if (sscanf(command, "%d %d %d", &xSize, &ySize, &zSize) != 3)
    {
        // error in file format
        Covise::
            sendError("ERROR: STRVDT command needs 3 integer arguments: xSize, ySize, zSize");
        return NULL;
    }

    // create an object
    coDoVec3 *strvdt = NULL;
    coDistributedObject *obj = NULL;
    float *vx, *vy, *vz;
    int i = 0;

    strvdt = new coDoVec3(name, xSize * ySize * zSize);
    strvdt->getAddresses(&vx, &vy, &vz);
    obj = strvdt;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete strvdt;
        return NULL;
    }

    int readVal = 0;
    int numVal = xSize * ySize * zSize;

    // read until closing brace has been reached
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (i < numVal && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // an error occured in VERTEX definition
                    sprintf(errBuf,
                            "ERROR: Illegal read in VERTEX definition '%s'",
                            buffer);

                    // clean up
                    delete strvdt;

                    return NULL;
                }
                i++;
                vx++;
                vy++;
                vz++;
                readVal++;
                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (numVal == readVal)
        return obj;
    else
    {
        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete strvdt;

        return NULL;
    }
}

// readUSTSDT
coDistributedObject *
RW_ASCII::readUSTSDT(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        Covise::sendError("ERROR: USTSDT command needs 1 integer argument: numElem");
        return NULL;
    }
    if (numElem < 0)
    {
        // illegal sizes
        Covise::sendError("ERROR: USTSDT with illegal sizes: numElem: %d", numElem);
        return NULL;
    }

    // create an object
    float *vx;
    coDoFloat *ustsdt = NULL;
    coDistributedObject *obj = NULL;

    ustsdt = new coDoFloat(name, numElem);
    ustsdt->getAddress(&vx);
    obj = ustsdt;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete ustsdt;

        return NULL;
    }

    // read until closing brace has been reached or an error occured
    int readElem = 0;
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("DATA", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readElem < numElem && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if (sscanf(command, "%f", vx) < 1)
                {
                    // error occured in DATA definition
                    Covise::sendError("ERROR: Illegal read in DATA definition: '%s", buffer);

                    // clean up
                    delete ustsdt;

                    return NULL;
                }
                vx++;
                command = getLine(str, buffer, 100000);
                readElem++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }
    if (readElem == numElem)
        return obj;
    else
    {

        // clean up - something undefined
        Covise::sendError("ERROR: missing data in file");
        delete ustsdt;

        return NULL;
    }
}

// readUSTSDT
coDistributedObject *
RW_ASCII::readINTARR(const char *name, char *command, istream &str)
{
    // get sizes
    int numDim;
    char buffer[100000];
    int dim[8], i;

    if (sscanf(command, "%d  %d %d %d %d %d %d %d %d",
               &numDim,
               &dim[0], &dim[1], &dim[2], &dim[3],
               &dim[4], &dim[5], &dim[6], &dim[7]) < 2)
    {
        // error in file format
        Covise::sendError("ERROR: INTARR command needs at least 2 integer arguments");
        return NULL;
    }
    if (numDim < 1)
    {
        // illegal sizes
        Covise::sendError("ERROR: INTARR with illegal sizes: numElem: %d", numDim);
        return NULL;
    }

    // create an object
    int *vx;
    coDoIntArr *iarr = new coDoIntArr(name, numDim, dim);

    iarr->getAddress(&vx);
    int numElem = 1;

    for (i = 0; i < numDim; i++)
        numElem *= dim[i];

    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete iarr;
        return NULL;
    }

    // read until closing brace has been reached or an error occured
    int readElem = 0;
    command = getLine(str, buffer, 100000);
    while (command && *command && *command != '}')
    {
        if (strncasecmp("VALUES", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);
            while (readElem < numElem && command && (isdigit(*command) || (*command == '-')))
            {
                if (sscanf(command, "%i", vx) < 1)
                {

                    // error occured in DATA definition
                    Covise::sendError("ERROR: Illegal read in DATA definition: '%s", buffer);

                    // clean up
                    delete iarr;

                    return NULL;
                }

                vx++;
                command = getLine(str, buffer, 100000);
                readElem++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(iarr, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readElem == numElem)
        return iarr;
    else
    {
        // clean up - something undefined
        Covise::sendError("ERROR: missing data in file");
        delete iarr;
        return NULL;
    }
}

// readRCTGRD
coDistributedObject *
RW_ASCII::readRCTGRD(const char *name, char *command, istream &str)
{
    // get sizes
    char errBuf[600];
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;

    if (sscanf(command, "%d %d %d", &xSize, &ySize, &zSize) != 3)
    {

        // error in file format
        Covise::
            sendError("ERROR: STRGRD command needs 3 integer arguments: xSize, ySize, zSize");
        return NULL;
    }

    // create an object
    coDoRectilinearGrid *rctgrd = NULL;
    coDistributedObject *obj = NULL;
    float *vx, *vy, *vz;
    int i = 0, j = 0, k = 0;
    rctgrd = new coDoRectilinearGrid(name, xSize, ySize, zSize);
    rctgrd->getAddresses(&vx, &vy, &vz);
    obj = rctgrd;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete rctgrd;
        return NULL;
    }

    int readVal = 0;
    int numVal = xSize + ySize + zSize;

    // read until closing brace has been reached
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (i < xSize && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f", vx) != 1))
                {

                    // an error occured in VERTEX definition
                    sprintf(errBuf,
                            "ERROR: Illegal read in VERTEX definition '%s'",
                            buffer);

                    // clean up
                    delete rctgrd;

                    return NULL;
                }

                i++;
                vx++;
                readVal++;
                command = getLine(str, buffer, 100000);
            }

            while (j < ySize && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f", vy) != 1))
                {

                    // an error occured in VERTEX definition
                    sprintf(errBuf,
                            "ERROR: Illegal read in VERTEX definition '%s'",
                            buffer);

                    // clean up
                    delete rctgrd;

                    return NULL;
                }

                j++;
                vy++;
                readVal++;
                command = getLine(str, buffer, 100000);
            }

            while (k < zSize && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f", vz) != 1))
                {

                    // an error occured in VERTEX definition
                    sprintf(errBuf,
                            "ERROR: Illegal read in VERTEX definition '%s'",
                            buffer);

                    // clean up
                    delete rctgrd;

                    return NULL;
                }

                k++;
                vz++;
                readVal++;
                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (numVal == readVal)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete rctgrd;

        return NULL;
    }
}

// readSTRGRD
coDistributedObject *
RW_ASCII::readSTRGRD(const char *name, char *command, istream &str)
{

    // get sizes
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;

    if (sscanf(command, "%d %d %d", &xSize, &ySize, &zSize) != 3)
    {

        // error in file format
        Covise::
            sendError("ERROR: STRGRD command needs 3 integer arguments: xSize, ySize, zSize");
        return NULL;
    }

    if (xSize < 0 || ySize < 0 || zSize < 0)
    {

        // illegal size
        Covise::sendError("ERROR: STRGRD with illegal sizes");
        return NULL;
    }

    // create an object
    coDoStructuredGrid *strgrd = NULL;
    coDistributedObject *obj = NULL;
    float *vx, *vy, *vz;

    strgrd = new coDoStructuredGrid(name, xSize, ySize, zSize);
    strgrd->getAddresses(&vx, &vy, &vz);
    obj = strgrd;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {

        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete strgrd;

        return NULL;
    }

    int readVal = 0;
    int numVal = xSize * ySize * zSize;

    // read until closing brace has been reached
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readVal < numVal && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // an error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX defninition '%s'",
                                      buffer);

                    // clean up
                    delete strgrd;

                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVal++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (numVal == readVal)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete strgrd;

        return NULL;
    }
}

// readUNIGRD
coDistributedObject *
RW_ASCII::readUNIGRD(const char *name, char *command, istream &str)
{

    // get sizes
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;
    float xMin, xMax, yMin, yMax, zMin, zMax;

    if (sscanf(command, "%d%d%d%f%f%f%f%f%f", &xSize, &ySize, &zSize,
               &xMin, &xMax, &yMin, &yMax, &zMin, &zMax) != 9)
    {

        // error in file format
        Covise::sendError("ERROR: UNIGRD command needs 3 integer and 9 float arguments: xSize, ySize, zSize, xMin, xMax, yMin, yMax, zMin, zMax");
        return NULL;
    }

    if (xSize < 0 || ySize < 0 || zSize < 0)
    {

        // illegal sizes
        Covise::sendError("ERROR: UNIGRD with illegal sizes: xSize: %d ySize: %d zSize: %d",
                          xSize, ySize, zSize);
        return NULL;
    }

    // create an object
    coDoUniformGrid *grid = NULL;
    coDistributedObject *obj = NULL;

    grid = new coDoUniformGrid(name, xSize, ySize, zSize, xMin, xMax, yMin,
                               yMax, zMin, zMax);
    obj = grid;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {

        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete grid;
        delete obj;

        return NULL;
    }

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    return obj;
}

// readPOINTS
coDistributedObject *
RW_ASCII::readPOINTS(const char *name, char *command, istream &str)
{

    // get sizes
    int numVert;
    char buffer[100000];

    if (sscanf(command, "%d", &numVert) != 1)
    {
        // error in file format
        Covise::sendError("ERROR: POINTS command needs 1 integer argument: numVert");
        return NULL;
    }

    if (numVert < 1)
    {

        // illegal sizes
        Covise::sendError("ERROR: POINTS with illegal sizes: numVert: %d", numVert);
        return NULL;
    }

    // create an object
    float *vx, *vy, *vz;
    coDoPoints *points = NULL;
    coDistributedObject *obj = NULL;

    points = new coDoPoints(name, numVert);
    points->getAddresses(&vx, &vy, &vz);
    obj = points;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {

        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete points;

        return NULL;
    }

    int readVert = 0;

    // read until closing brace has been reached or an error occured
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);
            while (readVert < numVert && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {

                    // error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX definition: '%s'",
                                      buffer);

                    // clean up
                    delete points;

                    return NULL;
                }
                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readVert == numVert)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete points;

        return NULL;
    }
}

// readUNSGRD
coDistributedObject *
RW_ASCII::readUNSGRD(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem, numConn, numVert;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numElem, &numConn, &numVert) != 3)
    {

        // error in file format
        Covise::
            sendError("ERROR: UNSGRD command needs 3 integer arguments: numLines, numConn, numVert");
        return NULL;
    }

    if (numElem < 0 || numConn < numElem)
    {

        // illegal size
        Covise::sendError("ERROR: UNSGRD with illegal sizes");
        return NULL;
    }

    // create an object
    coDoUnstructuredGrid *unsgrd = NULL;
    coDistributedObject *obj = NULL;
    int *elemList, *connList, *typeList;
    float *vx, *vy, *vz;

    unsgrd = new coDoUnstructuredGrid(name, numElem, numConn, numVert, 1);
    unsgrd->getAddresses(&elemList, &connList, &vx, &vy, &vz);
    unsgrd->getTypeList(&typeList);
    obj = unsgrd;
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {

        // no opening brace found
        Covise::sendError("ERROR: Object definition lacks opening brace");

        // clean up
        delete unsgrd;

        return NULL;
    }

    int readVert = 0;
    int readConn = 0;
    int readPart = 0;

    // read until closing brace has been reached
    command = getLine(str, buffer, 100000);

    while (command && *command && *command != '}')
    {
        if (strncasecmp("VERTEX", command, 6) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readVert < numVert && command && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3) && (sscanf(command, "%f%f%f", vx, vy, vz) < 3))
                {
                    // an error occured in VERTEX definition
                    Covise::sendError("ERROR: Illegal read in VERTEX definition: '%s'",
                                      buffer);

                    // clean up
                    delete unsgrd;

                    return NULL;
                }

                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            command = getLine(str, buffer, 100000);

            while (readConn < numConn && readPart < numElem && command && isalpha(*command))
            {

                // the element starts here
                *elemList = readConn;
                readPart++;
                elemList++;

                // switch UNSGRD type
                if (strncasecmp("HEX", command, 3) == 0)
                    *typeList = 7;
                else if (strncasecmp("PRI", command, 3) == 0)
                    *typeList = 6;
                else if (strncasecmp("PYR", command, 3) == 0)
                    *typeList = 5;
                else if (strncasecmp("TET", command, 3) == 0)
                    *typeList = 4;
                else if (strncasecmp("QUA", command, 3) == 0)
                    *typeList = 3;
                else if (strncasecmp("TRI", command, 3) == 0)
                    *typeList = 2;
                else if (strncasecmp("BAR", command, 3) == 0)
                    *typeList = 1;
                else if (strncasecmp("POI", command, 3) == 0)
                    *typeList = 10;
                else
                    *typeList = 0;

                typeList++;

                while (*command && isalpha(*command))
                    command++;

                while (*command && isspace(*command))
                    command++;

                // split one line
                char *end, *nextComm;

                nextComm = command;

                while (*nextComm && (*nextComm != '#') && readConn < numConn)
                {
                    command = end = nextComm;

                    // skip forward to next non-number
                    while (*end && isdigit(*end))
                        end++;

                    if (*end)
                        nextComm = end + 1;
                    else
                        nextComm = end;

                    *end = '\0';

                    // read between
                    *connList = atoi(command);
                    connList++;
                    readConn++;

                    // skip forward to next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }

                command = getLine(str, buffer, 100000);
            }
        }
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {

// something undefined
#ifdef VERBOSE
            cerr << "WARNING: Ignoring line: '" << buffer << "'" << endl;
#endif
            command = getLine(str, buffer, 100000);
        }
    }

    if (readConn == numConn && readPart == numElem && readVert == numVert)
        return obj;
    else
    {

        // clean up - something went wrong
        Covise::sendError("ERROR: missing data in file");
        delete unsgrd;

        return NULL;
    }
}

// read an object from file
coDistributedObject *
RW_ASCII::readObj(const char *name, istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);
    char *param = command;

    // find command parameters if available
    while (param && *param && !isspace(*param))
        param++;

    while (param && *param && isspace(*param))
        param++;
    //std::cerr << "object type: ";
    // object types
    if (strncasecmp("LINES", command, 5) == 0)
    {
        //std::cerr << "LINES" << std::endl;
        return readLINES(name, param, str);
    }
    else if (strncasecmp("POLYGN", command, 6) == 0)
    {
        //std::cerr << "POLYGN" << std::endl;
        return readPOLYGN(name, param, str);
    }
    else if (strncasecmp("UNSGRD", command, 6) == 0)
    {
        //std::cerr << "UNSGRD" << std::endl;
        return readUNSGRD(name, param, str);
    }
    else if (strncasecmp("POINTS", command, 6) == 0)
    {
        //std::cerr << "POINTS" << std::endl;
        return readPOINTS(name, param, str);
    }
    else if (strncasecmp("USTSDT", command, 6) == 0)
    {
        //std::cerr << "USTSDT" << std::endl;
        return readUSTSDT(name, param, str);
    }
    else if (strncasecmp("USTVDT", command, 6) == 0)
    {
        //std::cerr << "USTVDT" << std::endl;
        return readUSTVDT(name, param, str);
    }
    else if (strncasecmp("SETELE", command, 6) == 0)
    {
        //std::cerr << "SETELE" << std::endl;
        return readSETELE(name, param, str);
    }
    else if (strncasecmp("TRIANG", command, 6) == 0)
    {
        //std::cerr << "TRIANG" << std::endl;
        return readTRIANG(name, param, str);
    }
    else if (strncasecmp("RGBADT", command, 6) == 0)
    {
        //std::cerr << "RGBADT" << std::endl;
        return readRGBADT(name, param, str);
    }
    else if (strncasecmp("UNIGRD", command, 6) == 0)
    {
        //std::cerr << "UNIGRD" << std::endl;
        return readUNIGRD(name, param, str);
    }
    else if (strncasecmp("RCTGRD", command, 6) == 0)
    {
        //std::cerr << "" << std::endl;
        return readRCTGRD(name, param, str);
    }
    else if (strncasecmp("STRGRD", command, 6) == 0)
    {
        //std::cerr << "STRGRD" << std::endl;
        return readSTRGRD(name, param, str);
    }
    else if (strncasecmp("STRSDT", command, 6) == 0)
    {
        //std::cerr << "STRSDT" << std::endl;
        return readSTRSDT(name, param, str);
    }
    else if (strncasecmp("STRVDT", command, 6) == 0)
    {
        //std::cerr << "STRVDT" << std::endl;
        return readSTRVDT(name, param, str);
    }
    else if (strncasecmp("INTARR", command, 6) == 0)
    {
        //std::cerr << "INTARR" << std::endl;
        return readINTARR(name, param, str);
    }
    //std::cerr << "NULL" << std::endl;
    return NULL;
}

// write an objet to file
void
RW_ASCII::writeObj(const char *offset, coDistributedObject *new_data, FILE *file)
{
    // get object type
    const char *type = new_data->getType();
    char *sp = (char *)"   ";
    int i, j, k, l;

    // switch object types
    if (strcmp(type, "LINES") == 0)
    {
        coDoLines *obj = (coDoLines *)new_data;
        int numE = obj->getNumLines();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();
        float *v[3];
        int *el, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);

        fprintf(file, "%sLINES %d %d %d\n", offset, numE, numC, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
        fprintf(file, "%s%sCONN\n", offset, sp);

        pos = *(el);
        *(el)++;
        next = *(el);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = pos; j < next; j++)
                fprintf(file, "%d ", *(cl)++);

            fprintf(file, "\n");
            pos = next;

            if (i == numE - 2)
                next = numC;
            else
            {
                *(el)++;
                next = *(el);
            }
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "POLYGN") == 0)
    {
        coDoPolygons *obj = (coDoPolygons *)new_data;
        int numE = obj->getNumPolygons();
        int numC = obj->getNumVertices();
        int numV = obj->getNumPoints();
        float *v[3];
        int *el, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &el);

        fprintf(file, "%sPOLYGN %d %d %d\n", offset, numE, numC, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
        fprintf(file, "%s%sCONN\n", offset, sp);

        pos = *(el);
        *(el)++;
        next = *(el);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = pos; j < next; j++)
                fprintf(file, "%d ", *(cl)++);

            fprintf(file, "\n");
            pos = next;

            if (i == numE - 2)
                next = numC;
            else
            {
                *(el)++;
                next = *(el);
            }
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "TRIANG") == 0)
    {
        coDoTriangleStrips *obj = (coDoTriangleStrips *)new_data;
        int numV = obj->getNumPoints();
        int numC = obj->getNumVertices();
        int numS = obj->getNumStrips();
        float *v[3];
        int *sl, *cl, pos, next, counter = 0;
        const char **name, **val;

        obj->getAddresses(&v[0], &v[1], &v[2], &cl, &sl);

        fprintf(file, "%sTRIANG %d %d %d\n", offset, numV, numC, numS);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
        fprintf(file, "%s%sCONN\n", offset, sp);

        pos = *(sl);
        *(sl)++;
        next = *(sl);

        for (i = 0; i < numS; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = pos; j < next; j++)
                fprintf(file, "%d ", *(cl)++);

            fprintf(file, "\n");
            pos = next;

            if (i == numS - 2)
                next = numC;
            else
            {
                *(sl)++;
                next = *(sl);
            }
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "SETELE") == 0)
    {
        int numElem;
        coDoSet *obj = (coDoSet *)new_data;
        coDistributedObject *const *elem = obj->getAllElements(&numElem);
        const char **name, **val;
        char *space = (char *)"      ";
        int counter = 0;

        fprintf(file, "SETELEM %d\n", numElem);
        fprintf(file, "{\n");

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%sELEM\n", sp);
        fprintf(file, "%s{\n", sp);

        for (i = 0; i < numElem; i++)
        {
            fprintf(file, "%s# elem number %d\n", space, i);
            writeObj(space, elem[i], file);

            if (i != numElem - 1)
                fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", sp);
        fprintf(file, "}\n");
    }
    else if (strcmp(type, "GEOMET") == 0)
    {
        int has_colors, has_normals, has_texture;
        coDoGeometry *obj = (coDoGeometry *)new_data;

        coDistributedObject *do1 = obj->getGeometry();
        coDistributedObject *do2 = obj->getColors();
        coDistributedObject *do3 = obj->getNormals();
        coDistributedObject *do4 = obj->getTexture();

        const int falseVal = 0;
        const int trueVal = 1;
        if (do2)
            has_colors = trueVal;
        else
            has_colors = falseVal;
        if (do3)
            has_normals = trueVal;
        else
            has_normals = falseVal;
        if (do4)
            has_texture = trueVal;
        else
            has_texture = falseVal;

        fprintf(file, "GEOMET\n");
        fprintf(file, "{\n");

        const char **name, **val;
        int counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }
        fprintf(file, "%sHAS_COLORS    %d\n", sp, has_colors);
        fprintf(file, "%sHAS_NORMALS   %d\n", sp, has_normals);
        fprintf(file, "%sHAS_TEXTURES  %d\n", sp, has_texture);

        fprintf(file, "%sELEM\n", sp);
        fprintf(file, "%s{\n", sp);

        const char *space = "      ";
        writeObj(space, do1, file);
        if (do2)
            writeObj(space, do2, file);
        if (do3)
            writeObj(space, do2, file);
        if (do4)
            writeObj(space, do2, file);

        fprintf(file, "%s}\n", sp);
        fprintf(file, "}\n");
    }
    else if (strcmp(type, "POINTS") == 0)
    {
        coDoPoints *obj = (coDoPoints *)new_data;
        int numV = obj->getNumPoints();
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sPOINTS %d\n", offset, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    /*   else if (strcmp (type, "SPHERE") == 0)
   {
      printf("in sphere");
      coDoSpheres *obj = (coDoSpheres *) new_data;
      int numV = obj->getNumSpheres ();
      float *v[3], *radius;
      char **name, **val;
      int counter = 0;

      obj->getAddresses (&v[0], &v[1], &v[2], &radius);

      fprintf (file, "%sSPHERE %d\n", offset, numV);
      fprintf (file, "%s{\n", offset);

      counter = obj->getAllAttributes (&name, &val);

      if (counter != 0)
      {
         for (i = 0; i < counter; i++)
            fprintf (file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

         fprintf (file, "\n");
      }

      fprintf (file, "%s%sVERTEX\n", offset, sp);

      for (i = 0; i < numV; i++)
      {
         fprintf (file, "%s%s%s", offset, sp, sp);

         for (j = 0; j < 3; j++)
         {
            if (((fabs (*v[j]) > 1e-6) && (fabs (*v[j]) < 1e12))
               || (*v[j] == 0.0))
               fprintf (file, "%f ", *(v[j])++);
            else
               fprintf (file, "%e ", *(v[j])++);
         }
         if (((fabs (*radius) > 1e-6) && (fabs (*radius) < 1e12)) || (*radius == 0.0))
            fprintf (file, "%f ", *radius);
         else
            fprintf (file, "%e ", *radius);
         fprintf (file, "\n");
      }

      fprintf (file, "%s}\n", offset);
   }*/

    else if (strcmp(type, "UNIGRD") == 0)
    {
        coDoUniformGrid *obj = (coDoUniformGrid *)new_data;
        int xSize, ySize, zSize;
        float xMin, yMin, zMin, xMax, yMax, zMax;
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);

        fprintf(file, "%sUNIGRD %d %d %d %f %f %f %f %f %f\n",
                offset, xSize, ySize, zSize, xMin, xMax, yMin, yMax, zMin, zMax);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "INTARR") == 0)
    {
        coDoIntArr *obj = (coDoIntArr *)new_data;
        int numDim = obj->getNumDimensions();
        int size = obj->getSize();

        int *dataArr = obj->getAddress();

        fprintf(file, "%sINTARR %d  ", offset, numDim);

        for (i = 0; i < numDim; i++)
            fprintf(file, " %d", obj->getDimension(i));

        fprintf(file, "\n%s{\n", offset);

        const char **name, **val;
        int counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVALUES\n", offset, sp);
        for (i = 0; i < size; i++)
            fprintf(file, "%s%s%s%d\n", offset, sp, sp, dataArr[i]);
        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "RCTGRD") == 0)
    {
        coDoRectilinearGrid *obj = (coDoRectilinearGrid *)new_data;
        int xSize, ySize, zSize;
        float *vx, *vy, *vz;
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getAddresses(&vx, &vy, &vz);
        fprintf(file, "%sRCTGRD %d %d %d\n", offset, xSize, ySize, zSize);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < xSize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vx[i]);
        }

        for (i = 0; i < ySize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vy[i]);
        }

        for (i = 0; i < zSize; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);
            fprintf(file, "%f\n", vz[i]);
        }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "STRGRD") == 0)
    {
        coDoStructuredGrid *obj = (coDoStructuredGrid *)new_data;
        int xSize, ySize, zSize;
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getGridSize(&xSize, &ySize, &zSize);
        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sSTRGRD %d %d %d\n", offset, xSize, ySize, zSize);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < xSize; i++)
            for (j = 0; j < ySize; j++)
                for (k = 0; k < zSize; k++)
                {
                    fprintf(file, "%s%s%s", offset, sp, sp);

                    for (l = 0; l < 3; l++)
                        if (((fabs(*v[l]) > 1e-6) && (fabs(*v[l]) < 1e12))
                            || (*v[l] == 0.0))
                            fprintf(file, "%f ", *(v[l])++);
                        else
                            fprintf(file, "%e ", *(v[l])++);

                    fprintf(file, "\n");
                }

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "USTVDT") == 0)
    {
        coDoVec3 *obj = (coDoVec3 *)new_data;
        int numE = obj->getNumPoints();
        float *v[3];
        const char **name, **val;
        int counter = 0;

        obj->getAddresses(&v[0], &v[1], &v[2]);

        fprintf(file, "%sUSTVDT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
            fprintf(file, "%s%s%s%f %f %f\n", offset, sp, sp,
                    *v[0]++, *v[1]++, *v[2]++);

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "USTSDT") == 0)
    {
        coDoFloat *obj = (coDoFloat *)new_data;
        int numE = obj->getNumPoints();
        float *v;
        const char **name, **val;
        int counter = 0;

        obj->getAddress(&v);

        fprintf(file, "%sUSTSDT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
            fprintf(file, "%s%s%s%f\n", offset, sp, sp, *v++);

        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "RGBADT") == 0)
    {
        coDoRGBA *obj = (coDoRGBA *)new_data;
        int numE = obj->getNumPoints();
        const char **name, **val;
        int counter = 0;

        unsigned int *data;

        obj->getAddress((int **)(void *)&data);

        fprintf(file, "%sRGBADT %d\n", offset, numE);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sDATA\n", offset, sp);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s%d %d %d %d\n", offset, sp, sp,
                    ((*data) & 0xff000000) >> 24,
                    ((*data) & 0x00ff0000) >> 16,
                    ((*data) & 0x0000ff00) >> 8, ((*data) & 0x000000ff));
            data++;
        }
        fprintf(file, "%s}\n", offset);
    }
    else if (strcmp(type, "UNSGRD") == 0)
    {
        coDoUnstructuredGrid *obj = (coDoUnstructuredGrid *)new_data;
        int numE, numC, numV, ht;
        float *v[3];
        int *el, *cl, *tl, counter = 0;
        const char **name, **val;

        obj->getGridSize(&numE, &numC, &numV);
        ht = obj->hasTypeList();
        obj->getAddresses(&el, &cl, &v[0], &v[1], &v[2]);
        obj->getTypeList(&tl);
        static const char *type[] = {
            "NONE", "BAR", "TRI", "QUA", "TET", "PYR",
            "PRI", "HEX", "POI"
        };

        fprintf(file, "%sUNSGRD %d %d %d\n", offset, numE, numC, numV);
        fprintf(file, "%s{\n", offset);

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "%s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);

            fprintf(file, "\n");
        }

        fprintf(file, "%s%sVERTEX\n", offset, sp);

        for (i = 0; i < numV; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            for (j = 0; j < 3; j++)
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);

            fprintf(file, "\n");
        }

        fprintf(file, "\n");

        fprintf(file, "%s%sCONN\n", offset, sp);

        for (i = 0; i < numE; i++)
        {
            fprintf(file, "%s%s%s", offset, sp, sp);

            if (ht)
            {
                if (tl[i] > -1 && tl[i] < 9)
                    if (type[tl[i]] == (char *)"10")
                        fprintf(file, "%s ", type[8]);
            }
            else
                fprintf(file, "%s ", type[tl[i]]);

            for (j = el[i]; j < ((i < numE - 1) ? el[i + 1] : numC); j++)
                fprintf(file, "%i ", cl[j]);

            fprintf(file, "\n");
        }

        fprintf(file, "%s}\n", offset);
    }
    else
    {
        Covise::sendError("ERROR: Object type not supported by COVISE: '%s'",
                          type);
        return;
    }
}

// compute callback
int
RW_ASCII::compute(const char *)
{

    const char *DataIn = Covise::get_object_name("DataIn");

    if (DataIn != NULL)
    {

        // write data to file
        const char *dataPath = p_filename->getValue();

        //Covise::get_browser_param ("path", &dataPath);

        if (dataPath == NULL)
        {

            // invalid data path
            Covise::sendError("ERROR: Could not get filename");
            return STOP_PIPELINE;
        }

        int newFile = p_newFile->getValue();

        FILE *file = Covise::fopen(dataPath, (newFile ? "w" : "a"));
#ifdef _WIN32
        if (!file)
#else
        if ((!file) || (fseek(file, 0, SEEK_CUR)))
#endif
        {

            // can't create this file
            Covise::sendError("ERROR: Could not create file");
            return STOP_PIPELINE;
        }

        // get input data
        char *inputName = Covise::get_object_name("DataIn");

        if (inputName == NULL)
        {

            // can't create object
            Covise::
                sendError("ERROR: Could not create object name for 'dataIn'");

            // close file
            fclose(file);
            return STOP_PIPELINE;
        }

        coDistributedObject *new_data = coDistributedObject::createFromShm(inputName);
        if (new_data == NULL)
        {

            // object creation failed
            Covise::sendError("ERROR: createFromShm() failed for data");

            // close file
            fclose(file);
            return STOP_PIPELINE;
        }

        // write obj
        writeObj("", new_data, file);

        // clean up
        delete new_data;

        // close file
        fclose(file);
    }
    else
    {

        coDirectory *dir = coDirectory::open(p_filename->getValue());
        if (!dir) //check to see if input is directory or just a particular file
        { //if just one file
            // read data from file -- ORIGINAL CODE
            const char *objName = p_dataOut->getObjName();
            ifstream inputFile(p_filename->getValue());

            if (!inputFile.good())
            {
                Covise::sendError("%s: %s", p_filename->getValue(), strerror(errno));
                return STOP_PIPELINE;
            }

            coDistributedObject *obj = readObj(objName, inputFile);

            if (obj)
            {
                p_dataOut->setCurrentObject(obj);
                return CONTINUE_PIPELINE;
            }
            else
                return STOP_PIPELINE;
        }
        else
        { //if directory, load all files in directory as timestep data

            //generate a list of files in the directory
            const std::string directory = p_filename->getValue();
            std::list<std::string> files;
            for (int i = 0; i < dir->count(); ++i)
            {
                std::string name = dir->name(i);
                if (name[0] != '.' && *(--(name.end())) != '~')
                    files.push_back(dir->name(i));
            }
            dir->close();
            delete dir;

            if (!files.size()) //no files in the directory
            {
                std::cerr << "No files were found at " << directory << std::endl;
                return STOP_PIPELINE;
            }

            //load each file and save it as a timestep
            coDistributedObject **outobjs = new coDistributedObject *[files.size() + 1];
            int counter = -1;
            for (std::list<std::string>::iterator iter = files.begin(); iter != files.end(); ++iter)
            {
                std::string inFile = directory;
                inFile += *iter;
                ifstream fileInput(inFile.c_str());
                coDistributedObject *object = readObj("object", fileInput);
                outobjs[++counter] = object;
            }
            outobjs[++counter] = NULL;

            coDoSet *set_output = new coDoSet(p_dataOut->getObjName(), outobjs);
            char buf[128];
            sprintf(buf, "1 %zu", files.size());
            set_output->addAttribute("TIMESTEP", buf);

            if (set_output)
            {
                p_dataOut->setCurrentObject(set_output);

                //clean up
                for (int i = 0; i < files.size(); ++i)
                    delete outobjs[i];
                delete outobjs;
                return CONTINUE_PIPELINE;
            }
            else
                return STOP_PIPELINE;
        }
    }

    return STOP_PIPELINE;
}

// destructor
RW_ASCII::~RW_ASCII()
{
}

MODULE_MAIN(IO, RW_ASCII)
