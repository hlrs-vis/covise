/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: Write Polygondata to files (VRML, STL, ***)               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Bruno Burbaum                               **
 **                                 HLRS                                   **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  08.10.08  V0.1                                                  **
\**************************************************************************/
// else if (strcmp (type, "SETELE ist geschlossen ") == 0)     //    "SETELE" ist geschlossen

#undef VERBOSE

#include "WritePolygon.h"
#include <ctype.h>
#include <util/coviseCompat.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoGeometry.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// constructor
WritePolygon::WritePolygon(int argc, char *argv[])
    : coModule(argc, argv, "Read/write COVISE ASCII files")
{
    //   // output port
    //   p_dataOut = addOutputPort ("DataOut",
    //      "IntArr|Polygons|Lines|Float|Vec3|UniformGrid|RectilinearGrid|TriangleStrips|StructuredGrid|UnstructuredGrid|Points|Vec3|Float|RGBA|USR_DistFenflossBoco",
    //      "OutputObjects");

    // input port
    p_dataIn = addInputPort("DataIn",
                            "Geometry|IntArr|Polygons|Lines|Float|Vec3|UniformGrid|RectilinearGrid|TriangleStrips|StructuredGrid|UnstructuredGrid|Points|Vec3|Float|RGBA|USR_DistFenflossBoco",
                            "InputObjects");
    p_dataIn->setRequired(0);

    // select file name from browser
    p_filename = addFileBrowserParam("path", "ASCII file");
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
WritePolygon::readPOLYGN(const char *name, char *command, istream &str)
{

    // get sizes
    int numPart, numConn, numVert;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPart, &numConn, &numVert) != 3)
    {

        // error in file format
        sendError("ERROR: POLYGN command needs 3 integer arguments: numLines, numConn, numVert");
        return NULL;
    }

    if (numPart < 1 || numConn < numPart)
    {

        // illegal sizes
        sendError(
            "ERROR: POLYGN with illegal sizes: numLines: %d numConn: %d numVert: %d",
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError("ERROR: Illegal read in VERTEX definition: '%s", buffer);

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
        sendError("ERROR: missing data in file");
        delete polygn;

        return NULL;
    }
}

// readLINES
coDistributedObject *
WritePolygon::readLINES(const char *name, char *command, istream &str)
{

    // get sizes
    int numPart, numConn, numVert;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPart, &numConn, &numVert) != 3)
    {

        // error in file format
        sendError(
            "ERROR: LINES command needs 3 integer arguments: numLines, numConn, numVert");
        return NULL;
    }

    if (numPart < 1 || numConn < numPart)
    {
        // illegal sizes
        sendError(
            "ERROR: LINES with illegal sizes: numLines: %d numConn: %d numVert: %d",
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError("ERROR: Illegal read in VERTEX definition: '%s",
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
        sendError("ERROR: missing data in file");
        delete lines;
        return NULL;
    }
}

// readTRIANG
coDistributedObject *
WritePolygon::readTRIANG(const char *name, char *command, istream &str)
{
    // get sizes
    int numPoints, numCorners, numStrips;
    char buffer[100000];

    if (sscanf(command, "%d%d%d", &numPoints, &numCorners, &numStrips) != 3)
    {

        // error in file format
        sendError(
            "ERROR: TRIANG command needs 3 integer arguments: numPoints, numCorners, numStrips");
        return NULL;
    }

    if (numPoints < 1 || numCorners < numPoints)
    {
        // illegal sizes
        sendError(
            "ERROR: TRIANG with illegal sizes: numPoints: %d numCorners: %d numStrips: %d",
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in VERTEX definition: %s", buffer);

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
        sendError("ERROR: missing data in file");
        delete triang;
        return NULL;
    }
}

// readRGBADT
coDistributedObject *
WritePolygon::readRGBADT(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        sendError(
            "ERROR: RGBADT command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        sendError("ERROR: RGBA with illegal sizes: numElem: %d", numElem);
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in DATA definition: '%s'", buffer);

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
        sendError("ERROR: missing data in file");
        delete rgbadt;

        return NULL;
    }
}

// readSETELE
coDistributedObject *
WritePolygon::readSETELE(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        sendError(
            "ERROR: SETELE command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        sendError(
            "ERROR: SETELE with illegal sizes: numElem: %d", numElem);
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
        sendError("ERROR: Object definition lacks opening brace");

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
        sendError("ERROR: missing data in file");
        delete setele;
        return NULL;
    }
}

// readUSTVDT
coDistributedObject *
WritePolygon::readUSTVDT(const char *name, char *command, istream &str)
{
    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        sendError(
            "ERROR: USTVDT command needs 1 integer argument: numElem");
        return NULL;
    }

    if (numElem < 0)
    {
        // illegal sizes
        sendError("ERROR: USTVDT with illegal sizes: numElem %d", numElem);
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in DATA definition: '%s'", buffer);

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
        sendError("ERROR: missing data in file");
        delete ustvdt;

        return NULL;
    }
}

// readSTRSDT
coDistributedObject *
WritePolygon::readSTRSDT(const char *name, char *command, istream &str)
{
    // get sizes
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;

    if (sscanf(command, "%d %d %d", &xSize, &ySize, &zSize) != 3)
    {

        // error in file format
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
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
        sendError("ERROR: missing data in file");
        delete strsdt;

        return NULL;
    }
}

// readSTRVDT
coDistributedObject *
WritePolygon::readSTRVDT(const char *name, char *command, istream &str)
{
    // get sizes
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
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
        sendError("ERROR: missing data in file");
        delete strvdt;

        return NULL;
    }
}

// readUSTSDT
coDistributedObject *
WritePolygon::readUSTSDT(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem;
    char buffer[100000];

    if (sscanf(command, "%d", &numElem) != 1)
    {
        // error in file format
        sendError(
            "ERROR: USTSDT command needs 1 integer argument: numElem");
        return NULL;
    }
    if (numElem < 0)
    {
        // illegal sizes
        sendError(
            "ERROR: USTSDT with illegal sizes: numElem: %d", numElem);
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in DATA definition: '%s", buffer);

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
        sendError("ERROR: missing data in file");
        delete ustsdt;

        return NULL;
    }
}

// readUSTSDT
coDistributedObject *
WritePolygon::readINTARR(const char *name, char *command, istream &str)
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
        sendError(
            "ERROR: INTARR command needs at least 2 integer arguments");
        return NULL;
    }
    if (numDim < 1)
    {
        // illegal sizes
        sendError("ERROR: INTARR with illegal sizes: numElem: %d", numDim);
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in DATA definition: '%s", buffer);

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
        sendError("ERROR: missing data in file");
        delete iarr;
        return NULL;
    }
}

// readRCTGRD
coDistributedObject *
WritePolygon::readRCTGRD(const char *name, char *command, istream &str)
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
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
                    sendError(
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
                    sendError(
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
        sendError("ERROR: missing data in file");
        delete rctgrd;

        return NULL;
    }
}

// readSTRGRD
coDistributedObject *
WritePolygon::readSTRGRD(const char *name, char *command, istream &str)
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
        sendError("ERROR: STRGRD with illegal sizes");
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in VERTEX defninition '%s'",
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
        sendError("ERROR: missing data in file");
        delete strgrd;

        return NULL;
    }
}

// readUNIGRD
coDistributedObject *
WritePolygon::readUNIGRD(const char *name, char *command, istream &str)
{

    // get sizes
    char buffer[100000];
    int xSize = 0, ySize = 0, zSize = 0;
    float xMin, xMax, yMin, yMax, zMin, zMax;

    if (sscanf(command, "%d%d%d%f%f%f%f%f%f", &xSize, &ySize, &zSize,
               &xMin, &xMax, &yMin, &yMax, &zMin, &zMax) != 9)
    {

        // error in file format
        sendError(
            "ERROR: UNIGRD command needs 3 integer and 9 float arguments: xSize, ySize, zSize, xMin, xMax, yMin, yMax, zMin, zMax");
        return NULL;
    }

    if (xSize < 0 || ySize < 0 || zSize < 0)
    {

        // illegal sizes
        sendError(
            "ERROR: UNIGRD with illegal sizes: xSize: %d ySize: %d zSize: %d",
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
        sendError("ERROR: Object definition lacks opening brace");

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
WritePolygon::readPOINTS(const char *name, char *command, istream &str)
{

    // get sizes
    int numVert;
    char buffer[100000];

    if (sscanf(command, "%d", &numVert) != 1)
    {
        // error in file format
        sendError(
            "ERROR: POINTS command needs 1 integer argument: numVert");
        return NULL;
    }

    if (numVert < 1)
    {

        // illegal sizes
        sendError(
            "ERROR: POINTS with illegal sizes: numVert: %d", numVert);
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in VERTEX definition: '%s'",
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
        sendError("ERROR: missing data in file");
        delete points;

        return NULL;
    }
}

// readUNSGRD
coDistributedObject *
WritePolygon::readUNSGRD(const char *name, char *command, istream &str)
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
        sendError("ERROR: UNSGRD with illegal sizes");
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
        sendError("ERROR: Object definition lacks opening brace");

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
                    sendError(
                        "ERROR: Illegal read in VERTEX definition: '%s'",
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
        sendError("ERROR: missing data in file");
        delete unsgrd;

        return NULL;
    }
}

// read an object from file
coDistributedObject *
WritePolygon::readObj(const char *name, istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);
    char *param = command;

    // find command parameters if available
    while (param && *param && !isspace(*param))
        param++;

    while (param && *param && isspace(*param))
        param++;

    // object types
    if (strncasecmp("LINES", command, 5) == 0)
        return readLINES(name, param, str);
    else if (strncasecmp("POLYGN", command, 6) == 0)
        return readPOLYGN(name, param, str);
    else if (strncasecmp("UNSGRD", command, 6) == 0)
        return readUNSGRD(name, param, str);
    else if (strncasecmp("POINTS", command, 6) == 0)
        return readPOINTS(name, param, str);
    else if (strncasecmp("USTSDT", command, 6) == 0)
        return readUSTSDT(name, param, str);
    else if (strncasecmp("USTVDT", command, 6) == 0)
        return readUSTVDT(name, param, str);
    else if (strncasecmp("SETELE", command, 6) == 0)
        return readSETELE(name, param, str);
    else if (strncasecmp("TRIANG", command, 6) == 0)
        return readTRIANG(name, param, str);
    else if (strncasecmp("RGBADT", command, 6) == 0)
        return readRGBADT(name, param, str);
    else if (strncasecmp("UNIGRD", command, 6) == 0)
        return readUNIGRD(name, param, str);
    else if (strncasecmp("RCTGRD", command, 6) == 0)
        return readRCTGRD(name, param, str);
    else if (strncasecmp("STRGRD", command, 6) == 0)
        return readSTRGRD(name, param, str);
    else if (strncasecmp("STRSDT", command, 6) == 0)
        return readSTRSDT(name, param, str);
    else if (strncasecmp("STRVDT", command, 6) == 0)
        return readSTRVDT(name, param, str);
    else if (strncasecmp("INTARR", command, 6) == 0)
        return readINTARR(name, param, str);

    return NULL;
}

// write an objet to file
void
WritePolygon::writeObj(const char *offset, const coDistributedObject *new_data, FILE *file)
{
    // get object type
    const char *type = new_data->getType();
    char *sp = (char *)"   ";
    int i, j, k, l;

    // switch object types
    // Uwe:  Was ist wenn gettype versagt hat und type = leer ist
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
        el++;
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
                el++;
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
        fprintf(file, "# Write  %s POLYGONs: %d with Verticies:%d ? and Points: %d\n", offset, numE, numC, numV);
        //      fprintf (file, "%sPOLYGN %d %d %d\n", offset, numE, numC, numV);  //Uwe? Wozu benutzt Ihr offset (=Möglichkeit zum Auskomntieren)
        fprintf(file, "# OK 1 %s   {  \n", offset);
        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#  %s%sATTR %s %s\n", offset, sp, *(name)++, *(val)++);
            fprintf(file, "\n");
        }
        fprintf(file, "Shape{                                 \n");
        fprintf(file, "	appearance Appearance               \n");
        fprintf(file, "	{ material Material {               \n");
        fprintf(file, "#			diffuseColor 0.8 0.1 0.1    \n");
        fprintf(file, "#			ambientIntensity 0.2        \n");
        fprintf(file, "#			emissiveColor 0.0 1.0 0.0   \n");
        fprintf(file, "#			specularColor 0.0 0.0 1.0   \n");
        fprintf(file, "#			shininess 0.2               \n");
        fprintf(file, "#			transparency 0.0            \n");
        fprintf(file, "	}}                                  \n");
        fprintf(file, "	geometry IndexedFaceSet {           \n");
        fprintf(file, "	coord Coordinate {                  \n");
        fprintf(file, "		point [                         \n");
        //	fprintf (file, "#  -1 -1 0,  // First                   \n");
        //	fprintf (file, "#   1 -1 0,                                \n");
        //	fprintf (file, "#   1  1 0,                               \n");
        //	fprintf (file, "#   0  0 1                                 \n");
        //	fprintf (file, "#   ] }                                   \n");

        fprintf(file, "# %s%sVERTEX\n", offset, sp);
        for (i = 0; i < numV - 1; i++) // not se last one, see below
        {
            fprintf(file, " ");
            for (j = 0; j < 3; j++)
            {
                if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12))
                    || (*v[j] == 0.0))
                    fprintf(file, "%f ", *(v[j])++);
                else
                    fprintf(file, "%e ", *(v[j])++);
            }
            fprintf(file, ",\n"); // every get a the end a "," m without the last one
        }

        // now the last one ,  without the last ","
        // fprintf (file, "%s%s%s", offset, sp, sp);
        for (j = 0; j < 3; j++)
        {
            if (((fabs(*v[j]) > 1e-6) && (fabs(*v[j]) < 1e12)) || (*v[j] == 0.0))
                fprintf(file, "%f ", *(v[j])++);
            else
                fprintf(file, "%e ", *(v[j])++);
        }
        fprintf(file, " \n"); // every get a the end a "," m without the last one
        fprintf(file, " ] } \n "); // End of Points

        fprintf(file, "coordIndex [       \n  "); // VRML
        fprintf(file, "#%s%sCONN\n", offset, sp); // old

        pos = *(el);
        el++;
        next = *(el);

        for (i = 0; i < numE; i++)
        {
            //   fprintf (file, "#  %s%s%s\n" , offset, sp, sp);
            for (j = pos; j < next; j++)
                fprintf(file, "%d  ", *(cl)++);
            fprintf(file, "-1, \n");
            pos = next;
            if (i == numE - 2)
                next = numC;
            else
            {
                el++;
                next = *(el);
            }
        }
        fprintf(file, "]                       \n");
        fprintf(file, "# color Color { color [1 0 0, 0 1 0, 0 0 1]} \n");
        fprintf(file, "# colorIndex [1 2 0 2  ] \n");
        fprintf(file, "# colorPerVertex TRUE    \n");
        fprintf(file, "#                        \n");
        fprintf(file, "  creaseAngle 20         \n");
        fprintf(file, "  solid TRUE             \n");
        fprintf(file, "  ccw TRUE               \n");
        fprintf(file, "  convex TRUE            \n");
        fprintf(file, "} }                      \n");
        fprintf(file, "# End Shape IndexFaceSet \n");
        fprintf(file, "# %s}\n", offset);
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
        sl++;
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
                sl++;
                next = *(sl);
            }
        }

        fprintf(file, "%s}\n", offset);
    }
    //  -------------------------------
    else if (strcmp(type, "SETELE") == 0) //    "SETELE" ist geschlossen
    {
        int numElem;
        const coDoSet *obj = (const coDoSet *)new_data;
        const coDistributedObject *const *elem = obj->getAllElements(&numElem);
        const char **name, **val;
        char *space = (char *)"      ";
        int counter = 0;

        fprintf(file, "#2 SETELEM %d\n", numElem);
        fprintf(file, "#2 {\n");

        counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#2 %sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "#2 \n");
        }

        fprintf(file, "#2 %sELEM\n", sp);
        fprintf(file, "#2 %s{\n", sp);

        for (i = 0; i < numElem; i++)
        {
            fprintf(file, "#2 %s# elem number %d\n", space, i);
            writeObj(space, elem[i], file);

            if (i != numElem - 1)
                fprintf(file, "#2 \n");
        }

        fprintf(file, "#2 %s}\n", sp);
        fprintf(file, "#2 }\n");
    }

    else if (strcmp(type, "GEOMET") == 0)
    {
        int has_colors, has_normals, has_texture;
        const coDoGeometry *obj = (const coDoGeometry *)new_data;

        const coDistributedObject *do1 = obj->getGeometry();
        const coDistributedObject *do2 = obj->getColors();
        const coDistributedObject *do3 = obj->getNormals();
        const coDistributedObject *do4 = obj->getTexture();

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

        fprintf(file, "#GEOMET\n");
        fprintf(file, "#{\n");

        const char **name, **val;
        int counter = obj->getAllAttributes(&name, &val);

        if (counter != 0)
        {
            for (i = 0; i < counter; i++)
                fprintf(file, "#%sATTR %s %s\n", sp, *(name)++, *(val)++);

            fprintf(file, "#\n");
        }
        fprintf(file, "#%sHAS_COLORS    %d\n", sp, has_colors);
        fprintf(file, "#%sHAS_NORMALS   %d\n", sp, has_normals);
        fprintf(file, "#%sHAS_TEXTURES  %d\n", sp, has_texture);

        fprintf(file, "#%sELEM\n", sp);
        fprintf(file, "#%s{\n", sp);

        const char *space = "      ";
        writeObj(space, do1, file);
        if (do2)
            writeObj(space, do2, file);
        if (do3)
            writeObj(space, do2, file);
        if (do4)
            writeObj(space, do2, file);

        fprintf(file, "#3 %s}\n", sp);
        fprintf(file, "#3 }\n");
    }
    else if (strcmp(type, "POINTS") == 0)
    {
        const coDoPoints *obj = (const coDoPoints *)new_data;
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
      const coDoSpheres *obj = (const coDoSpheres *) new_data;
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
        const coDoUniformGrid *obj = (const coDoUniformGrid *)new_data;
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
        const coDoIntArr *obj = (const coDoIntArr *)new_data;
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
        const coDoRectilinearGrid *obj = (const coDoRectilinearGrid *)new_data;
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
        const coDoStructuredGrid *obj = (const coDoStructuredGrid *)new_data;
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
        const coDoVec3 *obj = (const coDoVec3 *)new_data;
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
        const coDoFloat *obj = (const coDoFloat *)new_data;
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
        const coDoRGBA *obj = (const coDoRGBA *)new_data;
        int numE = obj->getNumPoints();
        const char **name, **val;
        int counter = 0;

        int *data;

        obj->getAddress(&data);

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
        const coDoUnstructuredGrid *obj = (const coDoUnstructuredGrid *)new_data;
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
                    if (strncmp(type[tl[i]], "10", 2) == 0)
                        //if (type[tl[i]] == (char *)"10")
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

        sendError("ERROR: Object type not supported by COVISE: '%s'",
                  type);
        return;
    }
}

// write writeFileBegin  to file
void WritePolygon::writeFileBegin(int outputtype, FILE *file)

// (const char *offset, coDistributedObject * new_data, FILE * file)
{
    // switch outputtype                    1 = VRML
    if (outputtype == 1)
    {
        fprintf(file, "#VRML V2.0 utf8                   \n");
        fprintf(file, "#  Covise Exporter via WritePolygon, B.Burbaum  \n");
        fprintf(file, "#  WorldInfo {                     \n");
        fprintf(file, "#  info [     ]                    \n");
        fprintf(file, "#  title \"  \" }                  \n");
        fprintf(file, "                                   \n");
        fprintf(file, "   NavigationInfo {                \n");
        fprintf(file, "      avatarSize [0.25, 1.6, 0.75] \n");
        fprintf(file, "      headlight TRUE               \n");
        fprintf(file, "      speed 1.0                    \n");
        fprintf(file, "      type \"WALK\"                \n");
        fprintf(file, "      visibilityLimit 0.0          \n");
        fprintf(file, "      }                            \n");
        fprintf(file, "                                   \n");
    }
    if (outputtype == 2) // 2 = stl
    {
        fprintf(file, "stl                                \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
    }
}

// write writeFileEnd  to file
void WritePolygon::writeFileEnd(int outputtype, FILE *file)

// (const char *offset, coDistributedObject * new_data, FILE * file)
{
    // switch outputtype                    1 = VRML
    if (outputtype == 1)
    {
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, " #   VRML Schluss                  \n");
    }
    if (outputtype == 2) // 2 = stl
    {
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, "                                   \n");
        fprintf(file, " #  stl   Schluss                  \n");
    }
}

// compute callback
int
WritePolygon::compute(const char *)
{

    // ???   const char *DataIn = Covise::get_object_name ("DataIn");

    //2   Nur noch im File schreiben daher kann //2 als auch //1 entfallen
    //2    if (DataIn != NULL)
    //2   {
    //2

    // write data to file
    const char *dataPath = p_filename->getValue();

    //Covise::get_browser_param ("path", &dataPath);

    if (dataPath == NULL)
    {

        // invalid data path
        sendError("ERROR: Could not get filename");
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
        sendError("ERROR: Could not create file");
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

    const coDistributedObject *new_data = coDistributedObject::createFromShm(inputName);
    if (new_data == NULL)
    {

        // object creation failed
        sendError("ERROR: createUnknown() failed for data");

        // close file
        fclose(file);
        return STOP_PIPELINE;
    }

    // write Stuff in the beginn of the file
    //BB   Hier kommt noch eine Funktion
    // write FileBegin
    int outputtyp = 1;
    writeFileBegin(outputtyp, file);

    // write obj

    writeObj("", new_data, file);

    // clean up
    delete new_data;

    // write Stuff at the end of the file
    //BB   Hier kommt noch eine Funktion
    // write FileEnd
    writeFileEnd(outputtyp, file);

    // close file
    fclose(file);
    //2 }
    //2  else
    //2  {
    //2
    //2      // read data from file
    //1      const char *objName = p_dataOut->getObjName ();
    //2      ifstream inputFile (p_filename->getValue ());
    //2
    //2      if (!inputFile.good ())
    //2      {
    //2         char buf[200];
    //2
    //2         strcpy (buf, p_filename->getValue ());
    //2         strcat (buf, ": ");
    //2         strcat (buf, strerror (errno));
    //2         sendError (buf);
    //2         return STOP_PIPELINE;
    //2      }
    //2
    //1     coDistributedObject *obj = readObj (objName, inputFile);
    //1
    //1      if (obj)
    //1      {
    //1         p_dataOut->setCurrentObject (obj);
    //1         return CONTINUE_PIPELINE;
    //1      }
    //1      else
    //1         return STOP_PIPELINE;
    //1

    //2    }

    return STOP_PIPELINE;
}

// destructor
WritePolygon::~WritePolygon()
{
}

MODULE_MAIN(IO, WritePolygon)
