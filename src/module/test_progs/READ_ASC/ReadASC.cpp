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
#include "ReadASC.h"
#include <ctype.h>

void main(int argc, char *argv[])
{
    ReadASC *application = new ReadASC();
    application->start(argc, argv);
}

ReadASC::ReadASC()
{

    // this info appears in the module setup window
    set_module_description("Read Covise ASCII files into Covise");

    // the output port
    p_data = addOutputPort("objects", "coDoPolygons|coDoUnstructuredGrid|coDoVec3|coDoFloat", "Objects");

    // select the file name with a file browser
    p_filename = addFileBrowserParam("inFile", "Input ASCII file");
    char buf[256];
    const char *covisedir = getenv("COVISEDIR");
    if (!covisedir)
        covisedir = ".";
    strcpy(buf, covisedir);
    strcat(buf, "/data/inputfile");
    p_filename->setValue("./TEST", "*");
}

/// get a line of up to <numChars-1> chars:
//     - skip indents
//     - remove end-comments
//     - skip empty lines
//     - remove end-blanks
// returns pointer to first non-blank char in buffer, NULL on error
static char *getLine(istream &str, char *oBuffer, int numChars)
{
    //cerr << " Buffer = " << ( (void*) oBuffer) << endl;

    char *buffer;
    do
    {
        buffer = oBuffer;

        // try to get a line...
        *buffer = '\0';
        if (!str.getline(buffer, numChars - 1))
            return NULL;

        //cerr << "accessing buffer[" << numChars-1 << "]" << endl;
        // make sure it is terminated
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
            while ((*buffer) && isspace(*cPtr))
            {
                *cPtr = '\0';
                cPtr--;
            }
        }
    } while (*buffer == '\0');

    // looks like we got a line...

    //cerr << ">>>>>>>>>>>>>>>>>>>>> '" << buffer << "'" << endl;
    return buffer;
}

static int noOpenBrace(istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        return 1;
    }
    else
        return 0;
}

static int noCloseBrace(istream &str)
{
    char buffer[100000];
    char *command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '}')
    {
        Covise::sendError("Object definition lacks closing brace '}'");
        return 1;
    }
    else
        return 0;
}

// ============== read Attribute and attach to object ==============

static void readAttrib(coDistributedObject *obj, char *buffer)
{
    // skip over 'attrib' how ever it is written
    while (*buffer && !isspace(*buffer))
        buffer++;

    // and all blanks behind
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

    // now ste the attrib
    obj->addAttribute(buffer, value);
}

//// ============== Read a geometry, either LINE or POLY =========================

coDistributedObject *ReadASC::readGeom(const char *name, char *command, istream &str)
{
    // get sizes
    int numPart, numConn, numVert;
    if (sscanf(command, "%d %d %d", &numPart, &numConn, &numVert) != 3)
    {
        Covise::sendError("LINE/POLY command needs 3 integer arguments: numLines,numConn,numVert");
        return NULL;
    }
    if (numPart < 1 || numConn < numPart || numVert < numPart)
    {
        Covise::sendError("LINE/POLY with illegal sizes");
        return NULL;
    }

    // create an object
    coDoLines *lines = NULL;
    coDoPolygons *poly = NULL;
    coDistributedObject *obj = NULL;
    int *partList, *connList;
    float *vx, *vy, *vz;

    if (*command == 'L')
    {
        lines = new coDoLines(name, numVert, numConn, numPart);
        lines->getAddresses(&vx, &vy, &vz, &connList, &partList);
        obj = lines;
    }
    else
    {
        poly = new coDoPolygons(name, numVert, numConn, numPart);
        poly->getAddresses(&vx, &vy, &vz, &connList, &partList);
        obj = poly;
    }

    //cerr << "create Object: numPart=" << numPart
    //     << ", numConn=" << numConn
    //     << ", numVert=" << numVert << endl;

    char buffer[100000];
    char errBuf[600];
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        delete lines;
        delete poly;
        return NULL;
    }

    int readVert = 0;
    int readConn = 0;
    int readPart = 0;

    // read until closing brace has been reached or error occurred
    command = getLine(str, buffer, 100000);
    while (command && *command && *command != '}')
    {
        /// VERT command
        if (strncasecmp("VERT", command, 4) == 0)
        {
            //cerr << "VERT" << endl;
            command = getLine(str, buffer, 100000);
            while (readVert < numVert
                   && command
                   && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f %f %f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f,%f,%f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f;%f;%f", vx, vy, vz) < 3))
                {
                    sprintf(errBuf, "Illegal read in VERT definition: '%s'", buffer);
                    Covise::sendError(errBuf);
                    delete lines;
                    delete poly;
                    return NULL;
                }
                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }

        /// CONN
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            //cerr << "CONN" << endl;
            command = getLine(str, buffer, 100000);
            while (readConn < numConn
                   && readPart < numPart
                   && command
                   && (isdigit(*command) || (*command == '-')))
            {
                // the element starts here:
                *partList = readConn;
                readPart++;
                partList++;

                // split one line securely
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

                    // skip forward till next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }
                command = getLine(str, buffer, 100000);
            }
        }

        /// ATTR
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            //cerr << "ATTR" << endl;
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
            cerr << "Warning ignoring line: '" << buffer << "'" << endl;
            command = getLine(str, buffer, 100000);
        }
    }

    if (readConn == numConn
        && readPart == numPart
        && readVert == numVert)
        return obj;
    else
    {
        delete lines;
        delete poly;
        return NULL;
    }
}

//// ============== Read an USG =========================

coDistributedObject *ReadASC::readUSG(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem, numConn, numVert;
    if (sscanf(command, "%d %d %d", &numElem, &numConn, &numVert) != 3)
    {
        Covise::sendError("USG command needs 3 integer arguments: numLines,numConn,numVert");
        return NULL;
    }
    cout << "read " << numElem << " elements" << endl;
    if (numElem < 1 || numConn < numElem)
    {
        Covise::sendError("USG with illegal sizes");
        return NULL;
    }

    // create an object
    int *elemList, *connList, *typeList;
    float *vx, *vy, *vz;

    coDoUnstructuredGrid *obj = new coDoUnstructuredGrid(name, numElem, numConn, numVert, 1);
    obj->getAddresses(&elemList, &connList, &vx, &vy, &vz);
    obj->getTypeList(&typeList);

    char buffer[100000];
    char errBuf[600];
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        delete obj;
        return NULL;
    }

    int readVert = 0;
    int readConn = 0;
    int readPart = 0;

    // read until closing brace has been reached or error occurred
    command = getLine(str, buffer, 100000);
    while (command && *command && *command != '}')
    {
        /// VERT command
        if (strncasecmp("VERT", command, 4) == 0)
        {
            //cerr << "VERT" << endl;
            command = getLine(str, buffer, 100000);
            while (readVert < numVert
                   && command
                   && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f %f %f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f,%f,%f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f;%f;%f", vx, vy, vz) < 3))
                {
                    sprintf(errBuf, "Illegal read in VERT definition: '%s'", buffer);
                    Covise::sendError(errBuf);
                    delete obj;
                    return NULL;
                }
                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVert++;
            }
        }

        /// CONN
        else if (strncasecmp("CONN", command, 4) == 0)
        {
            //cerr << "CONN" << endl;
            command = getLine(str, buffer, 100000);
            while (readConn < numConn
                   && readPart < numElem
                   && command
                   && isalpha(*command))
            {
                // the element starts here:
                *elemList = readConn;
                readPart++;
                elemList++;

                //cerr << "\ncommand='" << command << "'" << endl;

                // USG: first is type
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
                {
                    cerr << " Illegal element type " << command << endl;
                    *typeList = 0;
                }
                //cerr << *typeList << endl;
                typeList++;

                while (*command && isalpha(*command))
                    command++;

                while (*command && isspace(*command))
                    command++;

                //cerr << "----->>>>  '" << command << "'" << endl;

                // split one line securely
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

                    // skip forward till next digit
                    while (*nextComm && !isdigit(*nextComm))
                        nextComm++;
                }
                command = getLine(str, buffer, 100000);
            }
            cout << "Read CONN: " << readConn << " connections, "
                 << readPart << " elements"
                 << ((readConn == numConn) ? "All conn, " : "Missing conn")
                 << ((readPart == numElem) ? "All elem, " : "Missing elem")
                 << endl;
        }

        /// ATTR
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            //cerr << "ATTR" << endl;
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
            cerr << "Warning ignoring line: '" << buffer << "'" << endl;
            command = getLine(str, buffer, 100000);
        }
    }

    if (readConn == numConn
        && readPart == numElem
        && readVert == numVert)
        return obj;
    else
    {
        delete obj;
        return NULL;
    }
}

//// ============== Read an Unstructured_S3D =========================

coDistributedObject *ReadASC::readV3D(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem;
    if (sscanf(command, "%d", &numElem) != 1)
    {
        Covise::sendError("VECT-USG command needs 1 integer argument: numElem");
        return NULL;
    }
    if (numElem < 1)
    {
        Covise::sendError("SCAL-USG with illegal sizes");
        return NULL;
    }

    // create an object
    float *vx, *vy, *vz;

    coDoVec3 *obj
        = new coDoVec3(name, numElem);
    obj->getAddresses(&vx, &vy, &vz);

    char buffer[100000];
    char errBuf[600];
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        delete obj;
        return NULL;
    }

    int readVal = 0;

    // read until closing brace has been reached or error occurred
    command = getLine(str, buffer, 100000);
    while (*command && *command != '}')
    {
        /// VERT command
        if (strncasecmp("DATA", command, 4) == 0)
        {
            //cerr << "VERT" << endl;
            command = getLine(str, buffer, 100000);
            while (readVal < numElem
                   && command
                   && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if ((sscanf(command, "%f %f %f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f,%f,%f", vx, vy, vz) < 3)
                    && (sscanf(command, "%f;%f;%f", vx, vy, vz) < 3))
                {
                    sprintf(errBuf, "Illegal read in DATA definition: '%s'", buffer);
                    Covise::sendError(errBuf);
                    delete obj;
                    return NULL;
                }
                vx++;
                vy++;
                vz++;
                command = getLine(str, buffer, 100000);
                readVal++;
            }
        }

        /// ATTR
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            //cerr << "ATTR" << endl;
            readAttrib(obj, command);
            command = getLine(str, buffer, 100000);
        }
        else
        {
            cerr << "Warning ignoring line: '" << buffer << "'" << endl;
            command = getLine(str, buffer, 100000);
        }
    }

    if (readVal == numElem)
        return obj;
    else
    {
        delete obj;
        return NULL;
    }
}

//// ============== Read an Unstructured_S3D =========================

coDistributedObject *ReadASC::readS3D(const char *name, char *command, istream &str)
{

    // get sizes
    int numElem;
    if (sscanf(command, "%d", &numElem) != 1)
    {
        Covise::sendError("VECT-USG command needs 1 integer argument: numElem");
        return NULL;
    }
    if (numElem < 1)
    {
        Covise::sendError("SCAL-USG with illegal sizes");
        return NULL;
    }

    // create an object
    float *vx;

    coDoFloat *obj
        = new coDoFloat(name, numElem);
    obj->getAddress(&vx);

    char buffer[100000];
    //cerr << "Alloc buffer[100000] ptr = " << ((void*)buffer) << endl;
    char errBuf[600];
    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        delete obj;
        return NULL;
    }

    int readVal = 0;

    // read until closing brace has been reached or error occurred
    command = getLine(str, buffer, 100000);
    while (*command && *command != '}')
    {
        /// DATA command
        if (strncasecmp("DATA", command, 4) == 0)
        {
            //cerr << "VERT" << endl;
            command = getLine(str, buffer, 100000);
            while (readVal < numElem
                   && command
                   && (isdigit(*command) || (*command == '-') || (*command == '.')))
            {
                if (sscanf(command, "%f", vx) < 1)
                {
                    sprintf(errBuf, "Illegal read in DATA definition: '%s'", buffer);
                    Covise::sendError(errBuf);
                    delete obj;
                    return NULL;
                }
                vx++;
                //cerr << "getLine: buffer=" << ((void*)buffer) << endl;
                command = getLine(str, buffer, 100000);
                readVal++;
            }
        }

        /// ATTR
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            //cerr << "ATTR" << endl;
            readAttrib(obj, command);
            //cerr << "getLine: buffer=" << ((void*)buffer) << endl;
            command = getLine(str, buffer, 100000);
        }
        else
        {
            cerr << "Warning ignoring line: '" << buffer << "'" << endl;
            command = getLine(str, buffer, 100000);
        }
    }

    if (readVal == numElem)
        return obj;
    else
    {
        delete obj;
        return NULL;
    }
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
///
///    Read a Set

coDistributedObject *ReadASC::readSet(const char *name, char *command, istream &str)
{
    char buffer[100000];

    // get sizes
    int numElem;
    if (sscanf(command, "%d", &numElem) != 1)
    {
        Covise::sendError("SET command needs 1 integer argument: numElem");
        return NULL;
    }
    if (numElem < 1)
    {
        Covise::sendError("SET with illegal sizes");
        return NULL;
    }

    coDistributedObject **objs = new coDistributedObject *[numElem + 1];
    objs[numElem] = NULL;
    int i;

    char *attribs[100000]; // max. number of attribs
    int numAttr = 0;

    command = getLine(str, buffer, 100000);

    // opening brace
    if (!command || *command != '{')
    {
        Covise::sendError("Object definition lacks opening brace '{'");
        delete[] objs;
        return NULL;
    }

    // read until closing brace has been reached or error occurred
    command = getLine(str, buffer, 100000);
    while (*command && *command != '}')
    {
        /// Element list command
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
            }
            if (noCloseBrace(str))
                return NULL;
            command = getLine(str, buffer, 100000);
        }

        /// ATTR
        else if (strncasecmp("ATTR", command, 4) == 0)
        {
            attribs[numAttr] = strcpy(new char[strlen(command) + 1], command);
            numAttr++;
            command = getLine(str, buffer, 100000);
        }
        else
        {
            cerr << "Warning ignoring line: '" << buffer << "'" << endl;
            command = getLine(str, buffer, 100000);
        }
    }

    coDoSet *set = new coDoSet(name, objs);
    delete[] objs;
    for (i = 0; i < numAttr; i++)
    {
        readAttrib(set, attribs[i]);
        delete[] attribs[i];
    }

    return set;
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
//
// (recursively) read an object from the file: return NULL on error
//
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

coDistributedObject *ReadASC::readObj(const char *name, istream &str)
{
    char buffer[100000];

    char *command = getLine(str, buffer, 100000);

    // find command parameters if available
    char *param = command;
    while (param && *param && !isspace(*param))
        param++;
    while (param && *param && isspace(*param))
        param++;

    /// ========= LINE and POLY Objects

    if (strncasecmp("LINE", command, 4) == 0
        || strncasecmp("POLY", command, 4) == 0)
        return readGeom(name, param, str);

    else if (strncasecmp("USG", command, 3) == 0)
        return readUSG(name, param, str);

    else if (strncasecmp("SCAL-USG", command, 8) == 0)
        return readS3D(name, param, str);

    else if (strncasecmp("VECT-USG", command, 8) == 0)
        return readV3D(name, param, str);

    else if (strncasecmp("SET", command, 3) == 0)
        return readSet(name, param, str);

    else if (strncasecmp("REP", command, 3) == 0) // repeat lasty object for sets
        return NULL;

    return NULL;
}

int ReadASC::compute()
{
    const char *objName = p_data->getObjName();
    ifstream inputFile(p_filename->getValue());

    coDistributedObject *obj = readObj(objName, inputFile);

    if (obj)
    {
        p_data->setCurrentObject(obj);
        return CONTINUE_PIPELINE;
    }
    else
        return STOP_PIPELINE;
}

ReadASC::~ReadASC()
{
}
