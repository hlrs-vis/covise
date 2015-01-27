/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////
// Module:        ReadSIOPoints.cpp
// Author:        Jurgen Schulze
// Function:      Reads a file containing line coordinates from Scripps Inst.
// Usage:         Provide filename of lines file.
// History:       2005-10-24  creation date
///////////////////////////////////////////////////////////////

// C++:
#include <stdlib.h>
#include <stdio.h>

// Covise:
#include <util/coviseCompat.h>
#include <api/coModule.h>

// Local:
#include "ReadSIOPoints.h"

/// Constructor
coReadPoints::coReadPoints()
    : coModule("SIO Points Reader")
{
    // Create browser parameter:
    _pbrFilename = addFileBrowserParam("filename", "SIO Points file (.dat)");
    _pbrFilename->setValue("data/sio/atul/", "*.dat");
    _pfsXOffset = addFloatParam("xoffset", "Offset for X coordinate");
    _pfsXOffset->setValue(116.0f);
    _pfsYOffset = addFloatParam("yoffset", "Offset for Y coordinate");
    _pfsYOffset->setValue(-33.0f);
    _pfsZFactor = addFloatParam("zfactor", "Factor to multiply Z coordinate with");
    _pfsZFactor->setValue(0.0001f);
    _pfsSizeFactor = addFloatParam("sizefactor", "Factor to multiply size with");
    _pfsSizeFactor->setValue(1.0f);

    // Create output ports:
    _poPoints = addOutputPort("points", "Points", "Points");
    _poSizes = addOutputPort("sizes", "Float", "Point sizes");

    _fileType = UNKNOWN;
}

/// Destructor
coReadPoints::~coReadPoints()
{
    freeMemory();
}

/// @return true if line has been parsed alright
bool coReadPoints::parseLine(FILE *fp, coPoint *point)
{
    const int MAX_LINE_LENGTH = 256;
    char buf[MAX_LINE_LENGTH]; // line buffer

    if (feof(fp))
        return false;

    // Read the next ASCII line:
    memset(buf, 0, MAX_LINE_LENGTH);
    if (fgets(buf, MAX_LINE_LENGTH, fp) == NULL)
    {
        return false;
    }

    switch (_fileType)
    {
    case TOWNS:
    {
        char nameBuf[MAX_LINE_LENGTH];
        int itemsScanned = sscanf(buf, "%f %f %s", &point->_xyz[1], &point->_xyz[0], nameBuf);
        if (itemsScanned != 3 || strlen(nameBuf) == 0)
        {
            cerr << "Syntax error in towns file, line:" << endl << buf << endl;
            return false;
        }
        nameBuf[strlen(nameBuf) - 1] = '\0';
        point->setName(nameBuf + 1);
        point->_xyz[2] = 0.0f; // assume all towns are at sea level
        point->_size = 1.0f;

        point->print();
        break;
    }
    case STATIONS:
    {
        char nameBuf[MAX_LINE_LENGTH];
        char xBuf[32];
        char yBuf[32];
        char zBuf[32];
        memset(nameBuf, 0, MAX_LINE_LENGTH);
        memset(xBuf, 0, 32);
        memset(yBuf, 0, 32);
        memset(zBuf, 0, 32);

        // Now copy chunks of characters from line to buffers; the string sizes are empirical and may need to be adapted to new files:
        memcpy(nameBuf, buf, 5);
        memcpy(xBuf, buf + 52, 10);
        memcpy(yBuf, buf + 43, 8);
        memcpy(zBuf, buf + 63, 5);

        point->_xyz[0] = atof(xBuf);
        point->_xyz[1] = atof(yBuf);
        point->_xyz[2] = atof(zBuf); // meters above sea level
        point->setName(nameBuf);
        point->_size = 1.0f;

        point->print();
        break;
    }
    case EARTHQUAKES:
    {
        float seis[3];
        int itemsScanned = sscanf(buf, "%f %f %f %f %f %f", &point->_xyz[1], &point->_xyz[0], &point->_xyz[2], &seis[0], &seis[1], &seis[2]);
        if (itemsScanned != 6)
        {
            cerr << "Syntax error in earthquakes file, line:" << endl << buf << endl;
            return false;
        }
        point->_xyz[2] *= 1000.0f;
        point->_size = MAX(seis[0], MAX(seis[1], seis[2]));
        if (point->_size == -999.0f)
        {
            point->_size = 1.0f;
        }
        break;

        /*
      char nameBuf[MAX_LINE_LENGTH];
      char xBuf[32];
      char yBuf[32];
      char zBuf[32];
      char size[3][16];
      memset(nameBuf, 0, MAX_LINE_LENGTH);
      memset(xBuf, 0, 32);
      memset(yBuf, 0, 32);
      memset(zBuf, 0, 32);
      memset(size[0], 0, 16);
      memset(size[1], 0, 16);
      memset(size[2], 0, 16);
      
      // Now copy chunks of characters from line to buffers; the string sizes are empirical and may need to be adapted to new files:
      memcpy(nameBuf, buf+97, 16);
      memcpy(xBuf, buf + 10, 9);
      memcpy(yBuf, buf + 1, 8);
      memcpy(zBuf, buf + 20, 9);
      memcpy(size[0], buf + 73, 7);
      memcpy(size[1], buf + 81, 7);
      memcpy(size[2], buf + 89, 7);
      
      point->_xyz[0] = atof(xBuf);
      point->_xyz[1] = atof(yBuf);
      point->_xyz[2] = -1000.0f * atof(zBuf);   // convert value to meters above sea level
      point->setName(nameBuf);
      point->_size = MAX(atof(size[0]), MAX(atof(size[1]), atof(size[2])));
      if (point->_size==-999.0f)
      {
        point->_size = 0.0f;
        cerr << "Warning: earthquake " << point->_name << " has no magnitude" << endl;
      }
      break;
*/
    }
    case UNKNOWN:
    default:
        assert(0);
        break;
    }

    return true;
}

/** Read data file.
   @param fp file pointer
  @return true if ok, false if reading was aborted.
*/
bool coReadPoints::readFile(FILE *fp)
{
    coPoint *tmpPoint = new coPoint();
    coPoint *newPoint;

    freeMemory();
    cerr << "Reading source file..." << endl;

    // Read line points from file:
    while (parseLine(fp, tmpPoint))
    {
        tmpPoint->_xyz[0] += _pfsXOffset->getValue();
        tmpPoint->_xyz[1] += _pfsYOffset->getValue();
        tmpPoint->_xyz[2] *= _pfsZFactor->getValue();
        tmpPoint->_size *= _pfsSizeFactor->getValue();
        newPoint = new coPoint(tmpPoint);
        _pointList.push_back(newPoint);
    }

    delete tmpPoint;
    return true;
}

/// Actually draw the lines of one time step:
void coReadPoints::makeOutputData()
{
    coDoPoints **doPoints = NULL;
    coDoFloat **doSizes = NULL;
    coDoSet *doPointSet = NULL;
    coDoSet *doSizeSet = NULL;
    float *x, *y, *z, *col;
    const char *pointPort;
    const char *sizePort;
    char *objName;
    int len; // string length
    int i;

    // Create Covise points object:
    pointPort = _poPoints->getObjName();
    if (!pointPort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doPoints = new coDoPoints *[2];
    len = (int)(strlen(pointPort) + 2 + 1);
    objName = new char[len];
    sprintf(objName, "%s_0", pointPort);
    doPoints[0] = new coDoPoints(objName, _pointList.size());
    doPoints[1] = NULL;
    delete[] objName;

    // Create Covise size object:
    sizePort = _poSizes->getObjName();
    if (!sizePort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doSizes = new coDoFloat *[2];
    len = int(strlen(sizePort) + 2 + 1);
    objName = new char[len];
    sprintf(objName, "%s_0", sizePort);
    doSizes[0] = new coDoFloat(objName, _pointList.size());
    doSizes[1] = NULL;
    delete[] objName;

    // Request Covise data pointers:
    doPoints[0]->getAddresses(&x, &y, &z);
    doSizes[0]->getAddress(&col);

    // Copy points to Covise data structure:
    list<coPoint *>::iterator iter;
    for (i = 0, iter = _pointList.begin(); iter != _pointList.end(); iter++, ++i)
    {
        x[i] = (*iter)->_xyz[0];
        y[i] = (*iter)->_xyz[1];
        z[i] = (*iter)->_xyz[2];
        col[i] = (*iter)->_size;
    }

    // Generate set objects:
    doPointSet = new coDoSet(pointPort, (coDistributedObject **)doPoints);
    doSizeSet = new coDoSet(sizePort, (coDistributedObject **)doSizes);

    // Delete local objects:
    delete doPoints[0];
    delete doSizes[1];
    delete[] doPoints;
    delete[] doSizes;

    // Zugriffe auf Covise-Objekte aufraeumen:
    _poPoints->setCurrentObject(doPointSet);
    _poSizes->setCurrentObject(doSizeSet);
}

/// Gets rid of the allocated memory for the world lines.
void coReadPoints::freeMemory()
{
    list<coPoint *>::iterator iter;
    for (iter = _pointList.begin(); iter != _pointList.end(); iter++)
    {
        delete[](*iter) -> _name;
        delete *iter;
    }
    _pointList.clear();
}

/// Main module entry point.
int coReadPoints::compute()
{
    FILE *fp;
    const char *filename;

    // Get parameters from covise
    filename = _pbrFilename->getValue();

    // Determine file type:
    _fileType = guessFileType(filename);
    cerr << "File type is ";
    switch (_fileType)
    {
    case TOWNS:
        cerr << "TOWNS" << endl;
        break;
    case STATIONS:
        cerr << "STATIONS" << endl;
        break;
    case EARTHQUAKES:
        cerr << "EARTHQUAKES" << endl;
        break;
    case UNKNOWN:
        cerr << "UNKNOWN" << endl;
        break;
    default:
        break;
    }

    // Open line data file:
    fp = Covise::fopen(filename, "r");
    if (!fp)
    {
        Covise::sendError("Could not open data file");
        return 0;
    }

    // Read line data of selected time step from file:
    readFile(fp);
    fclose(fp);

    // Generate data objects:
    cerr << _pointList.size() << " points read. Converting...";
    if (_pointList.size() > 0)
        makeOutputData();
    cerr << "done" << endl;

    // Free memory:
    cerr << "Freeing memory...";
    freeMemory();
    cerr << "done" << endl;

    return 0;
}

/// Called before module terminates.
void coReadPoints::quit()
{
}

coReadPoints::FileType coReadPoints::guessFileType(const char *fileName)
{
    if (strstr(fileName, "towns") != NULL)
        return TOWNS;
    else if (strstr(fileName, "stations") != NULL)
        return STATIONS;
    else
        return EARTHQUAKES;
}

/// Startup routine
int main(int argc, char *argv[])
{
    coReadPoints *application = new coReadPoints();
    application->start(argc, argv);
    return 0;
}
