/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////
// Module:        ReadSIOLines.cpp
// Author:        Jurgen Schulze
// Function:      Reads a file containing line coordinates from Scripps Inst.
// Usage:         Provide filename of lines file.
// History:       2005-10-23  creation date
///////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <api/coModule.h>
#include "ReadSIOLines.h"

/// Constructor
coReadLines::coReadLines()
    : coModule("SIO Lines Reader")
{
    // Create browser parameters:
    _pbrFilename = addFileBrowserParam("filename", "SIO Lines file (.xyz)");
    _pbrFilename->setValue("data/sio/atul/", "*.xyz");
    _pfsXOffset = addFloatParam("xoffset", "Offset for X coordinate");
    _pfsXOffset->setValue(116.0f);
    _pfsYOffset = addFloatParam("yoffset", "Offset for Y coordinate");
    _pfsYOffset->setValue(-33.0f);
    _pfsZFactor = addFloatParam("zfactor", "Factor to multiply Z coordinate with");
    _pfsZFactor->setValue(0.0001f);

    // Create output ports:
    _poLines = addOutputPort("lines", "Lines", "Set of lines");
    _poLines->setInfo("Set of lines for all timesteps.");
    _poColors = addOutputPort("colors", "Float", "Set of colors");
    _poColors->setInfo("Set of colors for the lines of all timesteps.");

    // Initialize member variables:
    _lineData = NULL;
    _numPoints = 0;
    _numLines = 0;
}

/// Destructor
coReadLines::~coReadLines()
{
    freeMemory();
}

/// @return true if something was parsed, false on EOF
bool coReadLines::parseLine(FILE *fp, coParsedLine *parsed)
{
    const int MAX_LINE_LENGTH = 128;
    char buf[MAX_LINE_LENGTH]; // line buffer
    int scanned;

    // Read the next ASCII line:
    if (fgets(buf, MAX_LINE_LENGTH, fp) == NULL)
        return false;

    // Check for new frame command:
    if (strncasecmp(buf, "[NEWFRAME]", 10) == 0)
    {
        parsed->type = 1;
        return true;
    }

    // Check for new line command:
    if (strncasecmp(buf, "9999 9999 9999", 14) == 0)
    {
        parsed->type = 2;
        return true;
    }

    // Check for point coordinates:
    scanned = sscanf(buf, "%f %f %f", &parsed->x, &parsed->y, &parsed->z);

    parsed->x += _pfsXOffset->getValue();
    parsed->y += _pfsYOffset->getValue();
    parsed->z *= _pfsZFactor->getValue();

    if (scanned == 3)
        parsed->type = 3;
    else
        parsed->type = 0;
    return true;
}

/** Read data file.
   @param fp file pointer
  @return true if ok, false if reading was aborted.
*/
bool coReadLines::readFile(FILE *fp)
{
    coPoint *pNew, *pOld = NULL; // new and old points
    coParsedLine parsed; // parsed input line information
    int row = 0; // input file row

    freeMemory();
    cerr << "Reading source file..." << endl;

    // Read line points from file:
    while (parseLine(fp, &parsed))
    {
        ++row;
        switch (parsed.type)
        {
        case 0: // syntax error
            cerr << "Syntax error in line " << (row + 1) << endl;
            break;
        case 1: // new frame
        // fall through
        case 2: // new line
            if (_numPoints > 0)
                ++_numLines;
            break;
        case 3: // point coordinates
            pNew = new coPoint(parsed.x, parsed.y, parsed.z, _numLines);
            if (pNew == NULL)
            {
                cerr << "Out of memory error." << endl;
                return false;
            }
            if (_lineData == NULL)
                _lineData = pNew; // first point?
            else
                pOld->next = pNew;
            pOld = pNew;
            ++_numPoints;
            break;
        default:
            break;
        }
    }

    if (_numPoints > 0)
        ++_numLines;
    return true;
}

/// Actually draw the lines of one time step:
void coReadLines::makeOutputData()
{
    coDoLines **doLines = NULL;
    coDoFloat **doColors = NULL;
    coDoSet *doLinesSet = NULL;
    coDoSet *doColorsSet = NULL;
    int *ll, *vl;
    float *x, *y, *z, *col;
    coPoint *lp;
    int point, line; // currently processed point/line index
    int prevLine; // previous line
    const char *linesPort;
    const char *colorsPort;
    char *objName;
    int len; // string length

    // Request Covise data objects for lines:
    linesPort = _poLines->getObjName();
    if (!linesPort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doLines = new coDoLines *[1];
    len = int(strlen(linesPort) + 2 + 1);
    objName = new char[len];
    sprintf(objName, "%s_0", linesPort);
    doLines[0] = new coDoLines(objName, _numPoints, _numPoints, _numLines);
    doLines[1] = NULL;
    delete[] objName;

    // Request Covise data objects for colors:
    colorsPort = _poColors->getObjName();
    if (!colorsPort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doColors = new coDoFloat *[1];
    len = (int)(strlen(colorsPort) + 2 + 1);
    objName = new char[len];
    sprintf(objName, "%s_0", colorsPort);
    doColors[0] = new coDoFloat(objName, _numPoints);
    doColors[1] = NULL;
    delete[] objName;

    // Request Covise data pointers:
    doLines[0]->getAddresses(&x, &y, &z, &vl, &ll);
    doColors[0]->getAddress(&col);

    // Enter world lines into covise data structure:
    lp = _lineData;
    point = 0;
    line = 0;
    prevLine = -1;

    while (lp != NULL) // walk through all line points
    {
        // Set line list entry:
        if (prevLine != lp->line)
        {
            ll[line] = point;
            prevLine = lp->line;
            ++line;
        }

        // Set vertex list entry:
        vl[point] = point;

        // Set point coordinates:
        x[point] = lp->x;
        y[point] = lp->y;
        z[point] = lp->z;

        // Set color coordinates:
        col[point] = _color;

        // Advance to next list entry:
        lp = lp->next;
        ++point;
    }

    // Generate set objects:
    doLinesSet = new coDoSet(linesPort, (coDistributedObject **)doLines);
    doColorsSet = new coDoSet(colorsPort, (coDistributedObject **)doColors);

    // Delete sub-objects:
    delete doLines[0];
    delete doColors[0];
    delete[] doLines;
    delete[] doColors;

    // Zugriffe auf Covise-Objekte aufraeumen:
    _poLines->setCurrentObject(doLinesSet);
    _poColors->setCurrentObject(doColorsSet);
}

/// Gets rid of the allocated memory for the world lines.
void coReadLines::freeMemory()
{
    coPoint *lp, *lpNext;

    // Delete world line datasets:
    lp = _lineData;
    while (lp != NULL)
    {
        lpNext = lp->next;
        delete lp;
        lp = lpNext;
    }
    _lineData = NULL;
    _numPoints = 0;
    _numLines = 0;
}

/// Main module entry point.
int coReadLines::compute()
{
    FILE *fp;
    const char *filename;

    // Get parameters from covise
    filename = _pbrFilename->getValue();

    // Check filename for validity:
    if (strncasecmp(&filename[strlen(filename) - 4], ".xyz", 4) != 0)
    {
        Covise::sendError("Line data file expected (.xyz)");
        return 0;
    }

    // Open line data file:
    fp = Covise::fopen(filename, "r");
    if (!fp)
    {
        Covise::sendError("Could not open line data file");
        return 0;
    }

    // Read line data of selected time step from file:
    readFile(fp);
    fclose(fp);

    // Generate data objects:
    cerr.setf(ios::dec, ios::basefield);
    if (_numLines > 0)
        makeOutputData();
    cerr << _numLines << " lines read" << endl;

    // Free memory:
    cerr << "Freeing memory...";
    freeMemory();
    cerr << "done" << endl;

    return 0;
}

/// Called before module terminates.
void coReadLines::quit()
{
}

/// Startup routine
int main(int argc, char *argv[])
{
    coReadLines *application = new coReadLines();
    application->start(argc, argv);
    return 0;
}

/// Prints point data for all points starting with current.
void coPoint::print()
{
    coPoint *current;

    current = this;
    while (current != NULL)
    {
        cout << "Point: x=" << current->x << ", y=" << current->y << ", z=" << current->z << ", line=" << current->line << endl;
        current = current->next;
    }
}
