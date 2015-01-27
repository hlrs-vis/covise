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
    int i;

    // Create browser parameter:
    p_filename = addFileBrowserParam("filename", "SIO Lines file (.xyz)");
    p_filename->setValue("data/sio/debi/", "*.xyz");

    // Create output ports:
    p_lines = addOutputPort("lines", "Lines", "Set of lines");
    p_lines->setInfo("Set of lines for all timesteps.");
    p_colors = addOutputPort("colors", "Float", "Set of colors");
    p_colors->setInfo("Set of colors for the lines of all timesteps.");

    // Initialize member variables:
    for (i = 0; i < MAX_TIMESTEPS; ++i)
    {
        lineData[i] = NULL;
        numPoints[i] = 0;
        numLines[i] = 0;
    }
    numTimesteps = 0;
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
        parsed->color = atoi(&buf[10]);
        return true;
    }

    // Check for point coordinates:
    scanned = sscanf(buf, "%f %f %f", &parsed->x, &parsed->y, &parsed->z);
    parsed->x += 116.28f;
    parsed->x *= 100000.0f;
    parsed->y -= 34.5f;
    parsed->y *= 100000.0f;
    parsed->z /= 1.0f;
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
    int color = 12345; // point color
    coParsedLine parsed; // parsed input line information
    int row = 0; // input file row
    int curTimestep = 0; // current timestep
    bool done = false;

    freeMemory();
    cerr << "Reading source file..." << endl;

    // Read line points from file:
    while (parseLine(fp, &parsed) && !done)
    {
        ++row;
        switch (parsed.type)
        {
        case 0: // syntax error
            cerr << "Syntax error in line " << (row + 1) << endl;
            break;
        case 1: // new frame
            if (curTimestep == MAX_TIMESTEPS - 1)
                done = true;
            else if (numPoints[curTimestep] > 0)
            {
                ++numLines[curTimestep];
                ++curTimestep;
            }
            break;
        case 2: // new line
            if (numPoints[curTimestep] > 0)
                ++numLines[curTimestep];
            color = parsed.color;
            break;
        case 3: // point coordinates
            pNew = new coPoint(parsed.x, parsed.y, parsed.z, (float)color / 15.0, numLines[curTimestep]);
            if (pNew == NULL)
            {
                cerr << "Out of memory error." << endl;
                return false;
            }
            if (lineData[curTimestep] == NULL) // first point?
                lineData[curTimestep] = pNew;
            else
                pOld->next = pNew;
            pOld = pNew;
            ++numPoints[curTimestep];
            break;
        default:
            break;
        }
    }

    if (numPoints[curTimestep] > 0)
        ++numLines[curTimestep];
    numTimesteps = curTimestep + 1;
    cerr << numTimesteps << " time steps found." << endl;
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
    int i;
    char *maxTimestep; // last time step index as a string

    // Request Covise data objects for lines:
    linesPort = p_lines->getObjName();
    if (!linesPort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doLines = new coDoLines *[numTimesteps + 1];
    len = (int)(strlen(linesPort) + logf(numTimesteps) / logf(10) + 3);
    objName = new char[len];
    for (i = 0; i < numTimesteps; ++i)
    {
        sprintf(objName, "%s_%d", linesPort, i);
        doLines[i] = new coDoLines(objName, numPoints[i], numPoints[i], numLines[i]);
    }
    doLines[numTimesteps] = NULL;
    delete[] objName;

    // Request Covise data objects for colors:
    colorsPort = p_colors->getObjName();
    if (!colorsPort)
    {
        cerr << "Error: port object name missing!" << endl;
        return;
    }
    doColors = new coDoFloat *[numTimesteps + 1];
    len = (int)(strlen(colorsPort) + logf(numTimesteps) / logf(10) + 3);
    objName = new char[len];
    for (i = 0; i < numTimesteps; ++i)
    {
        sprintf(objName, "%s_%d", colorsPort, i);
        doColors[i] = new coDoFloat(objName, numPoints[i]);
    }
    doColors[numTimesteps] = NULL;
    delete[] objName;

    for (i = 0; i < numTimesteps; ++i)
    {
        // Request Covise data pointers:
        doLines[i]->getAddresses(&x, &y, &z, &vl, &ll);
        doColors[i]->getAddress(&col);

        // Enter world lines into covise data structure:
        lp = lineData[i];
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
            col[point] = lp->color;

            // Advance to next list entry:
            lp = lp->next;
            ++point;
        }
    }

    // Generate set objects:
    doLinesSet = new coDoSet(linesPort, (coDistributedObject **)doLines);
    doColorsSet = new coDoSet(colorsPort, (coDistributedObject **)doColors);
    if (numTimesteps == 1)
        len = 1;
    else
        len = (int)(logf(numTimesteps - 1) / logf(10) + 2);
    maxTimestep = new char[len + 1];
    sprintf(maxTimestep, "%d", numTimesteps - 1);
    doLinesSet->addAttribute("TIMESTEP", "0 ");
    doColorsSet->addAttribute("TIMESTEP", "0 ");

    // Delete sub-objects:
    for (i = 0; i < numTimesteps; ++i)
    {
        delete doLines[i];
        delete doColors[i];
    }
    delete[] doLines;
    delete[] doColors;

    // Zugriffe auf Covise-Objekte aufraeumen:
    p_lines->setCurrentObject(doLinesSet);
    p_colors->setCurrentObject(doColorsSet);
}

/// Gets rid of the allocated memory for the world lines.
void coReadLines::freeMemory()
{
    coPoint *lp, *lpNext;
    int i;

    // Delete world line datasets:
    for (i = 0; i < numTimesteps; ++i)
    {
        lp = lineData[i];
        while (lp != NULL)
        {
            lpNext = lp->next;
            delete lp;
            lp = lpNext;
        }
        lineData[i] = NULL;
        numPoints[i] = 0;
        numLines[i] = 0;
    }
    numTimesteps = 0;
}

/// Main module entry point.
int coReadLines::compute()
{
    FILE *fp;
    const char *filename;

    // Get parameters from covise
    filename = p_filename->getValue();

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
    if (numLines > 0)
        makeOutputData();
    cerr << numLines << " lines read." << endl;

    // Free memory:
    cerr << "Freeing memory" << endl;
    freeMemory();

    cerr << "Done." << endl;
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
        cout << "Point: x=" << current->x << ", y=" << current->y << ", z=" << current->z << ", color=" << current->color << ", line=" << current->line << endl;
        current = current->next;
    }
}
