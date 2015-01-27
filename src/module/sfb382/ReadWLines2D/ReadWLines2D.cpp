/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////
// Module:        ReadWLines2D.cpp
// Author:        Juergen Schulze-Doebold
// Function:      Reads a file containing 2D world lines.
// Usage:         Provide filename of world lines file.
// Affiliation:   SFB 382, projects D2 and A8 (Catia Lavalle)
// Creation Date: 2003-03-16
///////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <api/coModule.h>
#include <virvo/vvtokenizer.h>
#include <virvo/vvarray.h>
#include "ReadWLines2D.h"

/// Startup routine
int main(int argc, char *argv[])
{
    ReadWLines2D *application = new ReadWLines2D();
    application->start(argc, argv);
    return 0;
}

/// Constructor
ReadWLines2D::ReadWLines2D()
{
    // Create module parameters:
    p_filename = addFileBrowserParam("Filename", "2D world lines file (.dat)");
    p_filename->setValue("/mnt/pro/cod/sfb382/", "*.dat");
    p_cylHeight = addFloatParam("OutputHeight", "grid or cylinder height");
    p_cylHeight->setValue(1.0);
    p_cylDiameter = addFloatParam("OutputWidth", "grid witdh or cylinder diameter");
    p_cylDiameter->setValue(1.0);
    p_useCylinder = addBooleanParam("UseCylinder", "output on cylinder or 2D grid");
    p_useCylinder->setValue(true);
    p_normalColor = addFloatParam("RegularColor", "color of regular lines");
    p_normalColor->setValue(0.0);
    p_boundColor = addFloatParam("CrossingColor", "color of lines crossing the boundary");
    p_boundColor->setValue(0.5);
    p_edgeColor = addFloatParam("EdgeColor", "color of data set edges");
    p_edgeColor->setValue(1.0);

    // Create output ports:
    p_lines = addOutputPort("Lines", "coDoLines", "Set of world lines");
    p_colors = addOutputPort("Colors", "coDoFloat", "Set of world line colors");
    p_edgeLines = addOutputPort("EdgeLines", "coDoLines", "Edge lines");
    p_edgeColors = addOutputPort("EdgeColors", "coDoFloat", "Edge colors");

    // Initialize member variables:
    numTimesteps = 0;
    gridSize[0] = gridSize[1] = 0;
    dataGrid = NULL;
    doLines = NULL;
    doColors = NULL;
}

/// Destructor
ReadWLines2D::~ReadWLines2D()
{
}

/** Read world lines file header.
  @return true if ok, false on error.
*/
bool ReadWLines2D::parseHeader(vvTokenizer *tokenizer)
{
    vvTokenizer::TokenType ttype;
    bool done = false;
    int headerValue = 0;

    while (!done)
    {
        // Read parameter qualifier:
        ttype = tokenizer->nextToken();
        if (ttype != vvTokenizer::VV_WORD)
            return false;
        if (strcmp(tokenizer->sval, "xsize") == 0)
            headerValue = 0;
        else if (strcmp(tokenizer->sval, "ntimesteps") == 0)
            headerValue = 1;
        else if (strcmp(tokenizer->sval, "nplaq") == 0)
            headerValue = 2;
        else if (strcmp(tokenizer->sval, "nsweeps") == 0)
            headerValue = 3;

        // Parse parameter:
        ttype = tokenizer->nextToken();
        if (ttype != vvTokenizer::VV_NUMBER)
            return false;
        switch (headerValue)
        {
        case 0:
            gridSize[0] = (int)tokenizer->nval;
            break;
        case 1:
            gridSize[1] = (int)tokenizer->nval;
            break;
        case 3:
            numTimesteps = (int)tokenizer->nval;
            done = true;
            break;
        default:
            break;
        }
    }
    return true;
}

/** Parse world lines file: create line direction array.
  @return true if ok, false on error.
*/
bool ReadWLines2D::parseLines(vvTokenizer *tokenizer)
{
    vvTokenizer::TokenType ttype;
    char msg[128];
    int plaquettes[2]; // number of data fields on grid
    int offset; // offset into dataGrid for time step
    int leftX, rightX; // grid x coordinates of left and right side of plaquette
    int i, x, y;

    plaquettes[0] = gridSize[0] / 2;
    plaquettes[1] = gridSize[1];

    // Read grid information to memory:
    for (i = 0; i < numTimesteps; ++i)
    {
        offset = i * gridSize[0] * gridSize[1];
        for (y = 0; y < plaquettes[1]; ++y)
        {
            for (x = 0; x < plaquettes[0]; ++x)
            {
                // Read next cell description number:
                ttype = tokenizer->nextToken();
                if (ttype != vvTokenizer::VV_NUMBER)
                    return false;

                // Describe line direction in data grid:
                leftX = ((y % 2) == 0) ? (x * 2) : (x * 2 + 1);
                rightX = leftX + 1;
                if (rightX >= gridSize[0])
                    rightX = 0; // periodic boundaries
                switch (int(tokenizer->nval))
                {
                case 0:
                    break; // no line
                case 5: // right to right
                    dataGrid[offset + gridSize[0] * y + rightX] = STAY;
                    break;
                case 6: // left to right
                    dataGrid[offset + gridSize[0] * y + leftX] = RIGHT;
                    break;
                case 9: // right to left
                    dataGrid[offset + gridSize[0] * y + rightX] = LEFT;
                    break;
                case 10: // left to left
                    dataGrid[offset + gridSize[0] * y + leftX] = STAY;
                    break;
                case 15: // left to left and right to right
                    dataGrid[offset + gridSize[0] * y + leftX] = STAY;
                    dataGrid[offset + gridSize[0] * y + rightX] = STAY;
                    break;
                default:
                    sprintf(msg, "Error in world lines file at line %d", tokenizer->getLineNumber());
                    Covise::sendError(msg);
                    return false;
                    break;
                }
            }
        }
    }
    return true;
}

/// Prepare Covise lines.
void ReadWLines2D::prepareLines()
{
    const char *linesPort;
    const char *colorsPort;
    char *linesObjName;
    char *colorsObjName;
    int len; // string length
    int numLines;
    int i;

    // Request Covise data objects for lines and colors:
    linesPort = p_lines->getObjName();
    if (!linesPort)
    {
        cerr << "Error: lines port object name missing!" << endl;
        return;
    }
    colorsPort = p_colors->getObjName();
    if (!colorsPort)
    {
        cerr << "Error: colors port object name missing!" << endl;
        return;
    }

    doLines = new coDoLines *[numTimesteps + 1];
    doColors = new coDoFloat *[numTimesteps + 1];

    len = (int)(strlen(linesPort) + logf(numTimesteps) / logf(10) + 3);
    linesObjName = new char[len];
    len = (int)(strlen(colorsPort) + logf(numTimesteps) / logf(10) + 3);
    colorsObjName = new char[len];

    for (i = 0; i < numTimesteps; ++i)
    {
        numLines = getNumLines(i);
        sprintf(linesObjName, "%s_%d", linesPort, i);
        doLines[i] = new coDoLines(linesObjName, (gridSize[1] + 1) * numLines, (gridSize[1] + 1) * numLines, numLines);
        sprintf(colorsObjName, "%s_%d", colorsPort, i);
        doColors[i] = new coDoFloat(colorsObjName, (gridSize[1] + 1) * numLines);
    }
    doLines[numTimesteps] = NULL;
    doColors[numTimesteps] = NULL;
    delete[] linesObjName;
    delete[] colorsObjName;
}

/// Return the number of lines of a specific time step.
int ReadWLines2D::getNumLines(int timeStep)
{
    int i;
    int offset;
    int numLines = 0;

    offset = timeStep * gridSize[0] * gridSize[1];
    for (i = 0; i < gridSize[0]; ++i)
    {
        if (dataGrid[offset + i] != NO_LINE)
            ++numLines;
    }
    return numLines;
}

/** Convert 2D atom coordinates to flat grid or 3D cylinder coordinates.
  @param x,y             2D coordinates
  @param cylX,cylY,cylZ  3D coordinates on cylinder or flat grid
*/
void ReadWLines2D::convertCoordinates(int x, int y, float *cylX, float *cylY, float *cylZ)
{
    if (p_useCylinder->getValue())
    {
        float radius;
        radius = 0.5f * p_cylDiameter->getValue();
        *cylX = radius * sinf(float(x) / float(gridSize[0]) * 2.0f * M_PI);
        *cylY = float(y) / float(gridSize[1]) * float(p_cylHeight->getValue());
        *cylZ = radius * cosf(float(x) / float(gridSize[0]) * 2.0f * M_PI);
    }
    else
    {
        *cylX = float(x) / float(gridSize[0]) * p_cylDiameter->getValue();
        *cylY = float(y) / float(gridSize[1]) * p_cylHeight->getValue();
        *cylZ = 0.0f;
    }
}

/** Create Covise lines from line direction grid.
  @param t current time step (0=first)
*/
void ReadWLines2D::assembleLines()
{
    int *ll, *vl; // pointers to line list and vertex list
    float *x, *y, *z; // pointers to coordinate lists
    float *col; // pointer to colors list
    int numLines; // number of lines to process
    int numVertices; // number of points on each line
    int i, yPos; // counters for lines and y positions
    int vertex; // index of current line vertex
    float cylX, cylY, cylZ; // coordinates of line points on cylinder
    bool crossesBoundary; // true if the line crosses the boundary
    int prevX[2]; // previous two x coordinates ([0] is previous, [1] is the one before)
    int xPos = 0; // x position of current line point
    int t; // time step
    int startX; // starting x coordinate of current line
    int offset; // offset into dataGrid for current time step

    numVertices = gridSize[1] + 1;
    for (t = 0; t < numTimesteps; ++t)
    {
        // Request Covise data pointers:
        doLines[t]->getAddresses(&x, &y, &z, &vl, &ll);
        doColors[t]->getAddress(&col);

        numLines = getNumLines(t);
        startX = -1;
        offset = t * gridSize[0] * gridSize[1];
        for (i = 0; i < numLines; ++i)
        {
            // Set line list entry:
            ll[i] = i * numVertices;

            // Find the points of this line:
            crossesBoundary = false;
            prevX[0] = prevX[1] = -1;

            // Move startX to the next line starting point:
            do
            {
                ++startX;
            } while (dataGrid[offset + startX] == NO_LINE);

            vertex = i * numVertices;
            for (yPos = 0; yPos < numVertices; ++yPos)
            {
                // Set vertex list entry:
                vl[vertex] = vertex;

                // Compute point coordinates:
                if (yPos == 0)
                {
                    xPos = startX;
                }
                else
                {
                    switch (dataGrid[offset + gridSize[0] * (yPos - 1) + xPos])
                    {
                    case NO_LINE:
                        cerr << "Line " << i << " broken in sweep " << t << ", at time step " << yPos << endl;
                        assert(0);
                        break;
                    case LEFT:
                        --xPos;
                        break;
                    case RIGHT:
                        ++xPos;
                        break;
                    case STAY:
                    default:
                        break;
                    }
                }

                // Periodic boundaries:
                if (xPos < 0)
                {
                    crossesBoundary = true;
                    xPos = gridSize[0] - 1;
                }
                else if (xPos >= gridSize[0])
                {
                    crossesBoundary = true;
                    xPos = 0;
                }

                // Set point coordinates:
                convertCoordinates(xPos, yPos, &cylX, &cylY, &cylZ);
                x[vertex] = cylX;
                y[vertex] = cylY;
                z[vertex] = cylZ;

                // Memorize x coordinates:
                prevX[1] = prevX[0];
                prevX[0] = xPos;

                ++vertex;
            }

            // Set color:
            for (yPos = 0; yPos < numVertices; ++yPos)
            {
                vertex = i * numVertices + yPos;
                if (crossesBoundary)
                    col[vertex] = p_boundColor->getValue();
                else
                    col[vertex] = p_normalColor->getValue();
            }
        }
    }
}

/// Finish Covise lines by creating set objects and cleaning up.
void ReadWLines2D::finishLines()
{
    coDoSet *doLinesSet = NULL;
    coDoSet *doColorsSet = NULL;
    const char *linesPort;
    const char *colorsPort;
    char *maxTimestep; // last time step index as a string
    int len; // string length
    int i;

    // Generate set objects:
    linesPort = p_lines->getObjName();
    colorsPort = p_colors->getObjName();
    doLinesSet = new coDoSet(linesPort, (coDistributedObject **)doLines);
    doColorsSet = new coDoSet(colorsPort, (coDistributedObject **)doColors);
    len = (int)(logf(numTimesteps - 1) / logf(10) + 2);
    maxTimestep = new char[len];
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

    // Assign sets to output ports:
    p_lines->setCurrentObject(doLinesSet);
    p_colors->setCurrentObject(doColorsSet);
}

/// Create circles or lines at world line end points.
void ReadWLines2D::createEdgeLines()
{
    coDoSet *doLinesSet = NULL;
    coDoSet *doColorsSet = NULL;
    const char *linesPort;
    const char *colorsPort;
    char *maxTimestep; // last time step index as a string
    char *linesObjName;
    char *colorsObjName;
    float *x, *y, *z; // pointers to coordinate lists
    float *col; // pointer to colors list
    float cylX, cylY, cylZ; // coordinates of line points on cylinder
    int *ll, *vl; // pointers to line list and vertex list
    int len; // string length
    int i, j, k;
    int vertex;
    int xPos, yPos;
    coDoLines **doEdgeLines;
    coDoFloat **doEdgeColors;

    // Request Covise data objects for lines and colors:
    linesPort = p_edgeLines->getObjName();
    if (!linesPort)
    {
        cerr << "Error: edge lines port object name missing!" << endl;
        return;
    }
    colorsPort = p_edgeColors->getObjName();
    if (!colorsPort)
    {
        cerr << "Error: edge colors port object name missing!" << endl;
        return;
    }

    doEdgeLines = new coDoLines *[numTimesteps + 1];
    doEdgeColors = new coDoFloat *[numTimesteps + 1];

    len = (int)(strlen(linesPort) + logf(numTimesteps) / logf(10) + 3);
    linesObjName = new char[len];
    len = (int)(strlen(colorsPort) + logf(numTimesteps) / logf(10) + 3);
    colorsObjName = new char[len];

    for (i = 0; i < numTimesteps; ++i)
    {
        sprintf(linesObjName, "%s_%d", linesPort, i);
        doEdgeLines[i] = new coDoLines(linesObjName, 2 * (gridSize[0] + 1), 2 * (gridSize[0] + 1), 2);
        sprintf(colorsObjName, "%s_%d", colorsPort, i);
        doEdgeColors[i] = new coDoFloat(colorsObjName, 2 * (gridSize[0] + 1));
    }
    doEdgeLines[numTimesteps] = NULL;
    doEdgeColors[numTimesteps] = NULL;
    delete[] linesObjName;
    delete[] colorsObjName;

    // Create edge lines:
    for (i = 0; i < numTimesteps; ++i)
    {
        // Request Covise data pointers:
        doEdgeLines[i]->getAddresses(&x, &y, &z, &vl, &ll);
        doEdgeColors[i]->getAddress(&col);

        // Create two lines:
        vertex = 0;
        for (j = 0; j < 2; ++j)
        {
            ll[j] = j * (gridSize[0] + 1);

            // Create line segments:
            yPos = (j == 0) ? 0 : gridSize[1];
            for (k = 0; k <= gridSize[0]; ++k)
            {
                vl[vertex] = vertex;
                if (k == gridSize[0])
                {
                    if (p_useCylinder->getValue())
                        xPos = 0;
                    else
                        xPos = k - 1;
                }
                else
                    xPos = k;
                convertCoordinates(xPos, yPos, &cylX, &cylY, &cylZ);
                x[vertex] = cylX;
                y[vertex] = cylY;
                z[vertex] = cylZ;

                // Set color:
                col[vertex] = p_edgeColor->getValue();

                ++vertex;
            }
        }
    }

    // Generate set objects:
    linesPort = p_edgeLines->getObjName();
    colorsPort = p_edgeColors->getObjName();
    doLinesSet = new coDoSet(linesPort, (coDistributedObject **)doEdgeLines);
    doColorsSet = new coDoSet(colorsPort, (coDistributedObject **)doEdgeColors);
    len = (int)(logf(numTimesteps - 1) / logf(10) + 2);
    maxTimestep = new char[len];
    sprintf(maxTimestep, "%d", numTimesteps - 1);
    doLinesSet->addAttribute("TIMESTEP", "0 ");
    doColorsSet->addAttribute("TIMESTEP", "0 ");

    // Delete sub-objects:
    for (i = 0; i < numTimesteps; ++i)
    {
        delete doEdgeLines[i];
        delete doEdgeColors[i];
    }
    delete[] doEdgeLines;
    delete[] doEdgeColors;

    // Assign sets to output ports:
    p_edgeLines->setCurrentObject(doLinesSet);
    p_edgeColors->setCurrentObject(doColorsSet);
}

/// Main module entry point.
int ReadWLines2D::compute()
{
    vvTokenizer *tokenizer; // stream tokenizer to parse data file
    FILE *fp;
    const char *filename;
    char msg[128];
    int i;
    int retVal = CONTINUE_PIPELINE;

    // Get parameters from covise
    filename = p_filename->getValue();

    // Check filename for validity:
    if (strncasecmp(&filename[strlen(filename) - 4], ".dat", 4) != 0)
    {
        Covise::sendError(".dat file expected.");
        return STOP_PIPELINE;
    }

    // Open line data file:
    fp = Covise::fopen(filename, "rb");
    if (!fp)
    {
        Covise::sendError("Could not open world lines file.");
        return STOP_PIPELINE;
    }

    // Create stream tokenizer:
    tokenizer = new vvTokenizer(fp);
    tokenizer->setEOLisSignificant(false);
    tokenizer->setCaseConversion(vvTokenizer::VV_LOWER);
    tokenizer->setParseNumbers(true);
    tokenizer->setWhitespaceCharacter('#');
    tokenizer->setWhitespaceCharacter('=');

    // Parse header:
    if (!parseHeader(tokenizer))
    {
        Covise::sendError("Could not parse file header.");
        delete tokenizer;
        fclose(fp);
        return STOP_PIPELINE;
    }
    else
    {
        sprintf(msg, "World lines grid size: %d x %d, time steps: %d",
                gridSize[0], gridSize[1], numTimesteps);
        Covise::sendInfo(msg);
    }

    // Create data grid in memory:
    dataGrid = new unsigned char[gridSize[0] * gridSize[1] * numTimesteps];
    assert(dataGrid);
    for (i = 0; i < gridSize[0] * gridSize[1] * numTimesteps; ++i)
    {
        dataGrid[i] = NO_LINE;
    }

    // Create lines:
    if (parseLines(tokenizer))
    {
        prepareLines();
        assembleLines();
        finishLines();
    }
    else
        retVal = STOP_PIPELINE;

    delete tokenizer;
    fclose(fp);
    delete[] dataGrid;

    createEdgeLines();

    return retVal;
}

// EOF
