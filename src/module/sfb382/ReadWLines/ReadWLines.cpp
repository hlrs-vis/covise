/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////
// Module:  			ReadWLines.cpp
// Author:  			Juergen Schulze-Doebold
// Function: 			Reads a world lines file and displays world lines
// Relationship: 	SFB 382, Project A8
// Usage:					Provide filename and numbers of lines to be highlighted
// History: 			1999-05-21  creation date
//                2000-02-23  axis description added, y axis shortened
///////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include "ReadWLines.h"

// Values which produce warnings when overflow:
#define MAX_X 1000.0
#define MAX_Y 1000.0
#define MAX_Z 1000.0
#define MAX_COL 2.0

int main(int argc, char *argv[])
{
    ReadWLines *application = new ReadWLines(argc, argv);
    application->run();

    return 0;
}

ReadWLines::ReadWLines(int argc, char *argv[])
{
    // this info appears in the module setup window
    Covise::set_module_description("WorldLines Reader");

    // the output ports
    Covise::add_port(OUTPUT_PORT, "lines", "coDoLines", "World Lines");
    Covise::add_port(OUTPUT_PORT, "lineColors", "coDoFloat", "Line Colors");
    Covise::add_port(OUTPUT_PORT, "polygons", "coDoPolygons", "chessboard");

    // select the OBJ file name with a file browser
    Covise::add_port(PARIN, "wlPath", "Browser", "WorldLine file");
    Covise::set_port_default("wlPath", "/mnt/cod/worldlines/ *.wl");

    // the selected lines should be highlighted
    Covise::add_port(PARIN, "minWL", "Scalar", "index of first line to draw");
    Covise::set_port_default("minWL", "-1");

    Covise::add_port(PARIN, "maxWL", "Scalar", "index of last line to draw");
    Covise::set_port_default("maxWL", "-1");

    Covise::add_port(PARIN, "line1", "Scalar", "index of line #1 to highlight (-1 = none)");
    Covise::set_port_default("line1", "-1");

    Covise::add_port(PARIN, "line2", "Scalar", "index of line #2 to highlight (-1 = none)");
    Covise::set_port_default("line2", "-1");

    Covise::add_port(PARIN, "line3", "Scalar", "index of line #3 to highlight (-1 = none)");
    Covise::set_port_default("line3", "-1");

    Covise::add_port(PARIN, "scaleY", "Scalar", "scale value in Y direction");
    Covise::set_port_default("scaleY", "-1");

    Covise::add_port(PARIN, "scaleZ", "Scalar", "scale value in Z direction");
    Covise::set_port_default("scaleZ", "-1");

    Covise::add_port(PARIN, "grid", "Boolean", "draw grid");
    Covise::set_port_default("grid", "TRUE");

    // set up the connection to the controller and data manager
    Covise::init(argc, argv);

    // set the quit and the compute callback
    Covise::set_quit_callback(ReadWLines::quitCallback, this);
    Covise::set_start_callback(ReadWLines::computeCallback, this);
}

void ReadWLines::quitCallback(void *userData, void *callbackData)
{
    ReadWLines *thisApp = (ReadWLines *)userData;
    thisApp->quit(callbackData);
}

void ReadWLines::computeCallback(void *userData, void *callbackData)
{
    ReadWLines *thisApp = (ReadWLines *)userData;
    thisApp->compute(callbackData);
}

void ReadWLines::quit(void *)
{
    // dummy
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// returns 1 if ok, -1 if error
// read word can be found in buf
int ReadWLines::getWord(char *buf, FILE *fp)
{
    int i;
    char c;

    i = 0;
    do
    {
        do
        {
            c = fgetc(fp);
        } while (i == 0 && !feof(fp) && (c == ' ' || c == '\n' || c == '\r'));
        buf[i] = c;
        ++i;
    } while (c != ' ' && c != '\n' && c != '\r' && !feof(fp));
    buf[i] = '\0';
    if (feof(fp))
        return -1;
    else
        return 1;
}

// returns 1 if ok, -1 if error
int ReadWLines::getNextDataset(int *setnr, int *x, int *y, int *z, FILE *fp)
{
    char buffer[80];
    int i;

    for (i = 0; i < 3; ++i)
    {
        if (getWord(buffer, fp) == -1)
        {
            cerr << "EOF found." << endl;
            return -1;
        }
        if (i == 0 && buffer[0] == '#') // allow # as first character
        {
            i = -1;
            *setnr = *setnr + 1;
            continue;
        }
        if (buffer[0] < '0' || buffer[0] > '9') // check for vaild number entry
        {
            cerr << "World lines read error: unexpected entry in dataset " << *setnr << endl;
            return -1;
        }
        switch (i)
        {
        case 0:
            *x = atoi(buffer);
            break;
        case 1:
            *y = atoi(buffer);
            break;
        case 2:
            *z = atoi(buffer);
            break;
        default:
            break;
        }
    }
    //	cerr << "Particle " << *x << ", " << *y << ", " << *z << endl;
    return 1;
}

// returns number of particles read
void ReadWLines::readWLFile(FILE *fp, int minWL, int maxWL, int l1, int l2, int l3)
{
    WLinePoint *wlpNew, *wlpOld; // new and old world line points
    int x, y, z; // current world line coordinates
    int setOld = -1, setNew = -1; // line set index
    float color; // line color

    cerr << "Reading source file..." << endl;

    // Preprocess input variables:
    if (minWL > maxWL)
        minWL = maxWL = -1;
    if (minWL < 0)
        minWL = 0;
    if (maxWL < 0)
        maxWL = 0x7FFFFFFF;

    // Initialize class variables:
    numPoints = 0;
    numLines = 0;

    // Read line points from file:
    wlpOld = NULL;
    while (getNextDataset(&setNew, &x, &y, &z, fp) == 1)
    {
        if (setNew < 0)
            setNew = 0; // only happens if file does not start with '#'

        if (&x == NULL || &y == NULL || &z == NULL)
        {
            cerr << "Read error at point #" << numPoints + 1 << endl;
            continue;
        }

        if (setNew == l1)
            color = LINECOLOR1;
        else if (setNew == l2)
            color = LINECOLOR2;
        else if (setNew == l3)
            color = LINECOLOR3;
        else
            color = LINECOLOR;

        if (setNew >= minWL && setNew <= maxWL)
        {
            wlpNew = new WLinePoint((float)x, (float)y, (float)z, color, setNew);
            ++numPoints;

            if (wlpNew == NULL) // safety first
            {
                cerr << "Out of memory error." << endl;
                return;
            }
            if (wlpOld == NULL)
                wlineData = wlpNew; // first node is root of line structure

            if (setOld != setNew) // beginning of new line set?
            {
                wlpNew->newline = 1;
                ++numLines;
            }
            // if distance to last point is too large, start new line
            else if (setOld == setNew && wlpOld != NULL && wlpNew->maxDistance(wlpOld) > 4.5)
            {
                wlpNew->newline = 1;
                ++numLines;
            }

            if (wlpOld != NULL)
                wlpOld->next = wlpNew;
            wlpOld = wlpNew;
        }

        setOld = setNew;
    }
    numWL = setNew + 1;
    cerr << numPoints << " line points read." << endl;
    cerr << numLines << " line fragments read." << endl;
    cerr << numWL << " worldlines in file." << endl;
    return;
}

void ReadWLines::drawLines(float scaleY, float scaleZ)
{
    coDoLines *wlLines = NULL;
    coDoFloat *wlColors = NULL;
    char *coviseObjName = NULL;
    int numVert;
    int *ll, *vl;
    float *x, *y, *z, *col;
    WLinePoint *wlp;
    int pointNr, lineNr;
    int i;

    numVert = numPoints; // each point is also a vertex (no point compression done)

    // Covise-Objekte anfordern:
    coviseObjName = Covise::get_object_name("lines");
    wlLines = new coDoLines(coviseObjName, numPoints, numVert, numLines);
    coviseObjName = Covise::get_object_name("lineColors");
    wlColors = new coDoFloat(coviseObjName, numPoints);

    // Daten-Pointer von Covise anfordern:
    wlLines->getAddresses(&x, &y, &z, &vl, &ll);
    wlColors->getAddress(&col);

    // Enter world lines into covise data structure:
    wlp = wlineData;
    pointNr = 0;
    lineNr = 0;
    ll[0] = 0;

    while (wlp != NULL) // walk through all line points
    {
        if (pointNr >= numPoints)
            return; // safety first
        if (lineNr >= numLines)
            return;

        // Set line coordinates:
        x[pointNr] = wlp->x;
        y[pointNr] = wlp->z * scaleZ;
        z[pointNr] = -wlp->y * scaleY;
        col[pointNr] = wlp->col;
        vl[pointNr] = pointNr;
        if (wlp->newline == 1 && pointNr != 0)
        {
            ++lineNr;
            ll[lineNr] = pointNr; // set line list entry
        }
        wlp = wlp->next;
        ++pointNr;
    }

    // Safety check for covise object values:
    for (i = 0; i < numLines; ++i)
        if (ll[i] < 0 || ll[i] >= numPoints)
            cerr << "Warning: suspicious ll value found: " << ll[i] << endl;

    // Zugriffe auf Covise-Objekte aufraeumen:
    delete wlLines;
    delete wlColors;
}

// If no grid is drawn, polygons must not be visible!
void ReadWLines::drawNoGrid(float maxX, float maxY)
{
    coDoPolygons *polys = NULL; // covise polygons data object
    char *coviseObjName = NULL;
    float *px, *py, *pz; // point coordinates list
    int *pl, *vl; // polygon list and vertex list
    int rects; // number of rectangles
    int i; // counter
    int point = 0; // current point number
    int vertex = 0; // current vertex number
    int polygon = 0; // current polygon number
    int x, y, z; // counters
    int width, height; // numbers of fields on chessboard

    width = (int)maxX;
    height = (int)maxY;

    cout << "Drawing no " << width << " x " << height << " chessboard..." << endl;

    rects = 2 * ((width * height + 1) / 2);
    coviseObjName = Covise::get_object_name("polygons");
    polys = new coDoPolygons(coviseObjName, 2 * (width + 1) * (height + 1), 4 * rects, rects);
    polys->getAddresses(&px, &py, &pz, &vl, &pl);

    cout << "Generating point coordinates for " << 2 * (width + 1) * (height + 1) << " points..." << endl;

    // Generate point coordinates list:
    for (z = 0; z < 2; ++z)
        for (y = 0; y <= height; ++y)
            for (x = 0; x <= width; ++x) // draw all rectangles on one side
            {
                px[point] = 0.0;
                py[point] = 0.0;
                pz[point] = 0.0;
                ++point;
            }

    cout << point << " points generated." << endl;

    // Generate rectangles:
    for (z = 0; z < 2; ++z) // generate chessboards on both ends of worldlines area
    {
        for (i = 0; i < (width * height); i += 2) // i counts chessboard fields
        {
            vl[vertex] = 0;
            ++vertex;
            vl[vertex] = 0;
            ++vertex;
            vl[vertex] = 0;
            ++vertex;
            vl[vertex] = 0;
            ++vertex;
            pl[polygon] = (i == 0 && z == 0) ? 0 : (pl[polygon - 1] + 4);
            ++polygon;
        }
    }

    cout << "Generated " << vertex << " vertices and " << polygon << " polygons." << endl;

    delete polys;
}

// Draw chessboard style coordinates grid
void ReadWLines::drawGrid(float maxX, float maxY, float maxZ, float scaleY, float scaleZ)
{
    coDoPolygons *polys = NULL; // covise polygons data object
    char *coviseObjName = NULL;
    float *px, *py, *pz; // point coordinates list
    int *pl, *vl; // polygon list and vertex list
    int rects; // number of rectangles
    int i; // counter
    int point = 0; // current point number
    int vertex = 0; // current vertex number
    int polygon = 0; // current polygon number
    int x, y, z; // counters
    int col, row; // position within chessboard
    int width, height; // numbers of fields on chessboard

    // flat output image => no polygons can be drawn
    if (maxX < 1.0 || maxY < 1.0)
        drawNoGrid(maxX, maxY);

    width = (int)maxX;
    height = (int)maxY;

    cout << "Drawing " << width << " x " << height << " chessboard..." << endl;

    rects = 2 * ((width * height + 1) / 2);
    coviseObjName = Covise::get_object_name("polygons");
    polys = new coDoPolygons(coviseObjName, 2 * (width + 1) * (height + 1), 4 * rects, rects);
    polys->getAddresses(&px, &py, &pz, &vl, &pl);
    polys->addAttribute("COLOR_BINDING", "OVERALL");
    polys->addAttribute("COLOR", "0");
    polys->addAttribute("vertexOrder", "1"); // light both sides

    cout << "Generating point coordinates for " << 2 * (width + 1) * (height + 1) << " points..." << endl;

    // Generate point coordinates list:
    for (z = 0; z < 2; ++z)
        for (y = 0; y <= height; ++y)
            for (x = 0; x <= width; ++x) // draw all rectangles on one side
            {
                px[point] = x;
                py[point] = (z == 0) ? 0 : (maxZ * scaleZ);
                pz[point] = -y * scaleY;
                ++point;
            }

    cout << point << " points generated." << endl;

    cout << "Generating " << rects << " chessboard fields..." << endl;

    // Generate rectangles:
    for (z = 0; z < 2; ++z) // generate chessboards on both ends of worldlines area
    {
        for (i = 0; i < (width * height); i += 2) // i counts chessboard fields
        {
            col = i % width;
            row = i / width;

            // top left vertex of current chessboard field:
            vl[vertex] = (z * (width + 1) * (height + 1)) + (row * (width + 1) + col);
            ++vertex;

            // top right vertex of current chessboard field:
            vl[vertex] = vl[vertex - 1] + 1;
            ++vertex;

            // bottom right vertex of current chessboard field:
            vl[vertex] = vl[vertex - 1] + width + 1;
            ++vertex;

            // bottom left vertex of current chessboard field:
            vl[vertex] = vl[vertex - 1] - 1;
            ++vertex;

            pl[polygon] = (i == 0 && z == 0) ? 0 : (pl[polygon - 1] + 4);
            ++polygon;
        }
    }

    cout << "Generated " << vertex << " vertices and " << polygon << " polygons." << endl;

    delete polys;
}

// Check world line coordinates for plausibility
void ReadWLines::checkValues()
{
    WLinePoint *wlp;
    int pts = 0; /* number of points */

    wlp = wlineData;
    while (wlp != NULL) // walk through all line points
    {
        if (wlp->x < 0.0 || wlp->x > MAX_X)
            cerr << "Warning: suspicious x value found:" << wlp->x << endl;
        if (wlp->y < 0.0 || wlp->y > MAX_Y)
            cerr << "Warning: suspicious y value found:" << wlp->y << endl;
        if (wlp->z < 0.0 || wlp->z > MAX_Z)
            cerr << "Warning: suspicious z value found:" << wlp->z << endl;
        if (wlp->col < 0.0 || wlp->col > MAX_COL)
            cerr << "Warning: suspicious col value found:" << wlp->col << endl;
        wlp = wlp->next;
        ++pts;
    }
    if (pts != numPoints)
        cerr << "Warning: number of points incorrect" << endl;
}

void ReadWLines::compute(void *)
{
    long int minWL, maxWL, line1, line2, line3;
    float scaleY, scaleZ;
    int grid; // 1 = draw grid, 0 = don't
    char *filename;
    FILE *fp = NULL;
    WLinePoint *wlp;
    float maxX, maxY, maxZ; // maximum coordinates

    wlineData = NULL;

    // get parameters from covise
    Covise::get_scalar_param("minWL", &minWL);
    Covise::get_scalar_param("maxWL", &maxWL);
    Covise::get_scalar_param("line1", &line1);
    Covise::get_scalar_param("line2", &line2);
    Covise::get_scalar_param("line3", &line3);
    Covise::get_scalar_param("scaleY", &scaleY);
    Covise::get_scalar_param("scaleZ", &scaleZ);
    Covise::get_boolean_param("grid", &grid);
    Covise::get_browser_param("wlPath", &filename);

    fp = Covise::fopen(filename, "r");
    if (!fp)
    {
        Covise::sendError("could not open file");
        return;
    }

    // World lines file einlesen
    readWLFile(fp, minWL, maxWL, line1, line2, line3);
    fclose(fp);

    checkValues();

    // Compute maximum coordinates:
    wlp = wlineData;
    maxX = maxY = maxZ = 0.0;
    while (wlp != NULL) // walk through all line points
    {
        if (maxX < wlp->x)
            maxX = wlp->x;
        if (maxY < wlp->y)
            maxY = wlp->y;
        if (maxZ < wlp->z)
            maxZ = wlp->z;
        wlp = wlp->next;
    }

    // Adjust scale variables:
    if (scaleY < 0 && scaleZ < 0) // no scaling given => set default scaling
    {
        // Set scaling to cube shaped output object:
        scaleY = (maxX + 1.0) / (maxY + 1.0);
        scaleZ = (maxX + 1.0) / (maxZ + 1.0);
    }
    else
    {
        if (scaleY < 0.0)
            scaleY = 1.0;
        if (scaleZ < 0.0)
            scaleZ = 1.0;
    }

    if (numLines > 0)
    {
        cerr << "Drawing lines..." << endl;
        drawLines(scaleY, scaleZ); // generate lines from wlines data list
    }
    if (grid == 1)
    {
        cerr << "Drawing grid..." << endl;
        drawGrid(maxX, maxY, maxZ, scaleY, scaleZ);
    }
    else
        drawNoGrid(maxX, maxY);

    cerr << "Freeing memory..." << endl;
    freeWLMemory();

    cerr << "Done." << endl;
    return;
}

// this method gets rid of the allocated memory for the world lines
void ReadWLines::freeWLMemory()
{
    WLinePoint *wlp, *wlpNext;
    int i = 0;

    // Delete world line datasets:
    wlp = wlineData;
    while (wlp != NULL)
    {
        wlpNext = wlp->next;
        delete wlp;
        ++i;
        wlp = wlpNext;
    }
    numPoints = 0;
    numLines = 0;
    cerr << i << " world line points freed." << endl;
}
