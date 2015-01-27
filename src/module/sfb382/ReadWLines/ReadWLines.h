/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__READWLINES_H)
#define __READWLINES_H

#include <appl/ApplInterface.h>
using namespace covise;

class ReadWLines;

#define LINECOLOR 0.0 // default line color (blue)
#define LINECOLOR1 0.4 // color of particle line 1 (purple)
#define LINECOLOR2 0.6 // color of particle line 2 (orange)
#define LINECOLOR3 1.0 // color of particle line 3 (yellow)

class WLinePoint // a point as part of a line
{
public:
    float x, y, z; // 3D line point coordinate
    float col; // line color
    int set; // line set number
    int newline; // 1 if point starts new line, 0 else
    WLinePoint *next; // pointer to next point

    WLinePoint(float xIn, float yIn, float zIn, float c, int i)
    {
        x = xIn;
        y = yIn;
        z = zIn;
        col = c;
        set = i;
        newline = 0;
        next = NULL;
    }

    // returns maximum distance of the points in the same dimension respectively
    float maxDistance(WLinePoint *wlp)
    {
        return MAX(fabs(x - wlp->x), fabs(y - wlp->y));
    }
};

class ReadWLines
{

private:
    WLinePoint *wlineData; // world lines coordinates
    int numPoints; // number of points in world lines
    int numLines; // number of line fragments
    int numWL; // number of world lines

    //  member functions:
    void compute(void *);
    void readFile(FILE *);
    void quit(void *);
    int openFile();
    void readWLFile(FILE *, int, int, int, int, int);
    void drawLines(float, float);
    void drawNoGrid(float, float);
    void drawGrid(float, float, float, float, float);
    void checkValues();
    int getWord(char *, FILE *);
    int getNextDataset(int *, int *, int *, int *, FILE *);
    void freeWLMemory();

    //  Static callback stubs:
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

public:
    ReadWLines(int argc, char *argv[]);
    ~ReadWLines(){};

    void run()
    {
        Covise::main_loop();
    }
};
#endif
