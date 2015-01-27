/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __COSIOREADLINES_H
#define __COSIOREADLINES_H

#include <api/coModule.h>
using namespace covise;

class coParsedLine
{
public:
    float x, y, z; // line point coordinate
    int color; // color index
    int type; // -1=EOF, 0=syntax error, 1=new frame, 2=new line, 3=point coordinates
};

class coPoint // a point as part of a linked list, with a color and line affiliation
{
public:
    float x, y, z; // 3D point coordinate
    float color; // point (=line) color
    int line; // line number (first line=0)
    coPoint *next; // pointer to next point

    coPoint(float xi, float yi, float zi, float ci, int li)
    {
        x = xi;
        y = yi;
        z = zi;
        color = ci;
        line = li;
        next = NULL;
    }

    void print(); // print list for debugging
};

class coReadLines : public coModule
{
#define MAX_TIMESTEPS 50
private:
    coFileBrowserParam *p_filename; // file name of ASCII lines file
    coOutputPort *p_lines; // output port: line coordinates
    coOutputPort *p_colors; // output port: line colors
    coPoint *lineData[MAX_TIMESTEPS]; // line coordinates
    int numPoints[MAX_TIMESTEPS]; // number of points making up the lines
    int numLines[MAX_TIMESTEPS]; // number of lines
    int numTimesteps; // number of time steps

    int compute();
    void quit();
    bool parseLine(FILE *, coParsedLine *);
    bool readFile(FILE *);
    void makeOutputData();
    void freeMemory();

public:
    coReadLines();
    ~coReadLines();
};
#endif
