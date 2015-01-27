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
    float x, y, z; ///< line point coordinate
    int type; ///< -1=EOF, 0=syntax error, 1=new frame, 2=new line, 3=point coordinates
};

class coPoint // a point as part of a linked list, with a color and line affiliation
{
public:
    float x, y, z; ///< 3D point coordinate
    int line; ///< line number (first line=0)
    coPoint *next; ///< pointer to next point

    coPoint(float xi, float yi, float zi, int li)
    {
        x = xi;
        y = yi;
        z = zi;
        line = li;
        next = NULL;
    }

    void print(); // print list for debugging
};

class coReadLines : public coModule
{
private:
    coFileBrowserParam *_pbrFilename; ///< file name of ASCII lines file
    coFloatParam *_pfsXOffset; ///< value to offset x with
    coFloatParam *_pfsYOffset; ///< value to offset y with
    coFloatParam *_pfsZFactor; ///< factor to multiply z coordinate with
    coOutputPort *_poLines; ///< output port: line coordinates
    coOutputPort *_poColors; ///< output port: line colors
    coPoint *_lineData; ///< line coordinates
    int _numPoints; ///< number of points making up the lines
    int _numLines; ///< number of lines
    float _color; ///< color of lines; all lines have the same color

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
