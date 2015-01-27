/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __COREADLINES_H
#define __COREADLINES_H

#include <api/coModule.h>
using namespace covise;
#include <virvo/vvarray.h>
#include <virvo/vvtokenizer.h>

/**
  Reader module for 2D world lines as created by Catia Lavalle
  from the group of Prof. Muramatsu (SFB 382, project A8).
  @author Juergen Schulze
*/
class ReadWLines2D : public coModule
{
private:
    enum DirectionType ///< direction of line movement on grid point
    {
        NO_LINE = 0,
        LEFT = 1,
        STAY = 2,
        RIGHT = 3
    };
    coFileBrowserParam *p_filename; ///< file name of ASCII lines file
    coFloatParam *p_cylHeight; ///< height of cylinder
    coFloatParam *p_cylDiameter; ///< diameter of cylinder
    coBooleanParam *p_useCylinder; ///< true = output on cylinder, false = output on 2D grid
    coFloatParam *p_normalColor; ///< color for regular lines
    coFloatParam *p_boundColor; ///< color for lines crossing the boundary
    coFloatParam *p_edgeColor; ///< color for data set edges
    coOutputPort *p_lines; ///< output port: line coordinates
    coOutputPort *p_colors; ///< output port: line colors
    coOutputPort *p_edgeLines; ///< output port: edge line coordinates
    coOutputPort *p_edgeColors; ///< output port: edge line colors

    int numTimesteps; ///< number of timesteps
    int gridSize[2]; ///< number of line end points of simulation grid in x and y direction
    unsigned char *dataGrid; ///< information about line endpoints

    coDoLines **doLines;
    coDoFloat **doColors;

public:
    ReadWLines2D();
    ~ReadWLines2D();
    bool parseHeader(vvTokenizer *);
    bool parseLines(vvTokenizer *);
    void prepareLines();
    void convertCoordinates(int, int, float *, float *, float *);
    void assembleLines();
    void finishLines();
    int getNumLines(int);
    void createEdgeLines();
    int compute();
};
#endif

// EOF
