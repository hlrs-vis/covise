/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TGENDAT_H)
#define __TGENDAT_H

//=============================================================================
//
// New and improved Tracer-GenDat to validate and profile the tracers
//
// Date: March/2002, lf_te
//
//=============================================================================

#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <api/coModule.h>
using namespace covise;

// Zirkulationsfeld, optional mit komponente nach aussen/innen
// festes Gitter mit variablen Daten, variables Gitter mit variablen Daten
// bewegtes Gitter (z.B. Kolben)
// Labyrinth/gebogenes Gitter

class TGenDat : public coModule
{
private:
    // ports
    coOutputPort *oPGrid, *oPData;

    // parameters
    coChoiceParam *pGridChoice, *pStyleChoice;

    // member functions for the different pGridChoice's
    void rectGrid();
    void strGrid();
    void unstrGrid();

    // returns a random float in the range (0.0 , 1.0)
    float getRand();

    // loads an array with random float values in the range [0.0 , 1.0] which are
    //   sorted by value (ascending)
    void loadRand(int num, float *val);

    // this will generate a rectilinear grid between [x0 y0 z0] and
    //   [x1 y1 z1] with [numX numY numZ] nodes randomly distributed with seed
    coDoRectilinearGrid *genRectGrid(float x0, float y0, float z0,
                                     float x1, float y1, float z1,
                                     int numX, int numY, int numZ, int seed, char *objName);

    // this will generate a structured grid between [x0 y0 z0] and
    //   [x1 y1 z1] with [numX numY numZ] nodes randomly distributed with seed
    coDoStructuredGrid *genStrGrid(float x0, float y0, float z0,
                                   float x1, float y1, float z1,
                                   int numX, int numY, int numZ, int seed, char *objName);

public:
    TGenDat(int argc, char *argv[]);
    virtual ~TGenDat();

    // main-callback
    virtual int compute(const char *port);
};
#endif // __TGENDAT_H
