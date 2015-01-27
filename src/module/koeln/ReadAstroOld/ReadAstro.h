/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)2002 RUS  **
**                                                                        **
** Description: Read IMD checkpoint files from ITAP.                      **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                     Juergen Schulze-Doebold                            **
**     High Performance Computing Center University of Stuttgart          **
**                         Allmandring 30                                 **
**                         70550 Stuttgart                                **
**                                                                        **
** Cration Date: 03.09.2002                                               **
\**************************************************************************/

#ifndef _READ_ASTRO_H_
#define _READ_ASTRO_H_

#include <api/coModule.h>
using namespace covise;

class coReadAstro : public coModule
{
private:
    // Ports:
    coOutputPort *poPoints;
    coOutputPort *poSpeed;
    coOutputPort *poMass;
    coOutputPort *poGalaxy;

    // Parameters:
    coFileBrowserParam *pbrCheckpointFile; ///< name of first checkpoint file of a sequence
    coBooleanParam *pboWarnings; ///< true = display warnings when reading a file
    coBooleanParam *pboTimestepSets; ///< true = output time steps as sets
    coIntScalarParam *pisNumStarsGalaxy; ///< number of stars in first galaxy
    coIntScalarParam *pLimit; ///< max no. of timesteps to read
    coBooleanParam *pReorder; ///< intersperse stars of 1st and 2nd galaxay

    // Methods:
    virtual int compute(const char *port);
    virtual float idle();
    virtual void param(const char *name, bool inMapLoading);
    float absVector(float, float, float);
    bool readArray(FILE *fp, float **data, int numElems, bool reorder);

    // Data:
    int timestep;
    char *baseFilename;
    char *currentFilename;

    bool doExec;

public:
    coReadAstro(int argc, char *argv[]);
    bool displayWarnings();
};

#endif
