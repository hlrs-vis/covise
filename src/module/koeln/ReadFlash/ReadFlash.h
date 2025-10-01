/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read volume files in formats supported by Virvo.          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 28.10.2000                                               **
\**************************************************************************/

#ifndef _READ_FLASH_H
#define _READ_FLASH_H

#include <api/coModule.h>
using namespace covise;

/** Reader module for Virvo volume files. The volume data will be in the
    range from 0.0 to 1.0.
    Transient data support added on 02-06-18.
    The volume files can be written by WRITE_VOLUME.
    @see coWriteVolume
*/
class coReadFlash : public coModule
{
private:
    static const int MAX_CHANNELS = 8;
    // Ports:
    coOutputPort *poGrid;
    coOutputPort *poVolume[MAX_CHANNELS];

    // Parameters:
    coFileBrowserParam *pbrVolumeFile;
    coIntVectorParam *piSequence;
    // coIntScalarParam *piSequenceBegin;
    // coIntScalarParam *piSequenceEnd;
    // coIntScalarParam *piSequenceInc;

    // Field name
    coStringParam *var_names[MAX_CHANNELS];

    // Region selection
    coBooleanParam *pfSelectRegion;
    coFloatVectorParam *pfRegionMin;
    coFloatVectorParam *pfRegionMax;

    // Limit refinement level
    coIntVectorParam *pfLevels;
    coIntScalarParam *pfMaxLevel;

    // Custom range
    // TODO
    coIntVectorParam *pfDataOpt[MAX_CHANNELS];
    coFloatVectorParam *pfRange[MAX_CHANNELS];

    // Particles
    coBooleanParam *pfGetParticles;
    // Removed for now
    //coIntScalarParam *pfPartProperties;

    // Methods:
    virtual int compute(const char *port);

public:
    coReadFlash(int argc, char *argv[]);
    virtual ~coReadFlash()
    {
    }
};
#endif
