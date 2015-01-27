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

#ifndef _READ_VOLUME_H
#define _READ_VOLUME_H

#include <api/coModule.h>
using namespace covise;

/** Reader module for Virvo volume files. The volume data will be in the
    range from 0.0 to 1.0.
    Transient data support added on 02-06-18.
    The volume files can be written by WRITE_VOLUME.
    @see coWriteVolume
*/
class coReadVolume : public coModule
{
private:
    static const int MAX_CHANNELS = 8;
    // Ports:
    coOutputPort *poGrid;
    coOutputPort *poVolume[MAX_CHANNELS];

    // Parameters:
    coFileBrowserParam *pbrVolumeFile;
    coIntScalarParam *piSequenceBegin;
    coIntScalarParam *piSequenceEnd;
    coIntScalarParam *piSequenceInc;
    coBooleanParam *pboPreferByteData;
    coBooleanParam *pboCustomSize;
    coFloatParam *pfsVolWidth;
    coFloatParam *pfsVolHeight;
    coFloatParam *pfsVolDepth;
    coBooleanParam *pboReadRaw;
    coBooleanParam *pboReadBS;
    coIntScalarParam *piBPC;
    coIntScalarParam *piChannels;
    coIntScalarParam *piHeader;
    coIntScalarParam *piVoxelsX;
    coIntScalarParam *piVoxelsY;
    coIntScalarParam *piVoxelsZ;
    coIntScalarParam *minValue;
    coIntScalarParam *maxValue;

    // Methods:
    virtual int compute(const char *port);

public:
    coReadVolume(int argc, char *argv[]);
    virtual ~coReadVolume()
    {
    }
};
#endif
