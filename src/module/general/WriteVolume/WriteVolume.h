/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Write volume files in formats supported by Virvo.         **
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
 ** Cration Date: 26.06.02                                                 **
\**************************************************************************/

#ifndef _WRITE_VOLUME_H
#define _WRITE_VOLUME_H

#include <api/coModule.h>
using namespace covise;

#define MAX_CHANNELS 8

/** Write volume files in formats supported by Virvo.
  They can be read by the module READ_VOLUME.
  @see coReadVolume
*/
class coWriteVolume : public coModule
{
private:
    // Ports:
    coInputPort *piGrid; ///< input port for uniform grid
    coInputPort *piChan[MAX_CHANNELS]; ///< input port for scalar volume data

    // Parameters:
    coFileBrowserParam *pbrVolumeFile; ///< file name of Virvo output file
    coChoiceParam *pchFileType; ///< Virvo file type (rvf, xvf, ...)
    coBooleanParam *pboOverwrite; ///< true to overwrite existing files
    coChoiceParam *pchDataFormat; ///< volume data format (bytes per voxel)
    coFloatParam *pfsMin; ///< minimum float data value
    coFloatParam *pfsMax; ///< maximum float data value

    // Attributes:
    vvVolDesc *vd; ///< volume description
    int volSize[3]; ///< volume size [voxels]

    // Methods:
    virtual int compute(const char *port);
    bool getGridSize(const coDistributedObject *, int *, float *, float *);
    void getTimestepData(const coDistributedObject **, int no_channels);

public:
    coWriteVolume(int argc, char *argv[]);
};
#endif
