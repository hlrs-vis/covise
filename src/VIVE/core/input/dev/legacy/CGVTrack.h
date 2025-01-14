/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_CGVTRACK_H_
#define _CO_CGVTRACK_H_
/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			CGVTrack.cpp 				*
 *									*
 *	Description		CGVTrack optical tracking system interface class				*
 *									*
 *	Author			DUwe Woessner				*
 *									*
 *	Date			July 2007				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/coTypes.h>
#include "UDPClient.h"
#include "Marker3DList.h"
#include "RecognizedTargetList.h"
#include <sstream>
#include <ios>
#include <osg/Matrix>
#ifndef _WIN32
#include <unistd.h>
#endif

/************************************************************************
 *									*
 *	Description		class CGVTrack				*
 *									*
 *									*
 *	Author			D. Rainer				*
 *									*
 ************************************************************************/
#define CGVMAXSENSORS 5

class stationDataType
{
public:
    osg::Matrix mat;
    char *name;
};

class INPUT_LEGACY_EXPORT CGVTrack
{
private:
    int port;
    stationDataType *stationData;
    UDPClient client;

    CGVOpticalTracking::Marker3DList vMarker;
    CGVOpticalTracking::RecognizedTargetList vTarget;
    void mainLoop();
    void receiveData();

    static void *continuousThread(void *data);

public:
    CGVTrack(const char *host, int port);
    ~CGVTrack();
    void getPositionMatrix(int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void getButtons(int station, unsigned int *status); // get the bitmask of the first 32 buttons
    bool getButton(int station, int buttonNumber); // get any button status
};
#endif
