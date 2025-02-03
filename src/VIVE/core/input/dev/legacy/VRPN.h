/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VRPN_H_
#define CO_VRPN_H_
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
 *	File			VRPN.cpp 				*
 *									*
 *	Description		VRPN optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#ifdef HAVE_VRPN
#define _TIMEZONE_DEFINED
#include <vrpn_Tracker.h>
#endif

#include <vector>
#include <string>
#include <util/coExport.h>

class INPUT_LEGACY_EXPORT VRPN
{
private:
#ifdef HAVE_VRPN
    std::vector<vrpn_TRACKERCB> vrpnData;
    vrpn_Tracker_Remote *vrpnTracker;
#endif
    std::string trackerid;

#ifdef HAVE_PTHREAD
    pthread_t trackerThread;
    static void *startThread(void *);
#endif
    bool poll();
    void mainLoop();
    void initialize();
    bool addTracker(const std::string &host, const std::string &device);

#ifdef HAVE_VRPN
    static void vrpnCallback(void *thisclass, const vrpn_TRACKERCB t);
#endif

public:
    VRPN(const std::string &host, const std::string &device);
    ~VRPN();
    void getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void reset();
};
#endif
