/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "VRPN.h"
#include <util/unixcompat.h>
#include <iostream>

//#define VERBOSE

#ifdef HAVE_VRPN
#include <quat.h>

using namespace std;
void VRPN::vrpnCallback(void *userdata, const vrpn_TRACKERCB t)
{
    VRPN *vrpn = reinterpret_cast<VRPN *>(userdata);

    int station = t.sensor;

    if (station >= vrpn->vrpnData.size())
        vrpn->vrpnData.resize(station + 1);

    vrpn->vrpnData[station] = t;

#ifdef VERBOSE
    cerr << "VRPN: report for sensor " << station << " at " << vrpn->trackerid << endl;
#endif
}
#endif

//-----------------------------------------------------------------------------

VRPN::VRPN(const std::string &host, const std::string &dev)
{
    trackerid = dev + "@" + host;
#ifdef VERBOSE
    cerr << "VRPN: adding " << trackerid << endl;
#endif
    initialize();

#ifdef HAVE_PTHREAD
    if (pthread_create(&trackerThread, NULL, startThread, this))
    {
        cerr << "failed to create trackerThread: " << strerror(errno) << endl;
    }
#endif
}

void VRPN::initialize()
{
#ifdef HAVE_VRPN
    vrpnTracker = new vrpn_Tracker_Remote(trackerid.c_str());
    vrpnTracker->register_change_handler(this, vrpnCallback);
#else
    std::cerr << "VRPN support not available" << std::endl;
#endif
}

#ifdef HAVE_PTHREAD
void *VRPN::startThread(void *th)
{
    VRPN *vrpn = (VRPN *)th;
    vrpn->mainLoop();
    return NULL;
}
#endif

VRPN::~VRPN()
{
#ifdef HAVE_PTHREAD
    pthread_cancel(trackerThread);
    pthread_join(trackerThread, 0);
#endif
}

void VRPN::reset()
{
#ifdef HAVE_VRPN
    delete vrpnTracker;
#endif
    initialize();
}

void
VRPN::mainLoop()
{
    while (1)
    {
        if (!poll())
        {
            usleep(20000);
        }
    }
}

bool
VRPN::poll()
{
#ifdef HAVE_VRPN
    vrpnTracker->mainloop();
    return true;
#else
    return false;
#endif
}

void VRPN::getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
#ifndef HAVE_PTHREAD
    poll();
#endif

#ifdef HAVE_VRPN
#ifdef VERBOSE
    fprintf(stderr, "VRPN: getting data for station %d (num stations=%u)\n", station, (unsigned)vrpnData.size());
#endif

    if (station < vrpnData.size())
    {
        *x = vrpnData[station].pos[0];
        *y = vrpnData[station].pos[1];
        *z = vrpnData[station].pos[2];
#ifdef VERBOSE
        fprintf(stderr, "VRPN: pos %d: (%f %f %f)\n", station, *x, *y, *z);
#endif

        qgl_matrix_type mat;
        q_type quat;
        for (int i = 0; i < 4; ++i)
            quat[i] = vrpnData[station].quat[i];
        qgl_to_matrix(mat, quat);

#if 1
        *m00 = mat[0][0];
        *m01 = mat[0][1];
        *m02 = mat[0][2];
        *m10 = mat[1][0];
        *m11 = mat[1][1];
        *m12 = mat[1][2];
        *m20 = mat[2][0];
        *m21 = mat[2][1];
        *m22 = mat[2][2];
#else
        *m00 = 1.0f;
        *m01 = 0.0f;
        *m02 = 0.0f;
        *m10 = 0.0f;
        *m11 = 1.0f;
        *m12 = 0.0f;
        *m20 = 0.0f;
        *m21 = 0.0f;
        *m22 = 1.0f;
#endif

#ifdef VERBOSE
        fprintf(stderr, "VRPN: mat %d\n(%f %f %f)\n(%f %f %f)\n(%f %f %f)\n\n", station, mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1], mat[2][2]);
#endif
    }
#else
    (void)station;
    (void)x;
    (void)y;
    (void)z;
    (void)m00;
    (void)m01;
    (void)m02;
    (void)m10;
    (void)m11;
    (void)m12;
    (void)m20;
    (void)m21;
    (void)m22;
#endif
}
