/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SSD_H_
#define CO_SSD_H_
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
 *	File			SSD.cpp 				*
 *									*
 *	Description		SSD optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/
#define MAXSENSORS 22
#define MAXBODYS 10
#define MAXFLYSTICKS 10
#define MAXBYTES 4000

#include <net/covise_socket.h>
#include <util/coTypes.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <util/coTypes.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#ifdef HAVE_SSD
#include <pvr_ssd.h>
#endif

class INPUT_LEGACY_EXPORT PvrSSD
{
private:
    int connection;
    char *hostname;
    double lastTime;
#ifdef HAVE_SSD
    std::vector<SSDdata> ssdData;
#endif

#ifdef HAVE_PTHREAD
    pthread_t trackerThread;
    static void *startThread(void *);
#endif
    bool poll();
    void mainLoop();
    void initialize();

public:
    PvrSSD(const char *host = NULL);
    ~PvrSSD();
    void getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void reset();
};
#endif
