/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DYNASIGHT_H_
#define CO_DYNASIGHT_H_
/************************************************************************
 *									*
 *	File			DynaSight.cpp 				*
 *									*
 *	Description		DynaSight optical position tracking system interface class				*
 *									*
 ************************************************************************/

#include <util/coTypes.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <util/coTypes.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

namespace covise
{
class SerialCom;
}

class INPUT_LEGACY_EXPORT DynaSight
{
private:
    std::string serport;
    covise::SerialCom *serialcom;
    double lastTime;

#ifdef HAVE_PTHREAD
    pthread_t trackerThread;
    static void *startThread(void *);
#endif
    bool poll();
    void mainLoop();
    void initialize();
    int rawx, rawy, rawz;

public:
    DynaSight(const std::string &serport);
    ~DynaSight();
    void getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void reset();
};
#endif
