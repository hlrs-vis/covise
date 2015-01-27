/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *									*
 *	File			DynaSight.cpp 				*
 *									*
 *	Description		DynaSight optical position tracking system interface class				*
 *									*
 *									*
 ************************************************************************/

#if !defined(_WIN32) && !defined(__APPLE__)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#endif

#include <util/common.h>

#include "DynaSight.h"
#include <util/SerialCom.h>
#include <util/byteswap.h>

//#define VERBOSE

//-----------------------------------------------------------------------------

DynaSight::DynaSight(const std::string &serport)
    : serport(serport)
    , serialcom(NULL)
{
    cerr << "new DynaSight: serport=" << serport << endl;
    initialize();

#ifdef HAVE_PTHREAD
    if (pthread_create(&trackerThread, NULL, startThread, this))
    {
        cerr << "failed to create trackerThread: " << strerror(errno) << endl;
    }
#endif
}

void DynaSight::initialize()
{
#ifdef VERBOSE
    fprintf(stderr, "DynaSight::initialize()\n");
#endif
    rawx = rawy = rawz = 0;

    serialcom = new covise::SerialCom(serport.c_str(), 19200);

    int aftersync = 0;
    uint8_t byte;
    while (serialcom->read(&byte, sizeof(byte), 1) == sizeof(byte))
    {
#ifdef VERBOSE
        fprintf(stderr, "dynasight: read byte %x, aftersync=%d\n", (int)byte, aftersync);
#endif
        if ((byte & 0xf0) == 0x80)
            aftersync++;
        else if (aftersync < 2)
            aftersync = 0;
        else
            aftersync++;
        if (aftersync == 8)
            break;
    }

#ifdef VERBOSE
    fprintf(stderr, "DynaSight: aftersync=%d\n", aftersync);
#endif

    if (aftersync != 8)
    {
        delete serialcom;
        serialcom = NULL;

        cerr << "DynaSight initialisation failed" << endl;
    }
}

#ifdef HAVE_PTHREAD
void *DynaSight::startThread(void *th)
{
    DynaSight *t = (DynaSight *)th;
    t->mainLoop();
    return NULL;
}
#endif

DynaSight::~DynaSight()
{
#ifdef HAVE_PTHREAD
    pthread_cancel(trackerThread);
    pthread_join(trackerThread, 0);
#endif
}

void DynaSight::reset()
{
    delete serialcom;
    serialcom = NULL;

    initialize();
}

void
DynaSight::mainLoop()
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
DynaSight::poll()
{
    if (!serialcom)
        return false;

    uint16_t data[4];
    int n = serialcom->read(&data, sizeof(data), 1);
    if (n != sizeof(data))
    {
#ifdef VERBOSE
        fprintf(stderr, "dynasight: read only %d\n", n);
#endif
        return false;
    }

    if ((data[0] & 0xf0f0) != 0x8080)
    {
// out of sync
#ifdef VERBOSE
        fprintf(stderr, "dynasight: out of sync\n");
#endif
        reset();
        return false;
    }

    byteSwap(data, sizeof(data) / sizeof(data[0]));

    int e = (data[0] & 0x0300) >> 8;

    rawx = static_cast<int16_t>(data[1]);
    rawy = static_cast<int16_t>(data[2]);
    rawz = static_cast<int16_t>(data[3]);

    rawx <<= e;
    rawy <<= e;
    rawz <<= e;

#ifdef VERBOSE
    fprintf(stderr, "c=%04x, x=%d, y=%d, z=%d\n", data[0], rawx, rawy, rawz);
#endif

    return true;
}

void DynaSight::getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
#ifdef VERBOSE
    fprintf(stderr, "DynaSight::getPos(station=%d): raw [%d %d %d]\n", station, rawx, rawy, rawz);
#else
    (void)station;
#endif

#ifndef HAVE_PTHREAD
    poll();
#endif

    *x = rawx * 0.05;
    *y = rawy * 0.05;
    *z = rawz * 0.05;

    *m00 = 1.f;
    *m01 = 0.f;
    *m02 = 0.f;

    *m10 = 0.f;
    *m11 = 1.f;
    *m12 = 0.f;

    *m20 = 0.f;
    *m21 = 0.f;
    *m22 = 1.f;
}
