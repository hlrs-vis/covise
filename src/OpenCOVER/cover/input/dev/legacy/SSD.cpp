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

#if !defined(_WIN32) && !defined(__APPLE__)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#endif

#include <util/common.h>

#include <net/covise_socket.h>
#include <net/covise_host.h>
#include "SSD.h"

//#define VERBOSE

//-----------------------------------------------------------------------------

PvrSSD::PvrSSD(const char *h)
{
    if (h)
    {
        hostname = new char[strlen(h) + 1];
        strcpy(hostname, h);
    }
    else
    {
        hostname = new char[1];
        strcpy(hostname, "");
    }

    initialize();

#ifdef HAVE_PTHREAD
    if (pthread_create(&trackerThread, NULL, startThread, this))
    {
        cerr << "failed to create trackerThread: " << strerror(errno) << endl;
    }
#endif
}

void PvrSSD::initialize()
{
#ifdef HAVE_SSD
    connection = ssd_connect_server(hostname);
    if (!connection)
    {
        cerr << "failed to create ssd connection" << endl;
        return;
    }
    for (int i = 0; i < SSD_MAXDEVICES; i++)
        if (!ssd_enable_devices(SSD_SENSOR, i))
        {
            cerr << "failed to enable ssd sensor i" << endl;
            return;
        }
#else
    cerr << "SSD support not available" << endl;
#endif
}

#ifdef HAVE_PTHREAD
void *PvrSSD::startThread(void *th)
{
    PvrSSD *t = (PvrSSD *)th;
    t->mainLoop();
    return NULL;
}
#endif

PvrSSD::~PvrSSD()
{
#ifdef HAVE_PTHREAD
    pthread_cancel(trackerThread);
    pthread_join(trackerThread, 0);
#endif

#ifdef HAVE_SSD
// this blocks
//ssd_disconnect_server();
#endif
    delete[] hostname;
}

void PvrSSD::reset()
{
#ifdef HAVE_SSD
    ssd_disconnect_server();
#endif
    initialize();
}

void
PvrSSD::mainLoop()
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
PvrSSD::poll()
{
#ifdef HAVE_SSD
    if (!connection)
        return false;

    if (!ssd_report_exists())
        return false;

    SSDreport e;
    ssd_get_report(&e);
    for (int i = 0; i < e.nvalues; i++)
    {
        SSDdata d;
        ssd_get_data(&d);
        int j = -1;
        for (j = 0; j < ssdData.size(); j++)
        {
            if (ssdData[j].idx == d.idx)
            {
                memcpy(&ssdData[j], &d, sizeof(d));
                break;
            }
        }
        if (j < 0 || j >= ssdData.size())
        {
            ssdData.push_back(d);
            //memcpy(&ssdData[ssdData.size()-1], &d, sizeof(d));
        }
#ifdef VERBOSE
        fprintf(stderr, "i = %d, device = %d, idx = %d\n", i, ssdData[i].device, ssdData[i].idx);
#endif
#if 0
      if (ssdData[i].device == SSD_SENSOR)
      {
         for (int ii = 0; ii < 4; ii++)
         {
            for (int jj = 0; jj < 4; jj++)
               fprintf (stderr, "%.1f ", ssdData[i].val.sdata[ii][jj]);

            fprintf (stderr, "\n");
         }
      }
      fprintf (stderr, "-----------------------\n");
#endif
    }
    //ssd_ack (e.timestamp);

    return true;
#else
    return false;
#endif
}

void PvrSSD::getPositionMatrix(unsigned int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
#ifndef HAVE_PTHREAD
    poll();
#endif

#ifdef HAVE_SSD
#ifdef VERBOSE
//fprintf(stderr, "SSD: getting data for station %d (num stations=%u)\n", station, (unsigned)ssdData.size());
#endif
    int index = -1;
    for (int i = 0; i < ssdData.size(); i++)
    {
        if (ssdData[i].idx == station)
        {
            index = i;
            break;
        }
    }
    if (index < 0)
        return;

    *x = ssdData[index].val.sdata[3][0] * 1000.;
    *y = ssdData[index].val.sdata[3][1] * 1000.;
    *z = ssdData[index].val.sdata[3][2] * 1000.;

    *m00 = ssdData[index].val.sdata[0][0];
    *m01 = ssdData[index].val.sdata[0][1];
    *m02 = ssdData[index].val.sdata[0][2];

    *m10 = ssdData[index].val.sdata[1][0];
    *m11 = ssdData[index].val.sdata[1][1];
    *m12 = ssdData[index].val.sdata[1][2];

    *m20 = ssdData[index].val.sdata[2][0];
    *m21 = ssdData[index].val.sdata[2][1];
    *m22 = ssdData[index].val.sdata[2][2];
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
